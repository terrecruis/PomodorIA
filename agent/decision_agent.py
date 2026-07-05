"""
agent/decision_agent.py — Agente Decisionale (IoTWF Livello 6 — Application)

Implementa un agente razionale basato su modello + obiettivi secondo
il modello PEAS definito nella Sezione 3.3 del README:

    Performance  → diagnosi corrette, risparmio idrico, riduzione falsi allarmi
    Environment  → serra simulata (parzialmente osservabile, dinamica)
    Actuators    → pompa irrigazione, ventola, LED allarme, notifica
    Sensors      → fotocamera virtuale, sensori ambientali simulati

Tipologia agente: REATTIVO BASATO SU MODELLO + BASATO SU OBIETTIVI
    - Mantiene stato interno (contatore rilevamenti consecutivi,
      storico predizioni)
    - Regole condizione → azione (priorità decrescente):
        1. Confidenza bassa → human-in-the-loop
        2. Malattia virale → allarme alta urgenza
        3. Malattia fungina + umidità alta → ventilazione
        4. Soil moisture bassa → irrigazione
        5. Temperatura alta → ventilazione
        6. Healthy → disattiva tutto

Classi patologiche per categoria (da README §3.3):
    FUNGINE:  Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot,
              Target_Spot
    VIRALI:   Tomato_Yellow_Leaf_Curl_Virus, Tomato_mosaic_virus
    PARASSITI: Spider_mites
    BATTERICHE: Bacterial_spot
    SANA:     healthy
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from sensors.environment_simulator import EnvironmentReading
from edge.inference_engine import InferenceResult
from actuators.actuators import ActuatorBank, ActionResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Classificazione delle malattie per categoria
# ══════════════════════════════════════════════════════════════

# Malattie fungine: beneficiano di riduzione umidità + ventilazione
FUNGAL_DISEASES = {
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Target_Spot",
}

# Malattie virali: spesso trasmesse da insetti → alta urgenza
VIRAL_DISEASES = {
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
}

# Malattie da parassiti
PEST_DISEASES = {
    "Tomato___Spider_mites Two-spotted_spider_mite",
}

# Malattie batteriche
BACTERIAL_DISEASES = {
    "Tomato___Bacterial_spot",
}

HEALTHY_CLASS = "Tomato___healthy"

ALL_DISEASES = FUNGAL_DISEASES | VIRAL_DISEASES | PEST_DISEASES | BACTERIAL_DISEASES


# ══════════════════════════════════════════════════════════════
# Dataclass per il risultato della decisione dell'agente
# ══════════════════════════════════════════════════════════════
@dataclass
class AgentDecision:
    """
    Raccoglie tutte le azioni decise dall'agente per un ciclo.

    Campi:
        actions:            lista di ActionResult per ogni attuatore coinvolto
        reasoning:          lista di stringhe che spiegano la logica decisionale
        disease_category:   "healthy" | "fungal" | "viral" | "bacterial" | "pest" | "unknown"
        consecutive_alerts: contatore corrente dei rilevamenti malattia consecutivi
        timestamp:          epoch time della decisione
    """
    actions: list[ActionResult]
    reasoning: list[str]
    disease_category: str
    consecutive_alerts: int
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Stringa riassuntiva human-readable della decisione."""
        if not self.actions:
            return "Nessuna azione"
        return " | ".join(
            f"{a.actuator_name.replace('Actuator', '')}:{a.action}"
            for a in self.actions
            if a.action != "none"
        ) or "Nessuna azione"

    def __str__(self) -> str:
        return (
            f"AgentDecision("
            f"cat={self.disease_category}, "
            f"consecutive={self.consecutive_alerts}, "
            f"actions=[{self.summary()}])"
        )


# ══════════════════════════════════════════════════════════════
# Agente Decisionale
# ══════════════════════════════════════════════════════════════
class DecisionAgent:
    """
    Agente razionale basato su modello per la gestione della serra.

    Mantiene stato interno tra un ciclo e l'altro:
        - _consecutive_disease_count: quanti cicli consecutivi con malattia
        - _last_predictions: deque degli ultimi N risultati (finestra scorrevole)
        - _action_history: storico delle azioni eseguite

    Le soglie operative sono lette dal config.yaml (sezione "agent").
    """

    def __init__(self, config: dict, actuators: ActuatorBank):
        """
        Args:
            config:    dizionario di configurazione (config.yaml)
            actuators: banco degli attuatori simulati
        """
        self.config   = config
        self.actuators = actuators
        self._load_thresholds()

        # ── Stato interno ────────────────────────────────────
        self._consecutive_disease_count: int = 0
        self._last_predictions: deque = deque(maxlen=10)  # finestra scorrevole
        self._action_history: list[AgentDecision] = []
        self._total_cycles: int = 0
        self._correct_decisions: int = 0   # per metriche di comportamento

        logger.info(
            "DecisionAgent inizializzato | soglie: "
            f"conf={self._conf_threshold:.0%}, "
            f"hum={self._humidity_high}%, "
            f"soil={self._soil_low}%, "
            f"consecutive={self._consecutive_threshold}"
        )

    def _load_thresholds(self) -> None:
        """Carica le soglie operative dal config.yaml."""
        cfg = self.config.get("agent", {})
        self._conf_threshold         = cfg.get("confidence_threshold", 0.70)
        self._humidity_high          = cfg.get("humidity_high_threshold", 80.0)
        self._soil_low               = cfg.get("soil_moisture_low_threshold", 30.0)
        self._consecutive_threshold  = cfg.get("consecutive_alerts_threshold", 3)
        self._temp_high              = cfg.get("temperature_high", 35.0)
        self._temp_low               = cfg.get("temperature_low", 10.0)
        self._light_low              = cfg.get("light_low_threshold", 2000.0)

    # ─────────────────────────────────────────────────────────────
    # Metodo principale: decide le azioni per un ciclo
    # ─────────────────────────────────────────────────────────────

    def decide(
        self,
        inference: InferenceResult,
        env: EnvironmentReading,
        true_label: Optional[str] = None,
    ) -> AgentDecision:
        """
        Esegue il ragionamento condizione → azione per un ciclo.

        Regole in ordine di priorità decrescente:
            1. Confidenza < soglia → human-in-the-loop (alarm + notifica)
            2. Malattia virale → alarm CRITICAL (possibile vettore insetto)
            3. Malattia fungina + umidità alta → ventilazione + notifica
            4. Malattia batterica/parassita → notifica WARNING
            5. Soil moisture bassa → irrigazione
            6. Temperatura alta → ventilazione (indipendente dalla malattia)
            7. N rilevamenti consecutivi → alarm (epidemia simulata)
            8. Healthy → disattiva tutto + notifica OK se si esce da malattia

        Args:
            inference:  risultato dell'inferenza CNN (classe, confidenza, ...)
            env:        lettura dei sensori ambientali
            true_label: (opzionale) classe reale, per metriche di comportamento

        Returns:
            AgentDecision con lista di ActionResult e spiegazione del ragionamento
        """
        self._total_cycles += 1
        self._last_predictions.append(inference.predicted_label)

        actions: list[ActionResult] = []
        reasoning: list[str] = []
        label = inference.predicted_label
        conf  = inference.confidence

        # Classifica la malattia predetta
        disease_category = self._classify_disease(label)

        # ══════════════════════════════════════════════════════
        # REGOLA 1: Confidenza bassa → human-in-the-loop
        # ══════════════════════════════════════════════════════
        if inference.is_low_confidence(self._conf_threshold):
            reasoning.append(
                f"⚠️  Conf={conf:.1%} < soglia={self._conf_threshold:.0%} "
                f"→ predizione inaffidabile, richiesta ispezione umana"
            )
            actions.append(self.actuators.alarm.activate(
                f"confidenza modello bassa ({conf:.1%}) su '{label}'"
            ))
            actions.append(self.actuators.notification.send(
                f"Ispezione umana richiesta: CNN poco sicura ({conf:.1%}) "
                f"su immagine classificata come '{label}'",
                severity="WARNING",
            ))
            # Anche con bassa confidenza, gestiamo i sensori ambientali
            actions += self._handle_environmental(env, reasoning)
            self._update_consecutive(label, actions, reasoning)
            return self._make_decision(actions, reasoning, disease_category)

        # ══════════════════════════════════════════════════════
        # REGOLA 2: Malattia virale → allarme CRITICAL
        # ══════════════════════════════════════════════════════
        if label in VIRAL_DISEASES:
            self._consecutive_disease_count += 1
            reasoning.append(
                f"🦠 Malattia VIRALE rilevata: '{label}' "
                f"(conf={conf:.1%}) → possibile vettore insetto"
            )
            actions.append(self.actuators.alarm.activate(
                f"malattia virale: {label}"
            ))
            actions.append(self.actuators.notification.send(
                f"MALATTIA VIRALE: '{label}' (conf {conf:.1%}). "
                "Verificare presenza di insetti vettori (afidi, mosche bianche). "
                "Considerare trattamento insetticida.",
                severity="CRITICAL",
            ))
            actions += self._handle_environmental(env, reasoning)
            return self._make_decision(actions, reasoning, disease_category)

        # ══════════════════════════════════════════════════════
        # REGOLA 3: Malattia fungina + umidità alta → ventilazione
        # ══════════════════════════════════════════════════════
        if label in FUNGAL_DISEASES:
            self._consecutive_disease_count += 1
            reasoning.append(
                f"🍄 Malattia FUNGINA: '{label}' (conf={conf:.1%})"
            )
            if env.humidity_pct > self._humidity_high:
                reasoning.append(
                    f"   Umidità {env.humidity_pct:.1f}% > soglia {self._humidity_high}% "
                    f"→ condizioni favorevoli alla diffusione → ventilazione attivata"
                )
                actions.append(self.actuators.ventilation.activate(
                    f"malattia fungina '{label}' + umidità alta ({env.humidity_pct:.0f}%)"
                ))
                actions.append(self.actuators.notification.send(
                    f"Malattia fungina '{label}' con umidità alta ({env.humidity_pct:.0f}%). "
                    "Ventilazione attivata per ridurre umidità.",
                    severity="WARNING",
                ))
            else:
                reasoning.append(
                    f"   Umidità {env.humidity_pct:.1f}% nella norma, "
                    "solo notifica"
                )
                actions.append(self.actuators.notification.send(
                    f"Malattia fungina rilevata: '{label}' (conf {conf:.1%}). "
                    "Umidità nella norma, monitorare.",
                    severity="WARNING",
                ))

        # ══════════════════════════════════════════════════════
        # REGOLA 4: Malattia batterica o parassiti → notifica
        # ══════════════════════════════════════════════════════
        elif label in BACTERIAL_DISEASES | PEST_DISEASES:
            self._consecutive_disease_count += 1
            cat = "BATTERICA" if label in BACTERIAL_DISEASES else "PARASSITA"
            reasoning.append(f"🦠 Malattia {cat}: '{label}' (conf={conf:.1%})")
            actions.append(self.actuators.notification.send(
                f"Malattia {cat.lower()} rilevata: '{label}' (conf {conf:.1%}). "
                "Valutare trattamento specifico.",
                severity="WARNING",
            ))

        # ══════════════════════════════════════════════════════
        # REGOLA 8: Healthy → disattiva attuatori patologia
        # ══════════════════════════════════════════════════════
        elif label == HEALTHY_CLASS:
            if self._consecutive_disease_count > 0:
                reasoning.append(
                    f"✅ Pianta SANA dopo {self._consecutive_disease_count} "
                    "rilevamenti malattia → reset attuatori"
                )
                actions.append(self.actuators.alarm.deactivate("pianta tornata sana"))
                actions.append(self.actuators.ventilation.deactivate("pianta sana"))
                actions.append(self.actuators.notification.send(
                    "Pianta tornata in stato SANO. Attuatori di emergenza disattivati.",
                    severity="INFO",
                ))
            else:
                reasoning.append(f"✅ Pianta SANA (conf={conf:.1%}) → nessuna azione")
            self._consecutive_disease_count = 0

        # ══════════════════════════════════════════════════════
        # Gestione sensori ambientali (parallela alla diagnosi)
        # ══════════════════════════════════════════════════════
        actions += self._handle_environmental(env, reasoning)

        # ══════════════════════════════════════════════════════
        # REGOLA 7: N rilevamenti consecutivi → allarme epidemia
        # ══════════════════════════════════════════════════════
        if self._consecutive_disease_count >= self._consecutive_threshold:
            reasoning.append(
                f"🚨 {self._consecutive_disease_count} rilevamenti malattia "
                f"consecutivi (soglia={self._consecutive_threshold}) → "
                "ALLARME EPIDEMIA"
            )
            actions.append(self.actuators.alarm.activate(
                f"epidemia simulata: {self._consecutive_disease_count} "
                f"rilevamenti consecutivi di malattia"
            ))
            actions.append(self.actuators.notification.send(
                f"ATTENZIONE: {self._consecutive_disease_count} rilevamenti "
                f"consecutivi di malattia. Ultima: '{label}'. "
                "Intervento urgente richiesto.",
                severity="CRITICAL",
            ))

        # Aggiorna metriche di comportamento
        if true_label is not None:
            is_correct = (label == true_label) or (
                label in FUNGAL_DISEASES and true_label in FUNGAL_DISEASES
            )
            if is_correct:
                self._correct_decisions += 1

        decision = self._make_decision(actions, reasoning, disease_category)
        self._action_history.append(decision)
        return decision

    # ─────────────────────────────────────────────────────────────
    # Gestione sensori ambientali (indipendente dalla diagnosi)
    # ─────────────────────────────────────────────────────────────

    def _handle_environmental(
        self,
        env: EnvironmentReading,
        reasoning: list[str],
    ) -> list[ActionResult]:
        """
        Gestisce le condizioni ambientali indipendentemente dalla diagnosi CNN.

        Controlla: soil moisture, temperatura, luminosità.
        """
        actions = []

        # Irrigazione: soil moisture bassa
        if env.soil_moisture_pct < self._soil_low:
            reasoning.append(
                f"💧 Soil moisture {env.soil_moisture_pct:.1f}% < "
                f"soglia {self._soil_low}% → irrigazione attivata"
            )
            actions.append(self.actuators.irrigation.activate(
                f"umidità suolo bassa ({env.soil_moisture_pct:.0f}%)"
            ))
        else:
            if self.actuators.irrigation.is_active:
                reasoning.append(
                    f"💧 Soil moisture {env.soil_moisture_pct:.1f}% nella norma "
                    "→ irrigazione disattivata"
                )
                actions.append(self.actuators.irrigation.deactivate(
                    f"umidità suolo {env.soil_moisture_pct:.0f}% nella norma"
                ))

        # Ventilazione: temperatura alta
        if env.temperature_c > self._temp_high:
            reasoning.append(
                f"🌡️  Temperatura {env.temperature_c:.1f}°C > "
                f"soglia {self._temp_high}°C → ventilazione attivata"
            )
            actions.append(self.actuators.ventilation.activate(
                f"temperatura alta ({env.temperature_c:.1f}°C)"
            ))

        # Alert: temperatura bassa
        if env.temperature_c < self._temp_low:
            reasoning.append(
                f"❄️  Temperatura {env.temperature_c:.1f}°C < "
                f"soglia {self._temp_low}°C → alert freddo"
            )
            actions.append(self.actuators.notification.send(
                f"Temperatura troppo bassa: {env.temperature_c:.1f}°C. "
                "Verificare sistema di riscaldamento.",
                severity="WARNING",
            ))

        # Alert: luce scarsa
        if env.light_lux < self._light_low:
            reasoning.append(
                f"💡 Luminosità {env.light_lux:.0f} lux < "
                f"soglia {self._light_low:.0f} lux → alert luce"
            )
            actions.append(self.actuators.notification.send(
                f"Luminosità scarsa: {env.light_lux:.0f} lux. "
                "Considerare illuminazione supplementare.",
                severity="INFO",
            ))

        return actions

    def _update_consecutive(
        self,
        label: str,
        actions: list,
        reasoning: list,
    ) -> None:
        """Aggiorna il contatore dei rilevamenti consecutivi."""
        if label != HEALTHY_CLASS:
            self._consecutive_disease_count += 1
        else:
            self._consecutive_disease_count = 0

    @staticmethod
    def _classify_disease(label: str) -> str:
        """Restituisce la categoria della malattia."""
        if label == HEALTHY_CLASS:
            return "healthy"
        if label in FUNGAL_DISEASES:
            return "fungal"
        if label in VIRAL_DISEASES:
            return "viral"
        if label in BACTERIAL_DISEASES:
            return "bacterial"
        if label in PEST_DISEASES:
            return "pest"
        return "unknown"

    def _make_decision(
        self,
        actions: list[ActionResult],
        reasoning: list[str],
        disease_category: str,
    ) -> AgentDecision:
        """Factory per AgentDecision."""
        return AgentDecision(
            actions=actions,
            reasoning=reasoning,
            disease_category=disease_category,
            consecutive_alerts=self._consecutive_disease_count,
        )

    # ─────────────────────────────────────────────────────────────
    # Statistiche e stato dell'agente
    # ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Restituisce le statistiche di comportamento dell'agente."""
        return {
            "total_cycles": self._total_cycles,
            "consecutive_disease_count": self._consecutive_disease_count,
            "correct_decisions": self._correct_decisions,
            "behavior_accuracy": (
                self._correct_decisions / self._total_cycles
                if self._total_cycles > 0 else 0.0
            ),
            "last_predictions": list(self._last_predictions),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DecisionAgent("
            f"cycles={stats['total_cycles']}, "
            f"consecutive={stats['consecutive_disease_count']}, "
            f"behavior_acc={stats['behavior_accuracy']:.1%})"
        )
