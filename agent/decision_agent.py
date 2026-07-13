"""agent/decision_agent.py — Agente Decisionale (IoTWF Livello 6 — Application)."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque, Counter

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


def _friendly_label(label: str) -> str:
    """Nome leggibile per i messaggi (es. 'Tomato___Early_blight' -> 'Early blight')."""
    return label.replace("Tomato___", "").replace("_", " ")


# ══════════════════════════════════════════════════════════════
# Dataclass per il risultato della decisione dell'agente
# ══════════════════════════════════════════════════════════════
@dataclass
class AgentDecision:
    """Azioni decise dall'agente per un ciclo, con la motivazione (reasoning)."""
    actions: list[ActionResult]
    reasoning: list[str]
    disease_category: str
    consecutive_alerts: int
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
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
    """Agente razionale basato su modello per la gestione della serra."""

    def __init__(self, config: dict, actuators: ActuatorBank):
        self.config = config
        self.actuators = actuators
        self._load_thresholds()

        # ── Stato interno ────────────────────────────────────
        self._consecutive_disease_count: int = 0
        self._last_predictions: deque = deque(maxlen=10)
        # Finestra scorrevole delle categorie di malattia, usata per rilevare
        # un'epidemia come alta frequenza della STESSA categoria fra le ultime
        # scansioni (non come semplice sequenza di malattie qualsiasi).
        self._recent_categories: deque = deque(maxlen=self._epidemic_window)
        self._action_history: list[AgentDecision] = []
        self._total_cycles: int = 0
        self._correct_decisions: int = 0

        logger.info(
            "DecisionAgent inizializzato | soglie: "
            f"conf={self._conf_threshold:.0%}, "
            f"hum={self._humidity_high}%, "
            f"soil={self._soil_low}%, "
            f"epidemia={self._epidemic_min_count}/{self._epidemic_window}"
        )

    def _load_thresholds(self) -> None:
        cfg = self.config.get("agent", {})
        self._conf_threshold = cfg.get("confidence_threshold", 0.70)
        self._humidity_high = cfg.get("humidity_high_threshold", 80.0)
        self._soil_low = cfg.get("soil_moisture_low_threshold", 30.0)
        self._epidemic_window = cfg.get("epidemic_window", 10)
        self._epidemic_min_count = cfg.get("epidemic_min_count", 3)
        self._temp_high = cfg.get("temperature_high", 35.0)
        self._temp_low = cfg.get("temperature_low", 10.0)
        self._light_low = cfg.get("light_low_threshold", 2000.0)

    # ─────────────────────────────────────────────────────────────
    # Metodo principale: decide le azioni per un ciclo
    # ─────────────────────────────────────────────────────────────

    def decide(
        self,
        inference: InferenceResult,
        env: EnvironmentReading,
        true_label: Optional[str] = None,
    ) -> AgentDecision:
        """Esegue il ragionamento condizione → azione per un ciclo."""
        self._total_cycles += 1
        self._last_predictions.append(inference.predicted_label)

        actions: list[ActionResult] = []
        reasoning: list[str] = []
        label = inference.predicted_label
        conf = inference.confidence

        disease_category = self._classify_disease(label)
        self._recent_categories.append(disease_category)

        # ══════════════════════════════════════════════════════
        # REGOLA 1: Confidenza bassa → human-in-the-loop
        # ══════════════════════════════════════════════════════
        if inference.is_low_confidence(self._conf_threshold):
            reasoning.append(
                f"Confidenza {conf:.1%} sotto la soglia minima "
                f"({self._conf_threshold:.0%}): predizione inaffidabile, "
                f"richiesta ispezione umana"
            )
            actions.append(self.actuators.alarm.activate(
                f"confidenza modello bassa ({conf:.1%}) su '{label}'"
            ))
            actions.append(self.actuators.notification.send(
                f"Ispezione umana richiesta: CNN poco sicura ({conf:.1%}) "
                f"su immagine classificata come '{label}'",
                severity="WARNING",
            ))
            actions += self._handle_environmental(env, reasoning)
            self._update_consecutive(label, actions, reasoning)
            return self._make_decision(actions, reasoning, disease_category)

        # ══════════════════════════════════════════════════════
        # REGOLA 2: Malattia virale → allarme CRITICAL
        # ══════════════════════════════════════════════════════
        if label in VIRAL_DISEASES:
            self._consecutive_disease_count += 1
            reasoning.append(
                f"Rilevata malattia virale ({_friendly_label(label)}, "
                f"confidenza {conf:.1%}): possibile diffusione tramite insetti"
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
                f"Rilevata malattia fungina ({_friendly_label(label)}, "
                f"confidenza {conf:.1%})"
            )
            if env.humidity_pct > self._humidity_high:
                reasoning.append(
                    f"Umidità {env.humidity_pct:.1f}% sopra la soglia "
                    f"({self._humidity_high}%): condizioni favorevoli alla "
                    f"diffusione, ventilazione attivata"
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
                    f"Umidità {env.humidity_pct:.1f}% nella norma: "
                    "solo notifica, nessun attuatore azionato"
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
            cat = "batterica" if label in BACTERIAL_DISEASES else "da parassiti"
            reasoning.append(
                f"Rilevata malattia {cat} ({_friendly_label(label)}, "
                f"confidenza {conf:.1%})"
            )
            actions.append(self.actuators.notification.send(
                f"Malattia {cat} rilevata: '{label}' (conf {conf:.1%}). "
                "Valutare trattamento specifico.",
                severity="WARNING",
            ))

        # ══════════════════════════════════════════════════════
        # REGOLA 8: Healthy → disattiva attuatori patologia
        # ══════════════════════════════════════════════════════
        elif label == HEALTHY_CLASS:
            if self._consecutive_disease_count > 0:
                reasoning.append(
                    f"Pianta tornata sana dopo {self._consecutive_disease_count} "
                    "rilevamenti di malattia: attuatori di emergenza disattivati"
                )
                actions.append(self.actuators.alarm.deactivate("pianta tornata sana"))
                actions.append(self.actuators.ventilation.deactivate("pianta sana"))
                actions.append(self.actuators.notification.send(
                    "Pianta tornata in stato SANO. Attuatori di emergenza disattivati.",
                    severity="INFO",
                ))
            else:
                reasoning.append(
                    f"Pianta sana (confidenza {conf:.1%}): nessuna azione richiesta"
                )
            self._consecutive_disease_count = 0

        # ══════════════════════════════════════════════════════
        # Gestione sensori ambientali (parallela alla diagnosi)
        # ══════════════════════════════════════════════════════
        actions += self._handle_environmental(env, reasoning)

        # ══════════════════════════════════════════════════════
        # REGOLA 7: epidemia = alta frequenza della STESSA categoria
        # di malattia nella finestra delle ultime scansioni
        # ══════════════════════════════════════════════════════
        epidemic_cat, epidemic_count = self._detect_epidemic()
        if epidemic_cat is not None:
            reasoning.append(
                f"Allarme epidemia: {epidemic_count} scansioni su "
                f"{len(self._recent_categories)} recenti sono di categoria "
                f"'{epidemic_cat}' (soglia: {self._epidemic_min_count})"
            )
            actions.append(self.actuators.alarm.activate(
                f"epidemia '{epidemic_cat}': {epidemic_count} rilevamenti "
                f"nelle ultime {len(self._recent_categories)} scansioni"
            ))
            actions.append(self.actuators.notification.send(
                f"ATTENZIONE: possibile epidemia di tipo '{epidemic_cat}' "
                f"({epidemic_count} rilevamenti su {len(self._recent_categories)} "
                "scansioni recenti). Intervento urgente richiesto.",
                severity="CRITICAL",
            ))

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
        """Gestisce le condizioni ambientali indipendentemente dalla diagnosi CNN."""
        actions = []

        if env.soil_moisture_pct < self._soil_low:
            reasoning.append(
                f"Umidità del suolo {env.soil_moisture_pct:.1f}% sotto la "
                f"soglia ({self._soil_low}%): irrigazione attivata"
            )
            actions.append(self.actuators.irrigation.activate(
                f"umidità suolo bassa ({env.soil_moisture_pct:.0f}%)"
            ))
        else:
            if self.actuators.irrigation.is_active:
                reasoning.append(
                    f"Umidità del suolo {env.soil_moisture_pct:.1f}% tornata "
                    "nella norma: irrigazione disattivata"
                )
                actions.append(self.actuators.irrigation.deactivate(
                    f"umidità suolo {env.soil_moisture_pct:.0f}% nella norma"
                ))

        if env.temperature_c > self._temp_high:
            reasoning.append(
                f"Temperatura {env.temperature_c:.1f}°C sopra la soglia "
                f"({self._temp_high}°C): ventilazione attivata"
            )
            actions.append(self.actuators.ventilation.activate(
                f"temperatura alta ({env.temperature_c:.1f}°C)"
            ))

        if env.temperature_c < self._temp_low:
            reasoning.append(
                f"Temperatura {env.temperature_c:.1f}°C sotto la soglia "
                f"({self._temp_low}°C): possibile guasto al riscaldamento"
            )
            actions.append(self.actuators.notification.send(
                f"Temperatura troppo bassa: {env.temperature_c:.1f}°C. "
                "Verificare sistema di riscaldamento.",
                severity="WARNING",
            ))

        if env.light_lux < self._light_low:
            reasoning.append(
                f"Luminosità {env.light_lux:.0f} lux sotto la soglia "
                f"({self._light_low:.0f} lux): illuminazione insufficiente"
            )
            actions.append(self.actuators.notification.send(
                f"Luminosità scarsa: {env.light_lux:.0f} lux. "
                "Considerare illuminazione supplementare.",
                severity="INFO",
            ))

        return actions

    def _detect_epidemic(self) -> tuple[Optional[str], int]:
        """Rileva un'epidemia come alta frequenza della STESSA categoria di
        malattia nella finestra scorrevole delle ultime scansioni."""
        disease_cats = [
            c for c in self._recent_categories
            if c not in ("healthy", "unknown")
        ]
        if not disease_cats:
            return None, 0
        top_cat, top_count = Counter(disease_cats).most_common(1)[0]
        if top_count >= self._epidemic_min_count:
            return top_cat, top_count
        return None, 0

    def _update_consecutive(
        self,
        label: str,
        actions: list,
        reasoning: list,
    ) -> None:
        if label != HEALTHY_CLASS:
            self._consecutive_disease_count += 1
        else:
            self._consecutive_disease_count = 0

    @staticmethod
    def _classify_disease(label: str) -> str:
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
