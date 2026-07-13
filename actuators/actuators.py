"""
actuators/actuators.py — Actuator Layer (IoTWF Livello 1, lato output)

Ogni attuatore è una classe con metodi activate()/deactivate() che,
invece di azionare pin GPIO reali, logga l'azione tramite FakeGPIO.

Il codice è strutturato per essere "drop-in replaceable" su hardware
reale: basta sostituire `from actuators.fake_gpio import ...` con
`import RPi.GPIO as GPIO` senza modificare la logica degli attuatori.

Attuatori simulati:
    IrrigationActuator — pompa dell'acqua (GPIO pin 17)
    VentilationActuator — ventola di areazione (GPIO pin 27)
    AlarmActuator — LED/buzzer di allarme (GPIO pin 22)
    NotificationActuator — log testuale notifica all'agricoltore

ActionResult: dataclass restituita da ogni activate(), contiene
    il tipo di azione, il motivo e il timestamp.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from actuators import fake_gpio as GPIO

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Dataclass risultato di un'azione attuatore
# ══════════════════════════════════════════════════════════════
@dataclass
class ActionResult:
    """Descrive l'azione eseguita da un attuatore in un ciclo."""
    actuator_name: str
    action: str
    reason: str
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"[{self.actuator_name}] {self.action.upper()} "
            f"→ {self.reason}"
        )


# ══════════════════════════════════════════════════════════════
# Classe base attuatore
# ══════════════════════════════════════════════════════════════
class SimulatedActuator:
    """
    Classe base per tutti gli attuatori simulati.

    Gestisce lo stato (attivo/non attivo) e il logging.
    Sottoclassi devono definire `name` e `pin`.
    """

    name: str = "GenericActuator"
    pin: int = 0

    def __init__(self):
        self._active: bool = False
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str = "") -> ActionResult:
        """
        Attiva l'attuatore (porta il pin a HIGH).

        Args:
            reason: descrizione human-readable del motivo dell'attivazione

        Returns:
            ActionResult con i dettagli dell'azione
        """
        if not self._active:
            GPIO.output(self.pin, GPIO.HIGH)
            self._active = True
            msg = f"{self.name} ATTIVATO → {reason}"
            logger.info(msg)
            print(msg)
        else:
            logger.debug(f"{self.name} già attivo, motivo aggiornato: {reason}")

        return ActionResult(
            actuator_name=self.name,
            action="activate",
            reason=reason,
        )

    def deactivate(self, reason: str = "condizioni rientrate nella norma") -> ActionResult:
        """
        Disattiva l'attuatore (porta il pin a LOW).

        Args:
            reason: descrizione del motivo della disattivazione

        Returns:
            ActionResult con i dettagli dell'azione
        """
        if self._active:
            GPIO.output(self.pin, GPIO.LOW)
            self._active = False
            msg = f"{self.name} DISATTIVATO → {reason}"
            logger.info(msg)
            print(msg)

        return ActionResult(
            actuator_name=self.name,
            action="deactivate",
            reason=reason,
        )

    def no_action(self) -> ActionResult:
        """Restituisce un ActionResult di tipo 'none' senza modificare lo stato."""
        return ActionResult(
            actuator_name=self.name,
            action="none",
            reason="nessuna azione richiesta",
        )

    def __repr__(self) -> str:
        state = "ATTIVO" if self._active else "inattivo"
        return f"{self.name}(pin={self.pin}, stato={state})"


# ══════════════════════════════════════════════════════════════
# Attuatori specifici
# ══════════════════════════════════════════════════════════════

class IrrigationActuator(SimulatedActuator):
    """
    Pompa di irrigazione.

    Si attiva quando la soil moisture scende sotto la soglia
    o quando la CNN rileva una malattia che richiede idratazione controllata.

    Pin GPIO: 17 (BCM)
    """
    name = "IrrigationActuator"
    pin = 17


class VentilationActuator(SimulatedActuator):
    """
    Ventola di areazione.

    Si attiva quando:
    - temperatura troppo alta
    - umidità alta + malattia fungina rilevata (Early/Late blight)

    Pin GPIO: 27 (BCM)
    """
    name = "VentilationActuator"
    pin = 27


class AlarmActuator(SimulatedActuator):
    """
    LED/buzzer di allarme.

    Si attiva per segnalare:
    - N rilevamenti consecutivi di malattia (epidemia simulata)
    - Confidenza modello sotto soglia → richiesta ispezione umana
    - Rilevamento malattia virale (vettore insetto → alta urgenza)

    Pin GPIO: 22 (BCM)
    """
    name = "AlarmActuator"
    pin = 22


class NotificationActuator:
    """
    Notifica testuale all'agricoltore (IoTWF Livello 7 — Collaboration).

    Non ha pin GPIO (è un'azione software). Simula l'invio di una
    notifica push/SMS/email loggando il messaggio con alta visibilità.
    """
    name = "NotificationActuator"

    def __init__(self):
        self._notifications_sent: int = 0

    def send(self, message: str, severity: str = "INFO") -> ActionResult:
        """
        Invia una notifica (simulata) all'agricoltore.

        Args:
            message: testo della notifica
            severity: "INFO" | "WARNING" | "CRITICAL"

        Returns:
            ActionResult con il messaggio inviato
        """
        self._notifications_sent += 1

        msg = f"NOTIFICA [{severity}] → {message}"
        logger.warning(msg) if severity != "INFO" else logger.info(msg)
        print(msg)

        return ActionResult(
            actuator_name=self.name,
            action="notify",
            reason=f"[{severity}] {message}",
        )

    @property
    def total_notifications(self) -> int:
        return self._notifications_sent

    def __repr__(self) -> str:
        return f"NotificationActuator(sent={self._notifications_sent})"


# ══════════════════════════════════════════════════════════════
# Registro centralizzato degli attuatori
# ══════════════════════════════════════════════════════════════
class ActuatorBank:
    """
    Gestisce tutti gli attuatori del sistema in un unico oggetto.

    Fornisce metodi di utilità per resettare tutti gli attuatori
    e recuperare lo stato complessivo del sistema.
    """

    def __init__(self):
        self.irrigation = IrrigationActuator()
        self.ventilation = VentilationActuator()
        self.alarm = AlarmActuator()
        self.notification = NotificationActuator()

        logger.info("ActuatorBank inizializzato con 4 attuatori simulati")

    def deactivate_all(self, reason: str = "reset manuale") -> list[ActionResult]:
        """Disattiva tutti gli attuatori GPIO."""
        results = []
        for act in [self.irrigation, self.ventilation, self.alarm]:
            if act.is_active:
                results.append(act.deactivate(reason))
        return results

    def get_status(self) -> dict:
        """Restituisce lo stato corrente di tutti gli attuatori."""
        return {
            "irrigation_active": self.irrigation.is_active,
            "ventilation_active": self.ventilation.is_active,
            "alarm_active": self.alarm.is_active,
            "notifications_sent": self.notification.total_notifications,
            "gpio_states": GPIO.get_all_states(),
        }

    def __repr__(self) -> str:
        s = self.get_status()
        return (
            f"ActuatorBank("
            f"irr={'ON' if s['irrigation_active'] else 'off'}, "
            f"vent={'ON' if s['ventilation_active'] else 'off'}, "
            f"alarm={'ON' if s['alarm_active'] else 'off'}, "
            f"notif={s['notifications_sent']})"
        )
