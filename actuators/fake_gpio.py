"""
actuators/fake_gpio.py — Mock GPIO (IoTWF Livello 1)

Sostituisce la libreria RPi.GPIO con un'implementazione fittizia
che ha la stessa interfaccia pubblica, ma invece di azionare pin
reali stampa/logga l'azione.

Obiettivo: il codice degli attuatori è già scritto nella forma
"drop-in replaceable" su hardware reale — basta sostituire l'import
con RPi.GPIO senza modificare altro.

Costanti GPIO reali:
    GPIO.BCM  = 11  (numerazione Broadcom)
    GPIO.BOARD = 10 (numerazione fisica)
    GPIO.OUT  = 0
    GPIO.IN   = 1
    GPIO.HIGH = True
    GPIO.LOW  = False
"""

import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Costanti (identiche a RPi.GPIO)
# ──────────────────────────────────────────────────────────────
BCM   = 11
BOARD = 10
OUT   = 0
IN    = 1
HIGH  = True
LOW   = False

# Stato interno simulato dei pin
_pin_state: dict[int, bool] = {}
_mode: int | None = None


# ──────────────────────────────────────────────────────────────
# API pubblica (identica a RPi.GPIO)
# ──────────────────────────────────────────────────────────────

def setmode(mode: int) -> None:
    """Imposta la modalità di numerazione pin (BCM o BOARD)."""
    global _mode
    _mode = mode
    mode_name = "BCM" if mode == BCM else "BOARD"
    logger.debug(f"[FakeGPIO] setmode({mode_name})")


def setup(pin: int, direction: int, initial: bool = LOW) -> None:
    """
    Configura un pin come input o output con stato iniziale.

    Args:
        pin:       numero del pin GPIO
        direction: GPIO.OUT o GPIO.IN
        initial:   stato iniziale (GPIO.HIGH o GPIO.LOW)
    """
    _pin_state[pin] = initial
    dir_name = "OUT" if direction == OUT else "IN"
    logger.debug(f"[FakeGPIO] setup(pin={pin}, direction={dir_name}, initial={initial})")


def output(pin: int, state: bool) -> None:
    """
    Scrive il valore su un pin di output.

    Args:
        pin:   numero del pin GPIO
        state: GPIO.HIGH (True) o GPIO.LOW (False)
    """
    _pin_state[pin] = state
    state_name = "HIGH" if state else "LOW"
    logger.debug(f"[FakeGPIO] output(pin={pin}, state={state_name})")


def input(pin: int) -> bool:
    """
    Legge il valore di un pin.

    Args:
        pin: numero del pin GPIO

    Returns:
        True (HIGH) o False (LOW)
    """
    return _pin_state.get(pin, LOW)


def cleanup(pins: list[int] | None = None) -> None:
    """
    Rilascia le risorse GPIO. Resetta i pin specificati (o tutti).

    Args:
        pins: lista di pin da resettare, None = tutti
    """
    global _pin_state
    if pins is None:
        _pin_state.clear()
        logger.debug("[FakeGPIO] cleanup() — tutti i pin resettati")
    else:
        for pin in pins:
            _pin_state.pop(pin, None)
        logger.debug(f"[FakeGPIO] cleanup(pins={pins})")


def get_all_states() -> dict:
    """Restituisce lo stato corrente di tutti i pin (utile per debug/test)."""
    return dict(_pin_state)
