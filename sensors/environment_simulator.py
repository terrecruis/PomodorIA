"""
sensors/environment_simulator.py — Environmental Sensor Simulator (IoTWF Livello 1)

Simula i sensori ambientali della serra:
- Temperatura (°C) → come un DHT22
- Umidità relativa (%) → come un DHT22
- Umidità del suolo (%) → come un igrometro capacitivo
- Luminosità (lux) → come un BH1750

I valori NON sono puramente casuali: sono **correlati euristicamente
alla patologia rilevata dalla fotocamera**, implementando il concetto
di Data Fusion (§3.1 del doc di progetto).

Correlazioni botaniche reali usate nella simulazione:
- Malattie fungine (Early/Late blight, Leaf Mold, Septoria)
  → umidità alta, temperatura moderata, scarsa ventilazione
- Acari (Spider mites) → ambiente secco, caldo
- Malattie virali (TYLCV, Mosaic) → temperature varie, vettori insetti
- Pianta sana → condizioni ottimali
"""

import random
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════
# Dataclass per la lettura dei sensori ambientali
# ══════════════════════════════════════════════════════════════
@dataclass
class EnvironmentReading:
    """Lettura dei sensori ambientali simulati."""
    timestamp: float # epoch time della lettura
    temperature_c: float # temperatura in °C
    humidity_pct: float # umidità relativa in %
    soil_moisture_pct: float # umidità del suolo in %
    light_lux: float # luminosità in lux

    def to_dict(self) -> dict:
        """Converte la lettura in dizionario (per logging/CSV)."""
        return {
            "timestamp": self.timestamp,
            "temperature_c": round(self.temperature_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "soil_moisture_pct": round(self.soil_moisture_pct, 1),
            "light_lux": round(self.light_lux, 0),
        }


# ══════════════════════════════════════════════════════════════
# Lettura combinata (camera + ambiente) per il ciclo orchestratore
# ══════════════════════════════════════════════════════════════
@dataclass
class SensorReading:
    """
    Lettura completa di un ciclo sense: combina lo scatto della
    fotocamera con i dati ambientali. Questa è la struttura che
    viene passata all'Edge AI Layer e poi all'Agente Decisionale.
    """
    timestamp: float
    image_path: str
    true_label: str # ground truth dal dataset (per validazione PoC)
    true_label_idx: int
    temperature_c: float
    humidity_pct: float
    soil_moisture_pct: float
    light_lux: float

    def to_dict(self) -> dict:
        """Converte in dizionario per logging strutturato."""
        return {
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "true_label": self.true_label,
            "true_label_idx": self.true_label_idx,
            "temperature_c": round(self.temperature_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "soil_moisture_pct": round(self.soil_moisture_pct, 1),
            "light_lux": round(self.light_lux, 0),
        }


# ══════════════════════════════════════════════════════════════
# Profili ambientali correlati alle patologie
# ══════════════════════════════════════════════════════════════

# Ogni profilo definisce (media, std) per ogni sensore,
# riflettendo le condizioni ambientali tipiche della patologia
_DISEASE_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    # ── Malattie fungine: umidità alta, temperature moderate ──
    "Tomato___Bacterial_spot": {
        "temperature_c": (27.0, 2.0), # warm, humid
        "humidity_pct": (82.0, 5.0),
        "soil_moisture_pct": (65.0, 8.0),
        "light_lux": (15000.0, 5000.0),
    },
    "Tomato___Early_blight": {
        "temperature_c": (25.0, 2.5), # moderate temp, alta umidità
        "humidity_pct": (85.0, 4.0),
        "soil_moisture_pct": (60.0, 10.0),
        "light_lux": (12000.0, 4000.0),
    },
    "Tomato___Late_blight": {
        "temperature_c": (20.0, 3.0), # più fresco del Early
        "humidity_pct": (90.0, 3.0), # umidità molto alta
        "soil_moisture_pct": (70.0, 8.0),
        "light_lux": (8000.0, 3000.0), # scarsa luce
    },
    "Tomato___Leaf_Mold": {
        "temperature_c": (24.0, 2.0), # warm, very humid
        "humidity_pct": (88.0, 4.0),
        "soil_moisture_pct": (68.0, 7.0),
        "light_lux": (10000.0, 4000.0),
    },
    "Tomato___Septoria_leaf_spot": {
        "temperature_c": (25.0, 2.0), # warm, wet
        "humidity_pct": (84.0, 5.0),
        "soil_moisture_pct": (72.0, 6.0),
        "light_lux": (14000.0, 5000.0),
    },

    # ── Acari: ambiente secco e caldo ──
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "temperature_c": (32.0, 2.0), # caldo
        "humidity_pct": (35.0, 5.0), # secco
        "soil_moisture_pct": (25.0, 8.0), # terreno secco
        "light_lux": (40000.0, 8000.0), # alta esposizione
    },
    "Tomato___Target_Spot": {
        "temperature_c": (26.0, 2.5), # warm, humid
        "humidity_pct": (80.0, 6.0),
        "soil_moisture_pct": (62.0, 8.0),
        "light_lux": (18000.0, 6000.0),
    },

    # ── Malattie virali: temperature varie, vettori insetti ──
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "temperature_c": (30.0, 3.0), # caldo (favorisce mosche bianche)
        "humidity_pct": (55.0, 10.0),
        "soil_moisture_pct": (45.0, 10.0),
        "light_lux": (35000.0, 8000.0),
    },
    "Tomato___Tomato_mosaic_virus": {
        "temperature_c": (26.0, 4.0), # varie (trasmissione meccanica)
        "humidity_pct": (60.0, 12.0),
        "soil_moisture_pct": (50.0, 12.0),
        "light_lux": (25000.0, 8000.0),
    },

    # ── Pianta sana: condizioni ottimali ──
    "Tomato___healthy": {
        "temperature_c": (24.0, 2.0), # temperatura ideale
        "humidity_pct": (65.0, 5.0), # umidità moderata
        "soil_moisture_pct": (55.0, 5.0), # suolo ben irrigato
        "light_lux": (30000.0, 5000.0), # buona illuminazione
    },
}


# ══════════════════════════════════════════════════════════════
# Environment Sensor Simulator
# ══════════════════════════════════════════════════════════════
class EnvironmentSensorSimulator:
    """
    Simula i sensori ambientali della serra (DHT22, igrometro, luxmetro).

    I valori generati sono correlati alla patologia osservata dalla
    fotocamera (Data Fusion), rendendo la simulazione plausibile
    e coerente per l'agente decisionale.

    Parametri (letti dal config):
    - sensors.temperature: range e rumore
    - sensors.humidity: range e rumore
    - sensors.soil_moisture: range e rumore
    - sensors.light: range e rumore
    """

    def __init__(self, config: dict):
        """
        Inizializza il simulatore con i parametri dal config.

        Args:
            config: dizionario di configurazione (config.yaml caricato)
        """
        self.sensor_config = config["sensors"]

        # Range di clamp per evitare valori fisicamente impossibili
        self._clamp_ranges = {
            "temperature_c": (0.0, 50.0),
            "humidity_pct": (10.0, 100.0),
            "soil_moisture_pct": (0.0, 100.0),
            "light_lux": (0.0, 100000.0),
        }

        self.total_readings = 0

        print("EnvironmentSensorSimulator inizializzato")

    def read(self, disease_label: Optional[str] = None) -> EnvironmentReading:
        """
        Genera una lettura dei sensori ambientali.

        Se disease_label è fornito, i valori sono correlati
        alla patologia (Data Fusion). Altrimenti usa range
        generici dal config.

        Args:
            disease_label: nome della patologia rilevata dalla camera
                          (es. "Tomato___Early_blight"), o None per valori generici

        Returns:
            EnvironmentReading con i valori dei 4 sensori
        """
        if disease_label and disease_label in _DISEASE_PROFILES:
            reading = self._generate_correlated(disease_label)
        else:
            reading = self._generate_generic()

        self.total_readings += 1
        return reading

    def _generate_correlated(self, disease_label: str) -> EnvironmentReading:
        """
        Genera valori correlati al profilo della patologia.
        Aggiunge un leggero rumore gaussiano per variabilità realistica.
        """
        profile = _DISEASE_PROFILES[disease_label]

        values = {}
        for sensor_name, (mean, std) in profile.items():
            value = np.random.normal(mean, std)
            # Clamp nei range fisicamente possibili
            lo, hi = self._clamp_ranges[sensor_name]
            values[sensor_name] = float(np.clip(value, lo, hi))

        return EnvironmentReading(
            timestamp=time.time(),
            temperature_c=values["temperature_c"],
            humidity_pct=values["humidity_pct"],
            soil_moisture_pct=values["soil_moisture_pct"],
            light_lux=values["light_lux"],
        )

    def _generate_generic(self) -> EnvironmentReading:
        """
        Genera valori dai range generici definiti nel config.yaml,
        senza correlazione con alcuna patologia.
        """
        cfg = self.sensor_config

        temp = np.random.uniform(
            cfg["temperature"]["base_min"],
            cfg["temperature"]["base_max"]
        ) + np.random.normal(0, cfg["temperature"]["noise_std"])

        hum = np.random.uniform(
            cfg["humidity"]["base_min"],
            cfg["humidity"]["base_max"]
        ) + np.random.normal(0, cfg["humidity"]["noise_std"])

        soil = np.random.uniform(
            cfg["soil_moisture"]["base_min"],
            cfg["soil_moisture"]["base_max"]
        ) + np.random.normal(0, cfg["soil_moisture"]["noise_std"])

        light = np.random.uniform(
            cfg["light"]["base_min"],
            cfg["light"]["base_max"]
        ) + np.random.normal(0, cfg["light"]["noise_std"])

        # Clamp nei range fisici
        temp = float(np.clip(temp, *self._clamp_ranges["temperature_c"]))
        hum = float(np.clip(hum, *self._clamp_ranges["humidity_pct"]))
        soil = float(np.clip(soil, *self._clamp_ranges["soil_moisture_pct"]))
        light = float(np.clip(light, *self._clamp_ranges["light_lux"]))

        return EnvironmentReading(
            timestamp=time.time(),
            temperature_c=temp,
            humidity_pct=hum,
            soil_moisture_pct=soil,
            light_lux=light,
        )

    def get_disease_profile(self, disease_label: str) -> Optional[dict]:
        """Restituisce il profilo ambientale di una patologia (per debug)."""
        return _DISEASE_PROFILES.get(disease_label, None)

    def __repr__(self) -> str:
        return f"EnvironmentSensorSimulator(readings={self.total_readings})"
