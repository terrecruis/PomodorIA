"""
sensors/ — Sensor Layer (IoTWF Livello 1: Physical Devices & Controllers)

Contiene i sensori simulati della serra:
- VirtualCameraSensor: fotocamera che pesca dal dataset PlantVillage
- EnvironmentSensorSimulator: sensori ambientali (temp, umidità, suolo, luce)
"""

from sensors.virtual_camera import VirtualCameraSensor, CameraCapture
from sensors.environment_simulator import (
    EnvironmentSensorSimulator,
    EnvironmentReading,
    SensorReading,
)

__all__ = [
    "VirtualCameraSensor",
    "CameraCapture",
    "EnvironmentSensorSimulator",
    "EnvironmentReading",
    "SensorReading",
]
