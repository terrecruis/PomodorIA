"""
sensors/virtual_camera.py — Virtual Camera Sensor (IoTWF Livello 1)

Simula una fotocamera montata nella serra, pescando immagini reali
dal dataset PlantVillage-Tomato. Nel PoC sostituisce una fotocamera
fisica collegata al Raspberry Pi via CSI/USB.

Ogni "scatto" restituisce:
- L'immagine preprocessata come tensor PyTorch (pronta per la CNN)
- Il path dell'immagine originale
- La classe reale (ground truth), per validazione nel PoC
- Il timestamp dello scatto

Parametro opzionale `bias_class`: forza la simulazione a pescare
più spesso da una classe specifica (utile per demo mirate,
es. simulare un'epidemia di Early blight).
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from PIL import Image
from torchvision import transforms


# ══════════════════════════════════════════════════════════════
# Dataclass per il risultato di uno "scatto" della fotocamera
# ══════════════════════════════════════════════════════════════
@dataclass
class CameraCapture:
    """Risultato di una cattura della fotocamera virtuale."""
    timestamp: float # epoch time dello scatto
    image_tensor: torch.Tensor # tensor preprocessato (1, 3, 64, 64)
    image_path: str # path completo dell'immagine originale
    true_label: str # classe reale (ground truth dal dataset)
    true_label_idx: int # indice numerico della classe


# ══════════════════════════════════════════════════════════════
# Virtual Camera Sensor
# ══════════════════════════════════════════════════════════════
class VirtualCameraSensor:
    """
    Simula una fotocamera nella serra, pescando immagini dal dataset
    PlantVillage-Tomato.

    Parametri (letti dal config):
    - dataset_root: percorso alla cartella con le 10 classi
    - classes: lista ordinata dei nomi delle classi
    - image_size: dimensione di resize (64x64)
    - mean/std: normalizzazione ImageNet
    - bias_class: se impostato, pesca più spesso da quella classe
    """

    def __init__(self, config: dict):
        """
        Inizializza il sensore caricando l'indice del dataset.

        Args:
            config: dizionario di configurazione (config.yaml caricato)
        """
        self.dataset_root = config["paths"]["dataset_root"]
        self.classes = config["model"]["classes"]
        self.bias_class = config["virtual_camera"].get("bias_class", None)

        # Preprocessing identico al training (§1.1 del doc di progetto)
        preproc = config["preprocessing"]
        self.transform = transforms.Compose([
            transforms.Resize((preproc["image_size"], preproc["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preproc["mean"],
                std=preproc["std"]
            ),
        ])

        # Costruisce l'indice: {classe: [lista_path_immagini]}
        self._image_index: Dict[str, List[str]] = {}
        self._all_images: List[tuple] = [] # (path, classe, idx_classe)
        self._build_index()

        self.total_captures = 0

    def _build_index(self) -> None:
        """
        Scansiona il dataset e costruisce un indice in-memory
        di tutte le immagini disponibili per classe.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_root, class_name)

            if not os.path.isdir(class_dir):
                print(f"Cartella non trovata per classe '{class_name}': {class_dir}")
                self._image_index[class_name] = []
                continue

            images = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            images.sort() # ordinamento deterministico
            self._image_index[class_name] = images

            for img_path in images:
                self._all_images.append((img_path, class_name, class_idx))

        total = sum(len(v) for v in self._image_index.values())
        print(f"VirtualCamera inizializzata: {total:,} immagini, "
              f"{len(self.classes)} classi")

        if self.bias_class:
            n_bias = len(self._image_index.get(self.bias_class, []))
            print(f"   Bias attivo verso '{self.bias_class}' ({n_bias} immagini)")

    def capture(self) -> CameraCapture:
        """
        Esegue uno "scatto" della fotocamera virtuale.

        Se bias_class è impostato, con probabilità 50% pesca da
        quella classe, altrimenti pesca uniformemente da tutte.

        Returns:
            CameraCapture con immagine preprocessata e metadati
        """
        if self.bias_class and random.random() < 0.5:
            images = self._image_index.get(self.bias_class, [])
            if images:
                img_path = random.choice(images)
                true_label = self.bias_class
                true_label_idx = self.classes.index(self.bias_class)
            else:
                # Fallback a selezione casuale se la classe non esiste
                img_path, true_label, true_label_idx = random.choice(self._all_images)
        else:
            img_path, true_label, true_label_idx = random.choice(self._all_images)

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0) # (1, 3, 64, 64)

        self.total_captures += 1

        return CameraCapture(
            timestamp=time.time(),
            image_tensor=image_tensor,
            image_path=img_path,
            true_label=true_label,
            true_label_idx=true_label_idx,
        )

    def capture_from_class(self, class_name: str) -> CameraCapture:
        """
        Esegue uno scatto forzato da una classe specifica.
        Utile per test e demo mirate.

        Args:
            class_name: nome esatto della classe (es. "Tomato___Early_blight")

        Returns:
            CameraCapture dalla classe specificata

        Raises:
            ValueError: se la classe non esiste o non ha immagini
        """
        images = self._image_index.get(class_name, [])
        if not images:
            available = [c for c, imgs in self._image_index.items() if imgs]
            raise ValueError(
                f"Classe '{class_name}' non trovata o vuota. "
                f"Classi disponibili: {available}"
            )

        img_path = random.choice(images)
        true_label_idx = self.classes.index(class_name)

        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)

        self.total_captures += 1

        return CameraCapture(
            timestamp=time.time(),
            image_tensor=image_tensor,
            image_path=img_path,
            true_label=class_name,
            true_label_idx=true_label_idx,
        )

    def get_class_distribution(self) -> Dict[str, int]:
        """Restituisce la distribuzione delle immagini per classe."""
        return {cls: len(imgs) for cls, imgs in self._image_index.items()}

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._image_index.values())
        return (f"VirtualCameraSensor(images={total:,}, "
                f"classes={len(self.classes)}, "
                f"bias={self.bias_class})")
