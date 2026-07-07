"""
edge/inference_engine.py — Edge AI Layer (IoTWF Livello 3)

Wrapper di inferenza che carica la TomatoCNN e la esegue su un tensor
già preprocessato dalla VirtualCamera. Simula il comportamento di un
motore di inferenza su Raspberry Pi: singolo core CPU, no GPU, misura
RAM e latenza ad ogni predizione.

Supporta due modalità (config.yaml → model.mode):
  - "full_precision"  → modello originale float32 (.pth baseline)
  - "optimized"       → modello compresso prodotto da models/compress.py
                        (PyTorch quantizzato oppure ONNX Runtime)

Flusso:
    CameraCapture.image_tensor
           │
           ▼
    [InferenceEngine.predict()]
           │  torch.no_grad()
           │  softmax(logits) → argmax → label
           │  misura tempo + RAM
           ▼
    InferenceResult
"""

import sys
import time
import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn.functional as F
import psutil

# ── Backend di quantizzazione ────────────────────────────────
# Deve essere impostato PRIMA di caricare un modello .pth quantizzato
# (torch.load fallisce con "Unknown qengine" altrimenti), esattamente
# come fatto in models/compress.py al momento della creazione dei pesi.
_supported_qengines = torch.backends.quantized.supported_engines
if "qnnpack" in _supported_qengines:
    torch.backends.quantized.engine = "qnnpack"
elif "fbgemm" in _supported_qengines:
    torch.backends.quantized.engine = "fbgemm"

# Aggiunge la root del progetto al path per import relativi
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.model import TomatoCNN


# ══════════════════════════════════════════════════════════════
# Dataclass risultato inferenza
# ══════════════════════════════════════════════════════════════
@dataclass
class InferenceResult:
    """
    Risultato di una singola inferenza della CNN.

    Campi:
        predicted_label     : nome della classe predetta (es. "Tomato___Early_blight")
        predicted_idx       : indice numerico della classe predetta (0–9)
        confidence          : probabilità softmax della classe predetta (0.0–1.0)
        all_probabilities   : vettore softmax completo (10 valori), utile per log/debug
        inference_time_ms   : tempo totale di inferenza in millisecondi
        ram_used_mb         : RAM del processo in quel momento (MB) — proxy Raspberry Pi
        model_mode          : modalità del modello ("full_precision" o "optimized")
    """
    predicted_label: str
    predicted_idx: int
    confidence: float
    all_probabilities: List[float]
    inference_time_ms: float
    ram_used_mb: float
    model_mode: str

    def is_healthy(self) -> bool:
        """True se la pianta è classificata come sana."""
        return self.predicted_label == "Tomato___healthy"

    def is_low_confidence(self, threshold: float = 0.70) -> bool:
        """True se la confidenza è sotto la soglia → human-in-the-loop."""
        return self.confidence < threshold

    def __str__(self) -> str:
        status = "🌿 SANA" if self.is_healthy() else f"⚠️  MALATTIA"
        return (
            f"{status} | {self.predicted_label} "
            f"(conf: {self.confidence:.1%}, "
            f"t: {self.inference_time_ms:.1f}ms, "
            f"RAM: {self.ram_used_mb:.1f}MB) "
            f"[{self.model_mode}]"
        )


# ══════════════════════════════════════════════════════════════
# Inference Engine
# ══════════════════════════════════════════════════════════════
class InferenceEngine:
    """
    Motore di inferenza Edge AI per TomatoCNN.

    Carica il modello una sola volta all'avvio e lo mantiene in memoria.
    Ogni chiamata a predict() esegue l'inferenza su un singolo tensor
    (1, 3, 64, 64) già preprocessato dalla VirtualCamera.

    Modalità supportate:
        - "full_precision" : PyTorch float32, modello .pth originale
        - "optimized"      : PyTorch INT8 (quantizzato) o ONNX Runtime

    Simulazione Raspberry Pi:
        - torch.set_num_threads(1)  → 1 solo core CPU
        - device = "cpu"            → no GPU/MPS
        - psutil.Process().memory_info().rss → misura RAM
    """

    def __init__(self, config: dict):
        """
        Inizializza e carica il modello.

        Args:
            config: dizionario letto da config.yaml
        """
        self.config = config
        self.classes: List[str] = config["model"]["classes"]
        self.mode: str = config["model"].get("mode", "full_precision")
        self.optimized_variant: str = config["model"].get("optimized_variant", "auto")
        self.device = torch.device(config["edge_simulation"].get("device", "cpu"))

        # ── Simulazione Raspberry Pi: limita a 1 thread ──────────
        num_threads = config["edge_simulation"].get("num_threads", 1)
        torch.set_num_threads(num_threads)
        print(f"🔧 Edge simulation: {num_threads} thread(s) CPU, device={self.device}")

        # ── Caricamento modello ──────────────────────────────────
        self._model_pytorch: Optional[torch.nn.Module] = None
        self._onnx_session = None  # onnxruntime.InferenceSession, se usato
        self.loaded_variant: str = "full_precision"  # etichetta descrittiva del file caricato

        self._load_model()

        # ── psutil: handle al processo corrente ──────────────────
        self._process = psutil.Process()

        print(f"✅ InferenceEngine pronto | modalità: {self.mode}")

    # ─────────────────────────────────────────────────────────────
    # Caricamento modello
    # ─────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Carica il modello in base alla modalità configurata.

        full_precision: carica il checkpoint .pth originale (float32)
        optimized:      carica la variante scelta da `model.optimized_variant`
                        (config.yaml), oppure la cerca in cascata se
                        variant == "auto":
                            ONNX quantizzato > PyTorch pruned+quantizzato >
                            PyTorch pruned > fallback full_precision
        """
        if self.mode == "full_precision":
            self._load_pytorch_model(
                checkpoint_path=self.config["paths"]["checkpoint"],
                quantized=False
            )
            self.loaded_variant = "full_precision"

        elif self.mode == "optimized":
            optimized_dir = self.config["paths"]["optimized_dir"]

            variants = {
                "onnx":              ("onnx", os.path.join(optimized_dir, "model_quantized.onnx")),
                "pruned_quantized":  ("pth_quantized", os.path.join(optimized_dir, "model_pruned_quantized.pth")),
                "pruned":            ("pth", os.path.join(optimized_dir, "model_pruned.pth")),
            }

            if self.optimized_variant != "auto":
                # ── Variante scelta esplicitamente (da config.yaml o dashboard) ──
                if self.optimized_variant not in variants:
                    raise ValueError(
                        f"optimized_variant non valido: '{self.optimized_variant}'. "
                        f"Usa: 'auto', {', '.join(repr(k) for k in variants)}"
                    )
                kind, path = variants[self.optimized_variant]
                if not os.path.exists(path):
                    print(
                        f"⚠️  Variante '{self.optimized_variant}' richiesta ma file "
                        f"'{path}' non trovato. Fallback a full_precision. "
                        "(hai eseguito 'python models/compress.py'?)"
                    )
                    self.mode = "full_precision"
                    self._load_pytorch_model(
                        checkpoint_path=self.config["paths"]["checkpoint"],
                        quantized=False
                    )
                    self.loaded_variant = "full_precision"
                    return
                if kind == "onnx":
                    self._load_onnx_model(path)
                    self.loaded_variant = "optimized_onnx"
                elif kind == "pth_quantized":
                    self._load_pytorch_model(path, quantized=True)
                    self.loaded_variant = "optimized_pruned_quantized"
                else:
                    self._load_pytorch_model(path, quantized=False)
                    self.loaded_variant = "optimized_pruned"
                return

            # ── variant == "auto": cascata di fallback originale ──
            onnx_path = variants["onnx"][1]
            pt_quantized_path = variants["pruned_quantized"][1]
            pt_pruned_path = variants["pruned"][1]

            if os.path.exists(onnx_path):
                self._load_onnx_model(onnx_path)
                self.loaded_variant = "optimized_onnx"
            elif os.path.exists(pt_quantized_path):
                self._load_pytorch_model(pt_quantized_path, quantized=True)
                self.loaded_variant = "optimized_pruned_quantized"
            elif os.path.exists(pt_pruned_path):
                self._load_pytorch_model(pt_pruned_path, quantized=False)
                self.loaded_variant = "optimized_pruned"
            else:
                print(
                    "⚠️  Nessun modello ottimizzato trovato in "
                    f"'{optimized_dir}'. Fallback a full_precision."
                )
                self.mode = "full_precision"
                self._load_pytorch_model(
                    checkpoint_path=self.config["paths"]["checkpoint"],
                    quantized=False
                )
                self.loaded_variant = "full_precision"
        else:
            raise ValueError(
                f"Modalità non supportata: '{self.mode}'. "
                "Usa 'full_precision' o 'optimized'."
            )

    def _load_pytorch_model(self, checkpoint_path: str, quantized: bool) -> None:
        """
        Carica un modello PyTorch (.pth) con state_dict.

        Il modello originale è stato salvato come:
            torch.save(model.state_dict(), '...pth')
        quindi va prima istanziata l'architettura e poi caricati i pesi.

        Args:
            checkpoint_path: percorso al file .pth
            quantized: True se il .pth contiene un modello già quantizzato
                       (serializzato con torch.save dopo quantize_dynamic)
        """
        cfg_model = self.config["model"]

        print(f"📦 Caricamento modello PyTorch da '{checkpoint_path}'...")

        if quantized:
            # Un modello quantizzato viene serializzato/deserializzato
            # con torch.load direttamente (l'oggetto completo, non solo state_dict)
            self._model_pytorch = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False   # necessario per modelli quantizzati
            )
        else:
            # Ricostruisce l'architettura e carica i pesi (state_dict)
            architecture = TomatoCNN(
                n_filters=cfg_model["n_filters"],
                kernel_size=cfg_model["kernel_size"],
                num_blocks=cfg_model["num_blocks"],
                num_classes=cfg_model["num_classes"],
            )
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )
            architecture.load_state_dict(state_dict)
            self._model_pytorch = architecture

        # Modalità inferenza: disattiva Dropout, BatchNorm in eval mode
        self._model_pytorch.eval()
        self._model_pytorch.to(self.device)

        size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
        print(f"   ✓ Modello caricato | dimensione file: {size_mb:.2f} MB")

    def _load_onnx_model(self, onnx_path: str) -> None:
        """
        Carica un modello ONNX tramite onnxruntime.

        Args:
            onnx_path: percorso al file .onnx
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime non installato. "
                "Esegui: pip install onnxruntime"
            )

        print(f"📦 Caricamento modello ONNX da '{onnx_path}'...")

        # Usa solo CPU provider (no GPU, simula Raspberry Pi)
        self._onnx_session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )

        size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
        print(f"   ✓ Modello ONNX caricato | dimensione file: {size_mb:.2f} MB")

    # ─────────────────────────────────────────────────────────────
    # Inferenza principale
    # ─────────────────────────────────────────────────────────────

    def predict(self, image_tensor: torch.Tensor) -> InferenceResult:
        """
        Esegue l'inferenza su un tensor già preprocessato.

        Il tensor deve avere shape (1, 3, 64, 64) e valori normalizzati
        con ImageNet mean/std — esattamente come restituito da
        VirtualCameraSensor.capture().

        Args:
            image_tensor: tensor (1, 3, 64, 64), float32

        Returns:
            InferenceResult con predizione, confidenza e metriche di performance
        """
        if self._onnx_session is not None:
            return self._predict_onnx(image_tensor)
        else:
            return self._predict_pytorch(image_tensor)

    def _predict_pytorch(self, image_tensor: torch.Tensor) -> InferenceResult:
        """
        Inferenza con PyTorch (full_precision o quantizzato INT8).

        torch.no_grad() disattiva il calcolo del grafo dei gradienti,
        che in inferenza è inutile e spreca memoria/tempo.
        """
        tensor = image_tensor.to(self.device)

        # Misura RAM prima dell'inferenza
        ram_before_mb = self._process.memory_info().rss / (1024 ** 2)

        # ── Inferenza ────────────────────────────────────────────
        t_start = time.perf_counter()
        with torch.no_grad():
            logits = self._model_pytorch(tensor)          # shape: (1, 10)
            probabilities = F.softmax(logits, dim=1)      # converte in probabilità
        t_end = time.perf_counter()
        # ─────────────────────────────────────────────────────────

        inference_ms = (t_end - t_start) * 1000.0
        ram_after_mb = self._process.memory_info().rss / (1024 ** 2)

        # Estrae classe predetta
        probs_list = probabilities.squeeze(0).tolist()   # (10,) → lista Python
        predicted_idx = int(probabilities.argmax(dim=1).item())
        confidence = probs_list[predicted_idx]
        predicted_label = self.classes[predicted_idx]

        return InferenceResult(
            predicted_label=predicted_label,
            predicted_idx=predicted_idx,
            confidence=confidence,
            all_probabilities=probs_list,
            inference_time_ms=inference_ms,
            ram_used_mb=max(ram_before_mb, ram_after_mb),
            model_mode=self.loaded_variant,
        )

    def _predict_onnx(self, image_tensor: torch.Tensor) -> InferenceResult:
        """
        Inferenza con ONNX Runtime.

        ONNX Runtime richiede numpy array in input (non tensor PyTorch).
        """
        import numpy as np

        # Converti tensor → numpy float32
        np_input = image_tensor.numpy().astype(np.float32)

        # Nome dell'input del modello ONNX (di solito "input" o "input.1")
        input_name = self._onnx_session.get_inputs()[0].name

        ram_mb = self._process.memory_info().rss / (1024 ** 2)

        # ── Inferenza ONNX ───────────────────────────────────────
        t_start = time.perf_counter()
        outputs = self._onnx_session.run(None, {input_name: np_input})
        t_end = time.perf_counter()
        # ─────────────────────────────────────────────────────────

        inference_ms = (t_end - t_start) * 1000.0

        # outputs[0] ha shape (1, 10) — logits grezzi
        logits = torch.tensor(outputs[0])
        probabilities = F.softmax(logits, dim=1)
        probs_list = probabilities.squeeze(0).tolist()
        predicted_idx = int(probabilities.argmax(dim=1).item())
        confidence = probs_list[predicted_idx]
        predicted_label = self.classes[predicted_idx]

        return InferenceResult(
            predicted_label=predicted_label,
            predicted_idx=predicted_idx,
            confidence=confidence,
            all_probabilities=probs_list,
            inference_time_ms=inference_ms,
            ram_used_mb=ram_mb,
            model_mode=self.loaded_variant,
        )

    # ─────────────────────────────────────────────────────────────
    # Utility / Info
    # ─────────────────────────────────────────────────────────────

    def get_model_size_mb(self) -> float:
        """
        Stima la dimensione del modello in memoria (in MB).
        Per il modello PyTorch conta i parametri, per ONNX usa la dimensione del file.
        """
        if self._model_pytorch is not None:
            total_params = sum(
                p.numel() * p.element_size()
                for p in self._model_pytorch.parameters()
            )
            return total_params / (1024 ** 2)
        return 0.0

    def count_nonzero_params(self) -> int:
        """Conta i parametri non-zero (utile per misurare sparsità dopo pruning)."""
        if self._model_pytorch is None:
            return 0
        total = 0
        for p in self._model_pytorch.parameters():
            total += int(torch.count_nonzero(p).item())
        return total

    def count_total_params(self) -> int:
        """Conta il totale dei parametri del modello."""
        if self._model_pytorch is None:
            return 0
        return sum(p.numel() for p in self._model_pytorch.parameters())

    def get_info(self) -> dict:
        """Restituisce un dizionario con le info del motore di inferenza."""
        return {
            "mode": self.mode,
            "variant": self.loaded_variant,
            "device": str(self.device),
            "num_threads": torch.get_num_threads(),
            "model_size_mb": self.get_model_size_mb(),
            "total_params": self.count_total_params(),
            "nonzero_params": self.count_nonzero_params(),
            "backend": "onnxruntime" if self._onnx_session else "pytorch",
        }

    def __repr__(self) -> str:
        info = self.get_info()
        return (
            f"InferenceEngine("
            f"mode={info['mode']}, "
            f"backend={info['backend']}, "
            f"params={info['total_params']:,}, "
            f"size={info['model_size_mb']:.2f}MB)"
        )
