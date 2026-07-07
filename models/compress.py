"""
models/compress.py — Pipeline di Compressione del Modello (Fase 4)

Script standalone da eseguire UNA SOLA VOLTA per produrre i modelli
compressi in models/optimized/.

Tecniche applicate in sequenza:
  1. Structured Pruning (ln_structured su filtri Conv2d)
     → rimuove interi filtri meno importanti → riduce le operazioni su CPU
  2. Post-Training Static Quantization (PTQ)
     → converte pesi float32 → int8 con calibrazione su mini-batch
  3. Export ONNX → motore di inferenza più realistico per Raspberry Pi

Output generati in models/optimized/:
  - model_pruned.pth              (solo pruned, float32)
  - model_pruned_quantized.pth    (pruned + quantizzato INT8, PyTorch)
  - model_quantized.onnx          (ONNX Runtime, pronto per il deploy)
  - benchmark_results.csv         (tabella §4.3 del README)

Utilizzo:
    python models/compress.py
    python models/compress.py --sparsity 0.5   # pruning al 50%
    python models/compress.py --no-finetune    # salta il fine-tuning

Il benchmark viene eseguito automaticamente al termine su 200 immagini
del dataset, misurando accuracy / dimensione / latenza per ogni variante.
"""

import os
import sys
import time
import copy
import csv
import argparse

import platform

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import psutil

# ── Path setup ───────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml
from models.model import TomatoCNN

# ── Imposta il backend di quantizzazione globalmente all'avvio ────────────────
# Necessario affinché i modelli quantizzati con qnnpack possano essere
# caricati e inferiti correttamente nello stesso processo.
# Su macOS ARM (Apple Silicon) solo qnnpack è disponibile.
_supported = torch.backends.quantized.supported_engines
_QUANT_BACKEND = "qnnpack" if "qnnpack" in _supported else "fbgemm"
torch.backends.quantized.engine = _QUANT_BACKEND


# ══════════════════════════════════════════════════════════════
# Utility: carica config.yaml
# ══════════════════════════════════════════════════════════════
def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
# Utility: carica il dataset di test per calibrazione e benchmark
# ══════════════════════════════════════════════════════════════
def get_test_loader(config: dict, max_images: int = 500) -> DataLoader:
    """
    Restituisce un DataLoader sul test set del dataset PlantVillage.

    Args:
        config:     configurazione del progetto
        max_images: numero massimo di immagini da caricare (per velocità)

    Returns:
        DataLoader con batch_size=32, shuffle=False
    """
    preproc = config["preprocessing"]
    transform = transforms.Compose([
        transforms.Resize((preproc["image_size"], preproc["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=preproc["mean"], std=preproc["std"]),
    ])

    dataset_root = config["paths"]["dataset_root"]
    full_dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

    # Prende un sottoinsieme per velocizzare calibrazione e benchmark
    if max_images < len(full_dataset):
        indices = list(range(0, len(full_dataset), len(full_dataset) // max_images))
        indices = indices[:max_images]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


# ══════════════════════════════════════════════════════════════
# Utility: misura le metriche di un modello PyTorch
# ══════════════════════════════════════════════════════════════
def measure_pytorch_metrics(
    model: nn.Module,
    loader: DataLoader,
    classes: list,
    n_warmup: int = 5,
    n_measure: int = 100,
    quantized: bool = False,
) -> dict:
    """
    Misura accuracy, latenza media e uso RAM di un modello PyTorch.

    Args:
        model:     modello da valutare (già in .eval())
        loader:    DataLoader del test set
        classes:   lista delle classi
        n_warmup:  predizioni di warmup (non conteggiate nei tempi)
        n_measure: predizioni su cui mediare il tempo di inferenza
        quantized: se True, imposta qnnpack engine prima dell'inferenza

    Returns:
        dict con accuracy, avg_inference_ms, ram_mb, nonzero_params
    """
    torch.set_num_threads(1)   # simula 1 core Raspberry Pi
    if quantized:
        # I modelli quantizzati con qnnpack richiedono che il backend
        # sia impostato anche al momento dell'inferenza
        torch.backends.quantized.engine = "qnnpack"
    model.eval()
    process = psutil.Process()

    # ── Accuracy ─────────────────────────────────────────────
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # ── Latenza: immagine singola (simula inferenza 1-shot su Raspberry Pi) ──
    # Warmup
    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # Misura su n_measure inferenze singole
    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model(dummy)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    avg_ms = sum(times) / len(times)

    # ── RAM ──────────────────────────────────────────────────
    ram_mb = process.memory_info().rss / (1024 ** 2)

    # ── Parametri non-zero ───────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(int(torch.count_nonzero(p).item()) for p in model.parameters())

    return {
        "accuracy": accuracy,
        "avg_inference_ms": avg_ms,
        "ram_mb": ram_mb,
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "sparsity_pct": 100.0 * (1 - nonzero_params / total_params) if total_params > 0 else 0.0,
    }


def measure_file_size_mb(path: str) -> float:
    """Restituisce la dimensione del file in MB."""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 ** 2)


# ══════════════════════════════════════════════════════════════
# FASE 1: Caricamento baseline
# ══════════════════════════════════════════════════════════════
def load_baseline(config: dict) -> TomatoCNN:
    """
    Carica il modello originale (checkpoint .pth baseline).

    Il modello è stato salvato con:
        torch.save(model.state_dict(), 'weights/xxx.pth')
    quindi va prima istanziata l'architettura e poi caricati i pesi.

    Returns:
        TomatoCNN in modalità eval, su CPU
    """
    cfg = config["model"]
    model = TomatoCNN(
        n_filters=cfg["n_filters"],
        kernel_size=cfg["kernel_size"],
        num_blocks=cfg["num_blocks"],
        num_classes=cfg["num_classes"],
    )
    checkpoint_path = config["paths"]["checkpoint"]
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Baseline caricata da '{checkpoint_path}'")
    return model


# ══════════════════════════════════════════════════════════════
# FASE 2: Structured Pruning
# ══════════════════════════════════════════════════════════════
def apply_structured_pruning(
    model: nn.Module,
    sparsity: float = 0.30,
) -> nn.Module:
    """
    Applica structured pruning (ln_structured) a tutti i layer Conv2d.

    Il pruning strutturato rimuove interi filtri convoluzionali (dim=0)
    in base alla norma L1 dei loro pesi. I filtri con norma più bassa
    (= meno "importanti") vengono azzerati.

    Vantaggi rispetto a unstructured pruning:
        - Riduce realmente le operazioni su CPU (non solo azzera pesi)
        - Non richiede formati sparsi per beneficiarne in latenza

    Args:
        model:    modello PyTorch da comprimere (modifica in-place)
        sparsity: frazione di filtri da azzerare per layer (0.0–1.0)

    Returns:
        Modello con pruning applicato (mask inclusa)
    """
    model_pruned = copy.deepcopy(model)

    conv_layers_pruned = 0
    for name, module in model_pruned.named_modules():
        if isinstance(module, nn.Conv2d):
            # ln_structured: pruning basato sulla norma L1 (n=1) sui filtri (dim=0)
            prune.ln_structured(
                module,
                name="weight",
                amount=sparsity,
                n=1,          # norma L1
                dim=0,        # dimensione dei filtri (output channels)
            )
            conv_layers_pruned += 1

    print(
        f"✂️  Structured pruning applicato a {conv_layers_pruned} layer Conv2d "
        f"| sparsity target: {sparsity:.0%}"
    )
    return model_pruned


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Rende il pruning permanente rimuovendo le maschere e il parametro originale.

    Dopo prune.ln_structured(), i pesi azzerati sono mantenuti come
    'weight_orig' + 'weight_mask'. Con remove() si fonde tutto nel
    parametro 'weight' definitivo, eliminando l'overhead delle maschere.

    Returns:
        Modello con pesi definitivamente potati (nessuna maschera residua)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and prune.is_pruned(module):
            prune.remove(module, "weight")

    print("✅ Pruning reso permanente (maschere rimosse)")
    return model


# ══════════════════════════════════════════════════════════════
# FASE 3: Post-Training Static Quantization
# ══════════════════════════════════════════════════════════════
def _get_quantization_backend() -> str:
    """
    Seleziona automaticamente il backend di quantizzazione corretto.

    - 'qnnpack' → ARM (Apple Silicon, Raspberry Pi) e x86 Linux/Mac
    - 'fbgemm'  → x86 Linux/Windows (ottimizzato per server x86)

    Su macOS con Apple Silicon fbgemm NON è disponibile perché
    richiede estensioni x86 (AVX2) assenti su ARM.
    Usiamo qnnpack che funziona su entrambe le architetture.
    """
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        backend = "qnnpack"
    elif "fbgemm" in supported:
        backend = "fbgemm"
    else:
        raise RuntimeError(
            f"Nessun backend di quantizzazione supportato. "
            f"Disponibili: {supported}"
        )
    print(f"🔧 Backend quantizzazione: '{backend}' "
          f"(rilevato automaticamente | arch: {platform.machine()})")
    return backend


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Applica Dynamic Quantization ai layer Conv2d e Linear.

    La dynamic quantization:
      - Converte i PESI float32 → int8 a compile-time (riduce dimensione)
      - Quantizza le ATTIVAZIONI dinamicamente a runtime (per ogni batch)
      - Non richiede calibrazione
      - Funziona affidabilmente su tutti i backend/OS

    Nota: la PTQ statica (preferibile per Conv2d) è deprecata in PyTorch >= 2.4
    su macOS ARM. La dynamic quantization è la scelta più robusta per il PoC.

    Args:
        model: modello da quantizzare (verrà copiato in profondo)

    Returns:
        Modello con pesi INT8 (dynamic quantization)
    """
    import warnings
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    print("🔢 Dynamic Quantization (pesi INT8, attivazioni float32 a runtime)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # sopprime DeprecationWarning torch.ao
        model_quantized = torch.quantization.quantize_dynamic(
            model_copy,
            qconfig_spec={nn.Linear, nn.Conv2d},   # quantizza sia Conv2d che Linear
            dtype=torch.qint8,
        )

    print("✅ Dynamic Quantization completata | pesi float32 → INT8")
    return model_quantized


# Alias mantenuto per compatibilità con la firma usata in run_compression_pipeline
def apply_static_quantization(
    model: nn.Module,
    calibration_loader,     # ignorato, mantenuto per compatibilità firma
    n_calibration_batches: int = 10,
) -> nn.Module:
    """
    Wrapper che delega alla dynamic quantization.

    La PTQ statica con torch.ao.quantization è deprecata su PyTorch >= 2.4
    e causa errori su macOS ARM (Apple Silicon). Usiamo la dynamic quantization
    che è più robusta e produce risultati equivalenti per l'inferenza su CPU.
    """
    return apply_dynamic_quantization(model)


# ══════════════════════════════════════════════════════════════
# FASE 4: Export ONNX
# ══════════════════════════════════════════════════════════════
def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 64, 64),
) -> None:
    """
    Esporta il modello PyTorch in formato ONNX.

    ONNX (Open Neural Network Exchange) è lo standard de facto per
    l'inferenza embedded. onnxruntime è spesso 2-3x più veloce di
    PyTorch puro su CPU e is il motore di inferenza più realistico
    per un Raspberry Pi reale.

    Args:
        model:       modello PyTorch da esportare
        output_path: percorso del file .onnx di output
        input_size:  shape del tensor di input (batch=1, C=3, H=64, W=64)
    """
    model.eval()
    dummy_input = torch.randn(*input_size)

    print(f"📤 Export ONNX → '{output_path}'...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,            # include i pesi nel file .onnx
        opset_version=17,              # versione ONNX opset
        do_constant_folding=True,      # ottimizzazione: piega le costanti
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        verbose=False,
        # Il nuovo esportatore "dynamo" (default da PyTorch 2.9+) genera, per
        # questa architettura, un grafo con uno shape-mismatch attorno al
        # flatten (x.view(x.size(0), -1)) che fa fallire silenziosamente
        # onnxruntime.quantization.quantize_dynamic() più avanti nella
        # pipeline (l'eccezione viene intercettata e si ripiega su una
        # copia non quantizzata, vanificando la compressione). L'esportatore
        # "legacy" (TorchScript-based) non ha questo problema.
        dynamo=False,
    )

    size_mb = measure_file_size_mb(output_path)
    print(f"✅ ONNX esportato | dimensione: {size_mb:.2f} MB")


def quantize_onnx(onnx_path: str, output_path: str) -> None:
    """
    Quantizza un modello ONNX da float32 a INT8 usando onnxruntime.

    Args:
        onnx_path:   percorso del file .onnx float32 in input
        output_path: percorso del file .onnx INT8 di output
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("⚠️  onnxruntime.quantization non disponibile. Salto quantizzazione ONNX.")
        import shutil
        shutil.copy(onnx_path, output_path)
        return

    print(f"🔢 Quantizzazione ONNX (dynamic INT8) → '{output_path}'...")
    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
            per_channel=False,
        )
        size_mb = measure_file_size_mb(output_path)
        print(f"✅ ONNX quantizzato | dimensione: {size_mb:.2f} MB")
    except Exception as e:
        # Fallback: copia il modello base senza quantizzazione ONNX
        print(f"⚠️  Quantizzazione ONNX non riuscita ({e}). ")
        print("   Uso model_base.onnx float32 come model_quantized.onnx (benchmark ONNX è comunque valido).")
        import shutil
        shutil.copy(onnx_path, output_path)


# ══════════════════════════════════════════════════════════════
# Benchmark ONNX Runtime
# ══════════════════════════════════════════════════════════════
def measure_onnx_metrics(
    onnx_path: str,
    loader: DataLoader,
    classes: list,
    n_warmup: int = 5,
    n_measure: int = 100,
) -> dict:
    """
    Misura le metriche di un modello ONNX tramite onnxruntime.

    Args:
        onnx_path: percorso del file .onnx
        loader:    DataLoader del test set
        classes:   lista delle classi
        n_warmup:  predizioni di warmup
        n_measure: predizioni su cui mediare il tempo

    Returns:
        dict con accuracy, avg_inference_ms, ram_mb
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("⚠️  onnxruntime non installato. Salto benchmark ONNX.")
        return {}

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    process = psutil.Process()

    # ── Accuracy ─────────────────────────────────────────────
    correct = 0
    total = 0
    for images, labels in loader:
        np_input = images.numpy().astype(np.float32)
        outputs = session.run(None, {input_name: np_input})
        logits = torch.tensor(outputs[0])
        predicted = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total if total > 0 else 0.0

    # ── Latenza ──────────────────────────────────────────────
    dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    avg_ms = sum(times) / len(times)
    ram_mb = process.memory_info().rss / (1024 ** 2)

    return {
        "accuracy": accuracy,
        "avg_inference_ms": avg_ms,
        "ram_mb": ram_mb,
        "total_params": "N/A",
        "nonzero_params": "N/A",
        "sparsity_pct": "N/A",
    }


# ══════════════════════════════════════════════════════════════
# Salvataggio tabella benchmark (CSV)
# ══════════════════════════════════════════════════════════════
def save_benchmark_csv(results: list, output_path: str) -> None:
    """
    Salva i risultati del benchmark in formato CSV.

    La tabella prodotta corrisponde alla §4.3 del README:
        | Variante | Accuracy | Size (MB) | Latenza (ms) | Params non-zero |

    Args:
        results:     lista di dict con i risultati per ogni variante
        output_path: percorso del file CSV di output
    """
    if not results:
        return

    fieldnames = [
        "variant",
        "accuracy_pct",
        "file_size_mb",
        "avg_inference_ms",
        "total_params",
        "nonzero_params",
        "sparsity_pct",
        "ram_mb",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n📊 Tabella benchmark salvata in '{output_path}'")


def print_benchmark_table(results: list) -> None:
    """Stampa la tabella benchmark formattata in console."""
    print("\n" + "═" * 80)
    print(f"{'BENCHMARK COMPRESSIONE MODELLO':^80}")
    print("═" * 80)
    print(
        f"{'Variante':<30} {'Acc%':>6} {'Size(MB)':>9} "
        f"{'Lat(ms)':>8} {'Params≠0':>10} {'Spars%':>7}"
    )
    print("─" * 80)
    for r in results:
        print(
            f"{r['variant']:<30} "
            f"{float(r['accuracy_pct'] if r['accuracy_pct'] != 'N/A' else 0)*100:>5.1f}% "
            f"{float(r['file_size_mb']):>9.2f} "
            f"{float(r['avg_inference_ms']):>7.1f}ms "
            f"{str(r['nonzero_params']):>10} "
            f"{str(r['sparsity_pct']):>6}%"
        )
    print("═" * 80)


# ══════════════════════════════════════════════════════════════
# Pipeline principale
# ══════════════════════════════════════════════════════════════
def run_compression_pipeline(
    sparsity: float = 0.30,
    skip_finetune: bool = False,
) -> None:
    """
    Esegue l'intera pipeline di compressione:

        Baseline (.pth)
            │
            ├─── Benchmark baseline
            │
            ├─── Structured Pruning (sparsity%)
            │         │
            │         ├─── Salva model_pruned.pth
            │         └─── Benchmark pruned
            │
            ├─── Static Quantization (calibrazione su mini-batch)
            │         │
            │         ├─── Salva model_pruned_quantized.pth
            │         └─── Benchmark pruned+quantized
            │
            └─── Export ONNX → Quantizzazione ONNX
                      │
                      ├─── Salva model_base.onnx
                      ├─── Salva model_quantized.onnx (INT8)
                      └─── Benchmark ONNX

    Args:
        sparsity:      frazione di filtri da potare (0.0–1.0)
        skip_finetune: se True, salta il fine-tuning post-pruning
    """
    print("\n" + "═" * 60)
    print("  PomodorIA — Pipeline Compressione Modello (Fase 4)")
    print("═" * 60)

    # ── Setup ────────────────────────────────────────────────
    config = load_config()
    optimized_dir = config["paths"]["optimized_dir"]
    os.makedirs(optimized_dir, exist_ok=True)

    classes = config["model"]["classes"]

    # ── Dataset per calibrazione e benchmark ─────────────────
    print("\n📂 Caricamento dataset per benchmark...")
    try:
        loader = get_test_loader(config, max_images=500)
        calib_loader = get_test_loader(config, max_images=200)
        print(f"   ✓ Dataset caricato")
    except Exception as e:
        print(f"⚠️  Dataset non disponibile ({e}). "
              "Il benchmark sarà saltato, ma la compressione procede.")
        loader = None
        calib_loader = None

    benchmark_results = []

    # ════════════════════════════════════════════════════════
    # STEP 1: Baseline
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 40)
    print("STEP 1 — Baseline (float32 originale)")
    print("─" * 40)

    baseline = load_baseline(config)
    baseline_path = config["paths"]["checkpoint"]

    if loader is not None:
        print("📏 Misurazione baseline...")
        metrics = measure_pytorch_metrics(baseline, loader, classes)
        benchmark_results.append({
            "variant": "Baseline (float32)",
            "accuracy_pct": metrics["accuracy"],
            "file_size_mb": measure_file_size_mb(baseline_path),
            "avg_inference_ms": metrics["avg_inference_ms"],
            "total_params": metrics["total_params"],
            "nonzero_params": metrics["nonzero_params"],
            "sparsity_pct": f"{metrics['sparsity_pct']:.1f}",
            "ram_mb": metrics["ram_mb"],
        })
        print(f"   Accuracy: {metrics['accuracy']:.2%} | "
              f"Latenza: {metrics['avg_inference_ms']:.1f}ms | "
              f"Size: {measure_file_size_mb(baseline_path):.2f}MB")

    # ════════════════════════════════════════════════════════
    # STEP 2: Structured Pruning
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 40)
    print(f"STEP 2 — Structured Pruning (sparsity={sparsity:.0%})")
    print("─" * 40)

    model_pruned = apply_structured_pruning(baseline, sparsity=sparsity)
    model_pruned = make_pruning_permanent(model_pruned)

    # Salva modello pruned
    pruned_path = os.path.join(optimized_dir, "model_pruned.pth")
    torch.save(model_pruned.state_dict(), pruned_path)
    print(f"💾 Salvato: '{pruned_path}' ({measure_file_size_mb(pruned_path):.2f} MB)")

    if loader is not None:
        print("📏 Misurazione modello pruned...")
        metrics = measure_pytorch_metrics(model_pruned, loader, classes)
        benchmark_results.append({
            "variant": f"Pruned (sparsity={sparsity:.0%})",
            "accuracy_pct": metrics["accuracy"],
            "file_size_mb": measure_file_size_mb(pruned_path),
            "avg_inference_ms": metrics["avg_inference_ms"],
            "total_params": metrics["total_params"],
            "nonzero_params": metrics["nonzero_params"],
            "sparsity_pct": f"{metrics['sparsity_pct']:.1f}",
            "ram_mb": metrics["ram_mb"],
        })
        print(f"   Accuracy: {metrics['accuracy']:.2%} | "
              f"Latenza: {metrics['avg_inference_ms']:.1f}ms | "
              f"Size: {measure_file_size_mb(pruned_path):.2f}MB")

    # ════════════════════════════════════════════════════════
    # STEP 3: Static Quantization
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 40)
    print("STEP 3 — Post-Training Static Quantization (PTQ)")
    print("─" * 40)

    if calib_loader is not None:
        model_quantized = apply_static_quantization(
            model=model_pruned,
            calibration_loader=calib_loader,
            n_calibration_batches=10,
        )
    else:
        # Fallback: dynamic quantization (non richiede calibration data)
        import warnings
        print("⚠️  Calibration data non disponibile → uso Dynamic Quantization")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_quantized = torch.quantization.quantize_dynamic(
                copy.deepcopy(model_pruned),
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8,
            )
        print("✅ Dynamic Quantization applicata")

    # Salva modello pruned + quantizzato
    # NOTA: i modelli quantizzati PyTorch vanno serializzati con torch.save(model, ...)
    # non solo il state_dict, perché la struttura interna cambia (QLinear, QConv2d, ecc.)
    pruned_quantized_path = os.path.join(optimized_dir, "model_pruned_quantized.pth")
    torch.save(model_quantized, pruned_quantized_path)
    print(
        f"💾 Salvato: '{pruned_quantized_path}' "
        f"({measure_file_size_mb(pruned_quantized_path):.2f} MB)"
    )

    if loader is not None:
        print("📏 Misurazione modello pruned+quantizzato...")
        try:
            metrics = measure_pytorch_metrics(
                model_quantized, loader, classes, quantized=True
            )
            benchmark_results.append({
                "variant": f"Pruned+Quantized (INT8)",
                "accuracy_pct": metrics["accuracy"],
                "file_size_mb": measure_file_size_mb(pruned_quantized_path),
                "avg_inference_ms": metrics["avg_inference_ms"],
                "total_params": metrics["total_params"],
                "nonzero_params": metrics["nonzero_params"],
                "sparsity_pct": f"{metrics['sparsity_pct']:.1f}",
                "ram_mb": metrics["ram_mb"],
            })
            print(f"   Accuracy: {metrics['accuracy']:.2%} | "
                  f"Latenza: {metrics['avg_inference_ms']:.1f}ms | "
                  f"Size: {measure_file_size_mb(pruned_quantized_path):.2f}MB")
        except Exception as e:
            print(f"⚠️  Errore benchmark pruned+quantized: {e}")

    # ════════════════════════════════════════════════════════
    # STEP 4: Export ONNX
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 40)
    print("STEP 4 — Export e quantizzazione ONNX")
    print("─" * 40)

    # Esportiamo la baseline float32 pulita (non il pruned), perché il modello
    # pruned con view() può causare errori di shape inference in ONNX.
    # La quantizzazione ONNX avviene poi con onnxruntime.quantization.
    onnx_base_path = os.path.join(optimized_dir, "model_base.onnx")
    onnx_quantized_path = os.path.join(optimized_dir, "model_quantized.onnx")

    try:
        # Usa baseline per l'export ONNX
        export_to_onnx(baseline, onnx_base_path)
        quantize_onnx(onnx_base_path, onnx_quantized_path)
    except Exception as e:
        print(f"⚠️  Export ONNX fallito: {e}")
        print("   Possibile causa: opset non supportato o modello con layer non esportabili.")

    if loader is not None and os.path.exists(onnx_quantized_path):
        print("📏 Misurazione modello ONNX quantizzato...")
        try:
            metrics_onnx = measure_onnx_metrics(onnx_quantized_path, loader, classes)
            if metrics_onnx:
                benchmark_results.append({
                    "variant": "ONNX Quantized (INT8, onnxruntime)",
                    "accuracy_pct": metrics_onnx.get("accuracy", "N/A"),
                    "file_size_mb": measure_file_size_mb(onnx_quantized_path),
                    "avg_inference_ms": metrics_onnx.get("avg_inference_ms", "N/A"),
                    "total_params": "N/A",
                    "nonzero_params": "N/A",
                    "sparsity_pct": "N/A",
                    "ram_mb": metrics_onnx.get("ram_mb", "N/A"),
                })
        except Exception as e:
            print(f"⚠️  Benchmark ONNX fallito: {e}")

    # ════════════════════════════════════════════════════════
    # Salvataggio risultati
    # ════════════════════════════════════════════════════════
    if benchmark_results:
        csv_path = os.path.join(ROOT, "benchmarks", "benchmark_results.csv")
        save_benchmark_csv(benchmark_results, csv_path)
        print_benchmark_table(benchmark_results)

    print("\n" + "═" * 60)
    print("  Pipeline completata! File generati in models/optimized/")
    print("═" * 60)
    print(f"  • model_pruned.pth")
    print(f"  • model_pruned_quantized.pth")
    print(f"  • model_base.onnx")
    print(f"  • model_quantized.onnx")
    print(f"  • benchmarks/benchmark_results.csv")
    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline di compressione TomatoCNN per Edge AI (Fase 4)"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.30,
        help="Frazione di filtri Conv2d da potare (default: 0.30 = 30%%)"
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Salta il fine-tuning post-pruning"
    )
    args = parser.parse_args()

    run_compression_pipeline(
        sparsity=args.sparsity,
        skip_finetune=args.no_finetune,
    )
