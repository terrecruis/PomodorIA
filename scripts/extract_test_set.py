"""scripts/extract_test_set.py — Estrazione del test set "vero" (holdout 20%).

La VirtualCameraSensor pesca da tutto il dataset (train+test): senza questo
script la CNN si troverebbe a classificare immagini già viste in training
(accuracy ~100% fasulla). Lo script riproduce lo stesso ImageFolder + filtro
"Tomato" + random_split(seed=42) del progetto di training originale, per
estrarre lo stesso 20% di test set in una cartella dedicata.

IMPORTANTE: il dataset di origine deve essere identico (stessi file, stesso
ordine) a quello usato per il training, altrimenti il seed non riproduce lo
stesso split.

Utilizzo:
    python scripts/extract_test_set.py --dataset-root <path> --output-dir ./plantvillage_testset
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets


# ══════════════════════════════════════════════════════════════
# Replica identica della logica di dataset.py (progetto CNN---TOMATO-)
# ══════════════════════════════════════════════════════════════
def find_tomato_folder(root: str) -> str:
    """Trova la cartella che contiene direttamente le sottocartelle Tomato___*."""
    for dirpath, dirnames, _ in os.walk(root):
        tomato_dirs = [d for d in dirnames if d.startswith("Tomato")]
        if len(tomato_dirs) >= 8:
            return dirpath
    raise FileNotFoundError(f"Nessuna cartella Tomato trovata sotto '{root}'")


def build_filtered_dataset(data_dir: str):
    """Ricostruisce l'ImageFolder e il filtro "solo classi Tomato" esattamente
    come nel progetto di training originale (stesso ordine, per random_split)."""
    data_dir_final = find_tomato_folder(data_dir)
    print(f"Cartella dataset trovata: {data_dir_final}")

    full_dataset = datasets.ImageFolder(root=data_dir_final)

    tomato_mask = [name.startswith("Tomato") for name in full_dataset.classes]
    tomato_indices = [i for i, mask in enumerate(tomato_mask) if mask]

    valid_indices = [
        idx for idx, label in enumerate(full_dataset.targets) if label in tomato_indices
    ]
    classes = [full_dataset.classes[i] for i in tomato_indices]

    return full_dataset, valid_indices, classes


def get_test_indices(n_filtered: int, train_ratio: float = 0.8, seed: int = 42):
    """Riproduce lo split originale (stesso train_ratio e seed) per ottenere
    gli stessi indici che finivano nel test set."""
    train_size = int(train_ratio * n_filtered)
    test_size = n_filtered - train_size
    train_subset, test_subset = random_split(
        range(n_filtered),
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return sorted(test_subset.indices), sorted(train_subset.indices)


# ══════════════════════════════════════════════════════════════
# Estrazione fisica delle immagini di test
# ══════════════════════════════════════════════════════════════
def extract_test_set(
    dataset_root: str,
    output_dir: str,
    mode: str = "symlink",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    full_dataset, valid_indices, classes = build_filtered_dataset(dataset_root)
    n_filtered = len(valid_indices)
    print(f"Immagini Tomato totali: {n_filtered} | classi: {len(classes)}")

    test_idx_filtered, train_idx_filtered = get_test_indices(n_filtered, train_ratio, seed)
    print(
        f"Split riprodotto (seed={seed}, train_ratio={train_ratio:.0%}): "
        f"train={len(train_idx_filtered)} | test={len(test_idx_filtered)}"
    )

    overlap = set(test_idx_filtered) & set(train_idx_filtered)
    assert not overlap, f"ERRORE: {len(overlap)} indici in comune tra train e test!"

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    per_class_count = {c: 0 for c in classes}

    for filtered_idx in test_idx_filtered:
        full_idx = valid_indices[filtered_idx]
        src_path, label_idx = full_dataset.samples[full_idx]
        class_name = full_dataset.classes[label_idx]

        dst_class_dir = output_root / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_class_dir / os.path.basename(src_path)

        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()

        if mode == "symlink":
            dst_path.symlink_to(os.path.abspath(src_path))
        elif mode == "copy":
            shutil.copy2(src_path, dst_path)
        else:
            raise ValueError(f"Modalità non valida: '{mode}' (usa 'symlink' o 'copy')")

        per_class_count[class_name] += 1
        manifest_rows.append(
            {
                "filtered_idx": filtered_idx,
                "full_dataset_idx": full_idx,
                "class": class_name,
                "src_path": os.path.abspath(src_path),
                "dst_path": str(dst_path),
            }
        )

    manifest_path = output_root / "test_set_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filtered_idx", "full_dataset_idx", "class", "src_path", "dst_path"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nTest set estratto in '{output_root}' (mode={mode})")
    print(f"Manifest salvato in '{manifest_path}'")
    print("\nImmagini di test per classe:")
    for c, n in sorted(per_class_count.items()):
        print(f"   {c:<45} {n:>5}")
    print(f"   {'TOTALE':<45} {sum(per_class_count.values()):>5}")


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Estrae il test set holdout (seed=42, 80/20) usato nel "
        "training originale, per evitare che PomodorIA valuti la CNN su "
        "immagini già viste in training."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path alla cartella del dataset PlantVillage-Tomato originale "
        "(la stessa usata per il training, es. la cache kagglehub)",
    )
    parser.add_argument(
        "--output-dir",
        default="./plantvillage_testset",
        help="Cartella di destinazione per il test set estratto (default: ./plantvillage_testset)",
    )
    parser.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="'symlink' (default, non duplica spazio su disco) oppure 'copy'",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_test_set(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        mode=args.mode,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
