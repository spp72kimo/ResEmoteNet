import os
import argparse
import json
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from approach.ResEmoteNet import ResEmoteNet


def build_label_mapping(found_class_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map folder names to indices with robustness to case/alias.

    Accepts these canonical class names (7 classes):
    ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

    Will normalize incoming folder names to lowercase and strip spaces.
    For common aliases, we normalize e.g., 'anger'/'Anger' -> 'anger'.
    """
    canonical = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
    canonical_set = set(canonical)

    name_norm = lambda s: s.lower().strip()
    normalized = [name_norm(n) for n in found_class_names]

    # Warn if any class not in canonical set
    unknown = [n for n in normalized if n not in canonical_set]
    if unknown:
        print(f"警告：資料夾類別 {unknown} 不在預期類別 {canonical} 之中。將仍以字母排序索引處理。")

    # Prefer canonical order; include only those present
    final_names = [c for c in canonical if c in normalized]
    # Append any unknowns (sorted) to keep deterministic ordering
    final_names.extend(sorted([n for n in normalized if n not in final_names]))

    class_to_idx = {name: idx for idx, name in enumerate(final_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def create_dataloader(data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, Dict[int, str]]:
    # Discover classes from directories
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"資料夾不存在：{data_dir}")

    found = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not found:
        raise ValueError(f"指定資料夾內沒有情緒子資料夾：{data_dir}")

    class_to_idx, idx_to_class = build_label_mapping(found)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform,
        target_transform=lambda y: y,  # indices already mapped by ImageFolder
    )

    # Overwrite ImageFolder's class_to_idx with our robust mapping
    dataset.class_to_idx = class_to_idx
    dataset.classes = [c for c, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, idx_to_class


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = ResEmoteNet().to(device)
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(weights_path, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, idx_to_class: Dict[int, str]) -> Dict:
    total, correct = 0, 0
    per_class_total = {c: 0 for c in idx_to_class.values()}
    per_class_correct = {c: 0 for c in idx_to_class.values()}

    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        for y, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            true_c = idx_to_class[y]
            per_class_total[true_c] += 1
            if y == p:
                per_class_correct[true_c] += 1

    accuracy = correct / total if total > 0 else 0.0
    per_class_acc = {c: (per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else 0.0)
                     for c in per_class_total}

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'total_samples': total,
        'predictions': all_preds,
        'labels': all_labels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='資料集根目錄（每個情緒一個子資料夾）')
    parser.add_argument('--weights', required=True, help='best_model.pth 路徑')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--out', default='eval_report.json', help='輸出 JSON 報告')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備：{device}")

    loader, idx_to_class = create_dataloader(args.data_dir, args.batch_size, args.num_workers)
    model = load_model(args.weights, device)

    t0 = time.time()
    metrics = evaluate(model, loader, device, idx_to_class)
    metrics['idx_to_class'] = idx_to_class
    metrics['elapsed_sec'] = round(time.time() - t0, 2)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"整體準確率：{metrics['accuracy']:.4f}，共 {metrics['total_samples']} 張")
    print("各類別準確率：")
    for c, acc in metrics['per_class_accuracy'].items():
        print(f"  {c}: {acc:.4f}")
    print(f"報告已輸出：{args.out}")


if __name__ == '__main__':
    main()


