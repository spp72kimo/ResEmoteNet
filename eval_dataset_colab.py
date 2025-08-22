import os
import argparse
import json
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from approach.ResEmoteNet import ResEmoteNet


def mount_google_drive():
    """掛載 Google Drive 到 Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive 已成功掛載到 /content/drive")
        return True
    except ImportError:
        print("⚠️  不在 Colab 環境中，跳過 Drive 掛載")
        return False


def get_drive_path(base_path: str = None) -> str:
    """獲取 Google Drive 路徑"""
    if base_path:
        return os.path.join('/content/drive/MyDrive', base_path)
    return '/content/drive/MyDrive'


def create_drive_folder(folder_name: str) -> str:
    """在 Google Drive 中創建資料夾"""
    drive_path = get_drive_path()
    folder_path = os.path.join(drive_path, folder_name)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 已在 Google Drive 中創建資料夾：{folder_name}")
    
    return folder_path


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


def create_dataloader(data_dir: str, batch_size: int, num_workers: int, max_samples: int = None) -> Tuple[DataLoader, Dict[int, str]]:
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

    # 如果指定了最大樣本數，則創建子集
    if max_samples and max_samples < len(dataset):
        print(f"原始數據集大小：{len(dataset)}，限制為：{max_samples} 個樣本")
        # 為了保持類別平衡，我們從每個類別中選擇樣本
        indices = []
        for class_idx in range(len(idx_to_class)):
            class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
            if class_indices:
                # 從每個類別中選擇 min(max_samples_per_class, len(class_indices)) 個樣本
                max_per_class = max(1, max_samples // len(idx_to_class))
                selected = class_indices[:min(max_per_class, len(class_indices))]
                indices.extend(selected)
        
        # 如果總數超過限制，則截斷
        if len(indices) > max_samples:
            indices = indices[:max_samples]
        
        dataset = Subset(dataset, indices)
        print(f"實際選擇樣本數：{len(dataset)}")

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


def calculate_metrics(y_true: List[int], y_pred: List[int], idx_to_class: Dict[int, str]) -> Dict:
    """計算詳細的評估指標"""
    # 轉換為 numpy 數組
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 計算每個類別的 precision, recall, f1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(idx_to_class.keys())
    )
    
    # 計算整體的 macro 和 weighted 平均值
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_true, y_pred, labels=list(idx_to_class.keys()))
    
    # 計算每個類別的準確率
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # 計算整體準確率
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    
    return {
        'overall_accuracy': overall_accuracy,
        'per_class_accuracy': {idx_to_class[i]: acc for i, acc in enumerate(per_class_accuracy)},
        'per_class_precision': {idx_to_class[i]: prec for i, prec in enumerate(precision)},
        'per_class_recall': {idx_to_class[i]: rec for i, rec in enumerate(recall)},
        'per_class_f1': {idx_to_class[i]: f1_score for i, f1_score in enumerate(f1)},
        'per_class_support': {idx_to_class[i]: sup for i, sup in enumerate(support)},
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': [idx_to_class[i] for i in range(len(idx_to_class))]
    }

def calculate_roc_auc(y_true: List[int], y_pred_probs: np.ndarray, idx_to_class: Dict[int, str]) -> Dict:
    """計算 ROC 曲線和 AUC 值"""
    # 二值化標籤（one-vs-rest）
    y_true_bin = label_binarize(y_true, classes=list(idx_to_class.keys()))
    
    # 計算每個類別的 ROC 和 AUC
    roc_data = {}
    auc_scores = {}
    
    for i, class_name in idx_to_class.items():
        if len(np.unique(y_true_bin[:, i])) > 1:  # 確保有兩個類別
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            auc_score = auc(fpr, tpr)
            roc_data[class_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            auc_scores[class_name] = auc_score
    
    # 計算 micro-average ROC 和 AUC
    y_true_bin_ravel = y_true_bin.ravel()
    y_pred_probs_ravel = y_pred_probs.ravel()
    if len(np.unique(y_true_bin_ravel)) > 1:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin_ravel, y_pred_probs_ravel)
        auc_micro = auc(fpr_micro, tpr_micro)
        roc_data['micro'] = {'fpr': fpr_micro.tolist(), 'tpr': tpr_micro.tolist()}
        auc_scores['micro'] = auc_micro
    
    return {
        'roc_data': roc_data,
        'auc_scores': auc_scores
    }

def measure_computational_efficiency(model: torch.nn.Module, loader: DataLoader, device: torch.device, 
                                   num_runs: int = 10) -> Dict:
    """測量計算效率：推理時間、模型大小、FLOPs"""
    # 模型大小
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 模型大小（MB）
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    # 推理時間測量
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            # 使用一個批次進行測試
            try:
                batch = next(iter(loader))
                images = batch[0].to(device)
                
                start_time = time.time()
                _ = model(images)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                
                inference_time = (end_time - start_time) * 1000  # 轉換為毫秒
                inference_times.append(inference_time)
            except StopIteration:
                break
    
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    std_inference_time = np.std(inference_times) if inference_times else 0
    
    # 估算 FLOPs（簡化版本）
    # 注意：這是一個粗略估算，實際的 FLOPs 計算需要更複雜的工具
    sample_input = torch.randn(1, 3, 64, 64).to(device)
    try:
        with torch.no_grad():
            _ = model(sample_input)
        # 這裡可以添加更精確的 FLOPs 計算
        estimated_flops = "需要安裝 thop 或 ptflops 進行精確計算"
    except:
        estimated_flops = "計算失敗"
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(model_size_mb, 2),
        'avg_inference_time_ms': round(avg_inference_time, 2),
        'std_inference_time_ms': round(std_inference_time, 2),
        'estimated_flops': estimated_flops
    }

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], save_path: str = 'confusion_matrix.png', 
                         drive_folder: str = None):
    """繪製混淆矩陣"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # 保存到本地和 Drive
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩陣已保存至：{save_path}")
    
    # 如果指定了 Drive 資料夾，也保存到 Drive
    if drive_folder:
        drive_path = os.path.join(drive_folder, os.path.basename(save_path))
        plt.savefig(drive_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩陣已保存至 Google Drive：{drive_path}")
    
    plt.close()


def plot_roc_curves(roc_data: Dict, auc_scores: Dict, save_path: str = 'roc_curves.png',
                    drive_folder: str = None):
    """繪製 ROC 曲線"""
    plt.figure(figsize=(12, 8))
    
    for class_name, data in roc_data.items():
        if class_name != 'micro':
            plt.plot(data['fpr'], data['tpr'], 
                    label=f'{class_name} (AUC = {auc_scores[class_name]:.3f})')
    
    # 繪製 micro-average ROC 曲線
    if 'micro' in roc_data:
        plt.plot(roc_data['micro']['fpr'], roc_data['micro']['tpr'], 
                label=f'Micro-average (AUC = {auc_scores["micro"]:.3f})', 
                color='black', linestyle='--', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('ROC 曲線 (ROC Curves)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存到本地和 Drive
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC 曲線已保存至：{save_path}")
    
    # 如果指定了 Drive 資料夾，也保存到 Drive
    if drive_folder:
        drive_path = os.path.join(drive_folder, os.path.basename(save_path))
        plt.savefig(drive_path, dpi=300, bbox_inches='tight')
        print(f"ROC 曲線已保存至 Google Drive：{drive_path}")
    
    plt.close()

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, idx_to_class: Dict[int, str]) -> Dict:
    total, correct = 0, 0
    per_class_total = {c: 0 for c in idx_to_class.values()}
    per_class_correct = {c: 0 for c in idx_to_class.values()}

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    dataset_size = len(loader.dataset)
    num_batches = len(loader)
    print(f"資料集大小：{dataset_size}，批次數：{num_batches}，batch_size：{loader.batch_size}")

    for images, labels in tqdm(loader, desc='Evaluating', total=num_batches, leave=True):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())

        for y, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            true_c = idx_to_class[y]
            per_class_total[true_c] += 1
            if y == p:
                per_class_correct[true_c] += 1

    # 計算詳細指標
    metrics = calculate_metrics(all_labels, all_preds, idx_to_class)
    
    # 計算 ROC 和 AUC
    all_probs_array = np.array(all_probs)
    roc_auc_data = calculate_roc_auc(all_labels, all_probs_array, idx_to_class)
    
    # 合併結果
    metrics.update(roc_auc_data)
    metrics['total_samples'] = total
    metrics['predictions'] = all_preds
    metrics['labels'] = all_labels

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='資料集根目錄（每個情緒一個子資料夾）')
    parser.add_argument('--weights', required=True, help='best_model.pth 路徑')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None, help='限制測試的最大樣本數（用於快速測試）')
    parser.add_argument('--out', default='eval_report.json', help='輸出 JSON 報告')
    parser.add_argument('--plot', action='store_true', help='是否生成圖表')
    parser.add_argument('--drive_folder', default='ResEmoteNet_Results', help='Google Drive 中的資料夾名稱')
    parser.add_argument('--save_to_drive', action='store_true', help='是否保存結果到 Google Drive')
    args = parser.parse_args()

    # 嘗試掛載 Google Drive
    drive_mounted = False
    if args.save_to_drive:
        drive_mounted = mount_google_drive()
        if drive_mounted:
            # 創建 Drive 資料夾
            drive_results_folder = create_drive_folder(args.drive_folder)
        else:
            print("⚠️  無法掛載 Google Drive，將只保存到本地")
            args.save_to_drive = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備：{device}")

    loader, idx_to_class = create_dataloader(args.data_dir, args.batch_size, args.num_workers, args.max_samples)
    model = load_model(args.weights, device)

    # 測量計算效率
    print("測量計算效率...")
    efficiency_metrics = measure_computational_efficiency(model, loader, device)
    
    t0 = time.time()
    metrics = evaluate(model, loader, device, idx_to_class)
    metrics['idx_to_class'] = idx_to_class
    metrics['elapsed_sec'] = round(time.time() - t0, 2)
    
    # 合併計算效率指標
    metrics['computational_efficiency'] = efficiency_metrics
    
    # 添加測試樣本數限制信息
    if args.max_samples:
        metrics['max_samples_limit'] = args.max_samples
        metrics['note'] = f"測試限制為 {args.max_samples} 個樣本"

    # 保存詳細報告
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 如果啟用了 Drive 保存，也保存報告到 Drive
    if args.save_to_drive and drive_mounted:
        drive_report_path = os.path.join(drive_results_folder, args.out)
        with open(drive_report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"📄 評估報告已保存至 Google Drive：{drive_report_path}")

    # 生成圖表
    if args.plot:
        print("生成圖表...")
        drive_folder = drive_results_folder if args.save_to_drive and drive_mounted else None
        
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']), 
            metrics['confusion_matrix_labels'],
            'confusion_matrix.png',
            drive_folder
        )
        plot_roc_curves(
            metrics['roc_data'], 
            metrics['auc_scores'],
            'roc_curves.png',
            drive_folder
        )

    # 打印詳細結果
    print("\n" + "="*60)
    print("詳細評估結果")
    print("="*60)
    
    print(f"\n整體準確率：{metrics['overall_accuracy']:.4f}")
    print(f"Macro F1-Score：{metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score：{metrics['weighted_f1']:.4f}")
    
    if args.max_samples:
        print(f"注意：測試樣本數限制為 {args.max_samples}")
    
    print(f"\n各類別詳細指標：")
    print("-" * 80)
    print(f"{'類別':<12} {'準確率':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'樣本數':<8}")
    print("-" * 80)
    
    for class_name in metrics['confusion_matrix_labels']:
        acc = metrics['per_class_accuracy'][class_name]
        prec = metrics['per_class_precision'][class_name]
        rec = metrics['per_class_recall'][class_name]
        f1 = metrics['per_class_f1'][class_name]
        sup = metrics['per_class_support'][class_name]
        print(f"{class_name:<12} {acc:<8.4f} {prec:<10.4f} {rec:<8.4f} {f1:<10.4f} {sup:<8}")
    
    print("\n計算效率指標：")
    print(f"模型參數總數：{efficiency_metrics['total_parameters']:,}")
    print(f"可訓練參數：{efficiency_metrics['trainable_parameters']:,}")
    print(f"模型大小：{efficiency_metrics['model_size_mb']} MB")
    print(f"平均推理時間：{efficiency_metrics['avg_inference_time_ms']} ± {efficiency_metrics['std_inference_time_ms']} ms")
    
    print(f"\n報告已輸出：{args.out}")
    if args.plot:
        print("圖表已生成：confusion_matrix.png, roc_curves.png")
    
    if args.save_to_drive and drive_mounted:
        print(f"📁 所有結果已保存至 Google Drive 資料夾：{args.drive_folder}")
        print(f"📍 Drive 路徑：{drive_results_folder}")


if __name__ == '__main__':
    main()


