import os
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
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
import cv2
from PIL import Image

from approach.ResEmoteNet import ResEmoteNet


def get_timestamp():
    """獲取當前時間戳記，格式：YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_filename_with_timestamp(base_name: str, extension: str = None) -> str:
    """生成帶有時間戳記的檔案名稱"""
    timestamp = get_timestamp()
    if extension:
        if not extension.startswith('.'):
            extension = '.' + extension
        return f"{base_name}_{timestamp}{extension}"
    else:
        name_parts = base_name.rsplit('.', 1)
        if len(name_parts) == 2:
            base, ext = name_parts
            return f"{base}_{timestamp}.{ext}"
        else:
            return f"{base_name}_{timestamp}"


def mount_google_drive():
    """掛載 Google Drive 到 Colab"""
    try:
        # 首先檢查是否已經掛載
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive 已經掛載在 /content/drive")
            return True
        
        # 檢查是否在 Colab 環境中
        try:
            from google.colab import drive
            print("🔄 正在掛載 Google Drive...")
            drive.mount('/content/drive')
            print("✅ Google Drive 已成功掛載到 /content/drive")
            return True
        except Exception as e:
            print(f"⚠️  Drive 掛載失敗：{str(e)}")
            return False
            
    except ImportError:
        print("⚠️  不在 Colab 環境中，跳過 Drive 掛載")
        return False
    except Exception as e:
        print(f"⚠️  掛載過程中發生錯誤：{str(e)}")
        return False


def check_drive_mounted() -> bool:
    """檢查 Google Drive 是否已經掛載"""
    return os.path.exists('/content/drive/MyDrive')


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


class FaceDetector:
    """人臉偵測器類別，支援多種偵測方法"""
    
    def __init__(self, method: str = 'haar', min_face_size: int = 40):
        self.method = method
        self.min_face_size = min_face_size
        self.face_cascade = None
        self.retinaface_model = None
        
        self._initialize_detector()
    
    def _initialize_detector(self):
        """初始化選定的偵測器"""
        if self.method == 'haar':
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("✅ Haar Cascade 人臉偵測器初始化成功")
            except Exception as e:
                print(f"⚠️  Haar Cascade 初始化失敗：{e}")
                if os.path.exists('haarcascade_frontalface_default.xml'):
                    self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    print("✅ 使用專案內 Haar Cascade 檔案")
        
        elif self.method == 'retinaface':
            try:
                from retinaface import RetinaFace
                self.retinaface_model = RetinaFace
                print("✅ RetinaFace 人臉偵測器初始化成功")
            except ImportError:
                print("⚠️  RetinaFace 未安裝，請執行：pip install retina-face")
                print("🔄 自動切換到 Haar Cascade 方法")
                self.method = 'haar'
                self._initialize_detector()
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """偵測影像中的人臉"""
        if self.method == 'haar':
            return self._detect_haar(image)
        elif self.method == 'retinaface':
            return self._detect_retinaface(image)
        else:
            return []
    
    def _detect_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用 Haar Cascade 偵測人臉"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(self.min_face_size, self.min_face_size)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def _detect_retinaface(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """使用 RetinaFace 偵測人臉"""
        if self.retinaface_model is None:
            return []
        
        try:
            faces = self.retinaface_model.detect_faces(image)
            face_boxes = []
            for face_key in faces:
                face = faces[face_key]
                x1, y1, x2, y2 = face['facial_area']
                w, h = x2 - x1, y2 - y1
                if w >= self.min_face_size and h >= self.min_face_size:
                    face_boxes.append((x1, y1, w, h))
            return face_boxes
        except Exception as e:
            print(f"⚠️  RetinaFace 偵測失敗：{e}")
            return []
    
    def crop_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                  padding: int = 0) -> np.ndarray:
        """裁切人臉區域"""
        x, y, w, h = face_box
        
        # 添加填充
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        return image[y1:y2, x1:x2]


class FaceDetectionDataset:
    """支援人臉偵測的資料集類別"""
    
    def __init__(self, data_dir: str, face_detector: FaceDetector, 
                 transform=None, padding: int = 0, min_face_size: int = 40):
        self.data_dir = data_dir
        self.face_detector = face_detector
        self.transform = transform
        self.padding = padding
        self.min_face_size = min_face_size
        
        # 掃描資料集
        self.samples = self._scan_dataset()
        print(f"📊 資料集掃描完成：找到 {len(self.samples)} 個有效樣本")
    
    def _scan_dataset(self) -> List[Tuple[str, int, Optional[Tuple[int, int, int, int]]]]:
        """掃描資料集，找出所有影像和人臉位置"""
        samples = []
        
        # 遍歷所有情緒類別資料夾
        for emotion_dir in os.listdir(self.data_dir):
            emotion_path = os.path.join(self.data_dir, emotion_dir)
            if not os.path.isdir(emotion_path):
                continue
            
            # 獲取情緒類別索引
            try:
                emotion_idx = self._get_emotion_index(emotion_dir)
            except ValueError:
                continue
            
            # 掃描該情緒類別下的所有影像
            for img_name in os.listdir(emotion_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(emotion_path, img_name)
                
                # 偵測人臉
                face_box = self._detect_face_in_image(img_path)
                
                if face_box is not None:
                    samples.append((img_path, emotion_idx, face_box))
                else:
                    print(f"⚠️  未在 {img_path} 中偵測到人臉")
        
        return samples
    
    def _get_emotion_index(self, emotion_dir: str) -> int:
        """獲取情緒類別索引"""
        emotion_mapping = {
            'happy': 0, 'surprise': 1, 'sad': 2, 'anger': 3,
            'disgust': 4, 'fear': 5, 'neutral': 6
        }
        
        emotion_name = emotion_dir.lower().strip()
        if emotion_name in emotion_mapping:
            return emotion_mapping[emotion_name]
        else:
            raise ValueError(f"未知的情緒類別：{emotion_dir}")
    
    def _detect_face_in_image(self, img_path: str) -> Optional[Tuple[int, int, int, int]]:
        """在單張影像中偵測人臉"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                return None
            
            faces = self.face_detector.detect_faces(image)
            
            if len(faces) == 0:
                return None
            
            # 如果有多個人臉，選擇最大的
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                print(f"📸 {img_path} 中偵測到 {len(faces)} 個人臉，選擇最大的")
            
            return faces[0]
            
        except Exception as e:
            print(f"⚠️  處理 {img_path} 時發生錯誤：{e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """獲取資料樣本"""
        img_path, emotion_idx, face_box = self.samples[idx]
        
        # 讀取影像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 裁切人臉
        face_image = self.face_detector.crop_face(image, face_box, self.padding)
        
        # 轉換為 PIL 影像
        pil_image = Image.fromarray(face_image)
        
        # 應用轉換
        if self.transform:
            pil_image = self.transform(pil_image)
        
        return pil_image, emotion_idx


def create_dataloader(data_dir: str, batch_size: int, num_workers: int, 
                     face_detection: bool = False, face_method: str = 'haar',
                     face_padding: int = 0, min_face_size: int = 40,
                     max_samples: int = None) -> Tuple[DataLoader, Dict[int, str]]:
    """創建資料載入器，支援人臉偵測"""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"資料夾不存在：{data_dir}")

    found = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not found:
        raise ValueError(f"指定資料夾內沒有情緒子資料夾：{data_dir}")

    # 簡化的類別映射
    idx_to_class = {0: 'happy', 1: 'surprise', 2: 'sad', 3: 'anger', 
                    4: 'disgust', 5: 'fear', 6: 'neutral'}

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if face_detection:
        print(f"🔍 啟用人臉偵測，使用 {face_method} 方法")
        face_detector = FaceDetector(method=face_method, min_face_size=min_face_size)
        dataset = FaceDetectionDataset(
            data_dir=data_dir,
            face_detector=face_detector,
            transform=transform,
            padding=face_padding,
            min_face_size=min_face_size
        )
    else:
        print("📸 使用標準 ImageFolder 資料集（無人臉偵測）")
        dataset = datasets.ImageFolder(
            root=data_dir,
            transform=transform,
        )

    # 如果指定了最大樣本數，則創建子集
    if max_samples and max_samples < len(dataset):
        print(f"原始數據集大小：{len(dataset)}，限制為：{max_samples} 個樣本")
        
        if face_detection:
            # 對於人臉偵測資料集，直接截斷
            dataset.samples = dataset.samples[:max_samples]
            print(f"🎯 人臉偵測資料集截斷完成！實際樣本數：{len(dataset)}")
        else:
            # 對於標準資料集，進行簡單截斷
            dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
            print(f"🎯 標準資料集截斷完成！實際樣本數：{len(dataset)}")

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


def convert_numpy_types(obj):
    """將 numpy 類型轉換為 Python 原生類型，以便 JSON 序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def calculate_metrics(y_true: List[int], y_pred: List[int], idx_to_class: Dict[int, str]) -> Dict:
    """計算詳細的評估指標"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(idx_to_class.keys())
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(idx_to_class.keys()))
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
    
    result = {
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
    
    # 轉換 numpy 類型為 Python 原生類型
    return convert_numpy_types(result)


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
            auc_scores[class_name] = float(auc_score)  # 確保是 Python float
    
    # 計算 micro-average ROC 和 AUC
    y_true_bin_ravel = y_true_bin.ravel()
    y_pred_probs_ravel = y_pred_probs.ravel()
    
    if len(np.unique(y_true_bin_ravel)) > 1:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin_ravel, y_pred_probs_ravel)
        auc_micro = auc(fpr_micro, tpr_micro)
        roc_data['micro'] = {'fpr': fpr_micro.tolist(), 'tpr': tpr_micro.tolist()}
        auc_scores['micro'] = float(auc_micro)
    
    return {'roc_data': roc_data, 'auc_scores': auc_scores}


def calculate_efficiency_metrics(model: torch.nn.Module, test_loader: DataLoader, 
                                device: torch.device, num_test_samples: int = 100) -> Dict:
    """計算模型效率指標"""
    # 計算模型參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小（假設 float32）
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # 計算推理時間
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            if i >= num_test_samples:
                break
            
            inputs = inputs.to(device)
            
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            
            inference_times.append((end_time - start_time) * 1000)  # 轉換為毫秒
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(model_size_mb, 2),
        'avg_inference_time_ms': round(avg_inference_time, 2),
        'std_inference_time_ms': round(std_inference_time, 2)
    }


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], filename: str, 
                         drive_folder: str = None):
    """繪製混淆矩陣"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 保存到本地
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 混淆矩陣已保存：{filename}")
    
    # 如果指定了 Drive 資料夾，也保存到 Drive
    if drive_folder:
        drive_filename = os.path.join(drive_folder, filename)
        plt.savefig(drive_filename, dpi=300, bbox_inches='tight')
        print(f"📊 混淆矩陣已保存到 Drive：{drive_filename}")
    
    plt.close()


def plot_roc_curves(roc_data: Dict, auc_scores: Dict, filename: str, 
                    drive_folder: str = None):
    """繪製 ROC 曲線"""
    plt.figure(figsize=(12, 8))
    
    for class_name, roc_info in roc_data.items():
        if class_name != 'micro':
            fpr = roc_info['fpr']
            tpr = roc_info['tpr']
            auc_score = auc_scores.get(class_name, 0)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
    
    # 繪製 micro-average 曲線
    if 'micro' in roc_data:
        fpr_micro = roc_data['micro']['fpr']
        tpr_micro = roc_data['micro']['tpr']
        auc_micro = auc_scores.get('micro', 0)
        plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {auc_micro:.3f})', 
                color='black', linestyle='--', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存到本地
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 ROC 曲線已保存：{filename}")
    
    # 如果指定了 Drive 資料夾，也保存到 Drive
    if drive_folder:
        drive_filename = os.path.join(drive_folder, filename)
        plt.savefig(drive_filename, dpi=300, bbox_inches='tight')
        print(f"📈 ROC 曲線已保存到 Drive：{drive_filename}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='ResEmoteNet 資料集評估工具（支援人臉偵測）')
    
    # 基本參數
    parser.add_argument('--data_dir', type=str, required=True, help='測試資料目錄路徑')
    parser.add_argument('--weights', type=str, required=True, help='模型權重檔案路徑')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='資料載入工作程序數')
    parser.add_argument('--max_samples', type=int, default=None, help='最大測試樣本數限制')
    
    # 人臉偵測參數
    parser.add_argument('--face_detection', action='store_true', help='啟用人臉偵測和裁切')
    parser.add_argument('--face_method', type=str, default='haar', 
                       choices=['haar', 'retinaface'], help='人臉偵測方法')
    parser.add_argument('--face_padding', type=int, default=0, help='人臉邊界填充像素')
    parser.add_argument('--min_face_size', type=int, default=40, help='最小人臉尺寸')
    
    # 輸出選項
    parser.add_argument('--plot', action='store_true', help='生成圖表')
    parser.add_argument('--save_to_drive', action='store_true', help='保存結果到 Google Drive')
    parser.add_argument('--drive_folder', type=str, default='ResEmoteNet_Results', 
                       help='Google Drive 中的結果資料夾名稱')
    
    args = parser.parse_args()
    
    # 檢查 Google Drive 掛載
    drive_mounted = False
    if args.save_to_drive:
        drive_mounted = mount_google_drive()
        if drive_mounted:
            drive_results_folder = create_drive_folder(args.drive_folder)
        else:
            print("⚠️  Google Drive 掛載失敗，結果將只保存到本地")
    
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 使用設備：{device}")
    
    # 載入模型
    print("🔄 正在載入模型...")
    model = load_model(args.weights, device)
    print("✅ 模型載入完成")
    
    # 創建資料載入器
    print("🔄 正在創建資料載入器...")
    test_loader, idx_to_class = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        face_detection=args.face_detection,
        face_method=args.face_method,
        face_padding=args.face_padding,
        min_face_size=args.min_face_size,
        max_samples=args.max_samples
    )
    
    print(f"📊 測試資料集：{len(test_loader.dataset)} 個樣本")
    print(f"🏷️  類別映射：{idx_to_class}")
    
    # 開始評估
    print("\n🚀 開始評估...")
    t0 = time.time()
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="🔍 評估中")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 計算指標
    print("📊 正在計算評估指標...")
    metrics = calculate_metrics(all_labels, all_predictions, idx_to_class)
    roc_metrics = calculate_roc_auc(all_labels, np.array(all_probabilities), idx_to_class)
    
    # 計算效率指標
    print("⚡ 正在計算效率指標...")
    efficiency_metrics = calculate_efficiency_metrics(model, test_loader, device)
    
    # 合併所有指標
    metrics.update(roc_metrics)
    metrics['idx_to_class'] = idx_to_class
    metrics['elapsed_sec'] = round(time.time() - t0, 2)
    metrics['computational_efficiency'] = efficiency_metrics
    
    # 添加人臉偵測相關信息
    if args.face_detection:
        metrics['face_detection'] = {
            'enabled': True,
            'method': args.face_method,
            'padding': args.face_padding,
            'min_face_size': args.min_face_size
        }
    else:
        metrics['face_detection'] = {'enabled': False}
    
    # 添加測試樣本數限制信息
    if args.max_samples:
        metrics['max_samples_limit'] = args.max_samples
        metrics['note'] = f"測試限制為 {args.max_samples} 個樣本"
    
    # 添加時間戳記信息
    metrics['timestamp'] = get_timestamp()
    metrics['evaluation_time'] = datetime.now().isoformat()
    
    # 轉換所有 numpy 類型為 Python 原生類型，以便 JSON 序列化
    print("🔄 正在準備 JSON 序列化...")
    metrics = convert_numpy_types(metrics)
    
    # 生成檔案名稱
    base_name = "ResEmoteNet_Eval"
    if args.face_detection:
        base_name += f"_FaceDet_{args.face_method}"
    
    report_filename = generate_filename_with_timestamp(f"{base_name}_Report", ".json")
    confusion_matrix_filename = generate_filename_with_timestamp(f"{base_name}_CM", ".png")
    roc_curves_filename = generate_filename_with_timestamp(f"{base_name}_ROC", ".png")
    
    # 保存詳細報告
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 如果啟用了 Drive 保存，也保存報告到 Drive
    if args.save_to_drive and drive_mounted:
        drive_report_path = os.path.join(drive_results_folder, report_filename)
        with open(drive_report_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"📄 評估報告已保存至 Google Drive：{drive_report_path}")
    
    # 生成圖表
    if args.plot:
        print("🎨 正在生成圖表...")
        drive_folder = drive_results_folder if args.save_to_drive and drive_mounted else None
        
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']), 
            metrics['confusion_matrix_labels'],
            confusion_matrix_filename,
            drive_folder
        )
        plot_roc_curves(
            metrics['roc_data'], 
            metrics['auc_scores'],
            roc_curves_filename,
            drive_folder
        )
    
    # 打印詳細結果
    print("\n" + "="*60)
    print("詳細評估結果")
    print("="*60)
    
    print(f"\n整體準確率：{metrics['overall_accuracy']:.4f}")
    print(f"Macro F1-Score：{metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score：{metrics['weighted_f1']:.4f}")
    
    # 顯示人臉偵測信息
    if args.face_detection:
        print(f"\n🔍 人臉偵測設置：")
        print(f"  方法：{args.face_method}")
        print(f"  邊界填充：{args.face_padding} 像素")
        print(f"  最小人臉尺寸：{args.min_face_size} 像素")
    
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
    
    print(f"\n報告已輸出：{report_filename}")
    if args.plot:
        print(f"圖表已生成：{confusion_matrix_filename}, {roc_curves_filename}")
    
    if args.save_to_drive and drive_mounted:
        print(f"📁 所有結果已保存至 Google Drive 資料夾：{args.drive_folder}")
        print(f"📍 Drive 路徑：{drive_results_folder}")
    
    # 顯示檔案列表
    print(f"\n📋 本次評估生成的文件：")
    print(f"  📄 JSON 報告：{report_filename}")
    if args.plot:
        print(f"  📊 混淆矩陣：{confusion_matrix_filename}")
        print(f"  📈 ROC 曲線：{roc_curves_filename}")
    
    print(f"\n評估完成！耗時：{metrics['elapsed_sec']} 秒")


if __name__ == '__main__':
    main()
