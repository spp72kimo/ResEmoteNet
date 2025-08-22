# ResEmoteNet 在 Google Colab 中的使用指南

## 🚀 快速開始

### 1. 上傳腳本到 Colab
將修改後的 `eval_dataset_colab.py` 腳本上傳到 Colab 環境中。

### 2. 安裝依賴包

#### 方法一：使用 Colab 專用 requirements（推薦）
```python
!pip install -r requirements_colab.txt
```

#### 方法二：手動安裝核心套件
```python
# 安裝 PyTorch（Colab 預設已安裝，但版本可能較舊）
!pip install torch>=2.2.0 torchvision>=0.17.0

# 安裝其他必要套件
!pip install numpy>=1.26.0 scikit-learn>=1.4.0
!pip install opencv-python>=4.9.0 Pillow>=10.0.0
!pip install matplotlib>=3.8.0 seaborn>=0.13.0
!pip install pandas>=2.0.0 tqdm>=4.65.0
```

#### 方法三：檢查並升級現有套件
```python
# 檢查 Python 版本
import sys
print(f"Python 版本：{sys.version}")

# 檢查 PyTorch 版本
import torch
print(f"PyTorch 版本：{torch.__version__}")

# 如果需要升級 PyTorch
!pip install --upgrade torch torchvision
```

### 3. 執行評估腳本

#### 🎯 推薦方法：在 Colab 中執行（避免 Drive 掛載衝突）

```python
# 1) 掛載 Drive（如果資料或權重在 Drive）
from google.colab import drive
drive.mount('/content/drive')

# 2) 設定路徑
DATA_DIR = '/content/drive/MyDrive/datasets/AffectNet/Test'     # 指向您的資料夾
WEIGHTS  = '/content/drive/MyDrive/Models/ResEmoteNet/affectnet7_model.pth'  # 權重

# 3) 執行評估（腳本會自動檢測已掛載的 Drive）
!python eval_dataset_colab.py \
    --data_dir "$DATA_DIR" \
    --weights "$WEIGHTS" \
    --max_samples 1000 \
    --plot \
    --save_to_drive \
    --drive_folder "Quick_Test_Results"
```

#### 替代方法：直接在 Colab 中執行 Python 代碼

```python
# 1) 掛載 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2) 設定路徑
DATA_DIR = '/content/drive/MyDrive/datasets/AffectNet/Test'
WEIGHTS  = '/content/drive/MyDrive/Models/ResEmoteNet/affectnet7_model.pth'

# 3) 執行評估（使用 Python 而不是 shell 命令）
import subprocess
import sys

cmd = [
    sys.executable, 'eval_dataset_colab.py',
    '--data_dir', DATA_DIR,
    '--weights', WEIGHTS,
    '--max_samples', '1000',
    '--plot',
    '--save_to_drive',
    '--drive_folder', 'Quick_Test_Results'
]

result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

#### 基本用法（只保存到本地，帶時間戳記）：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --plot
```

#### 保存到 Google Drive（帶時間戳記）：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --plot \
    --save_to_drive \
    --drive_folder "My_Emotion_Results"
```

#### 快速測試（限制樣本數，帶時間戳記）：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --max_samples 1000 \
    --plot \
    --save_to_drive \
    --drive_folder "Quick_Test_Results"
```

#### 不使用時間戳記（覆蓋舊檔案）：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --plot \
    --no_timestamp
```

## 📁 輸出文件結構

### 本地輸出（帶時間戳記）：
```
Colab 工作目錄/
├── eval_report_20241201_143052.json    # 詳細評估報告
├── confusion_matrix_20241201_143052.png # 混淆矩陣圖
└── roc_curves_20241201_143052.png      # ROC 曲線圖
```

### Google Drive 輸出（帶時間戳記）：
```
Google Drive/MyDrive/
└── ResEmoteNet_Results/                 # 或您自定義的資料夾名稱
    ├── eval_report_20241201_143052.json # 詳細評估報告
    ├── confusion_matrix_20241201_143052.png # 混淆矩陣圖
    └── roc_curves_20241201_143052.png  # ROC 曲線圖
```

## 🔧 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--data_dir` | 數據集根目錄路徑 | 必需 |
| `--weights` | 模型權重文件路徑 | 必需 |
| `--batch_size` | 批次大小 | 64 |
| `--num_workers` | 數據載入器工作進程數 | 2 |
| `--max_samples` | 限制測試的最大樣本數 | 無限制 |
| `--out` | 輸出 JSON 報告文件名 | eval_report.json |
| `--plot` | 是否生成圖表 | False |
| `--drive_folder` | Google Drive 中的資料夾名稱 | ResEmoteNet_Results |
| `--save_to_drive` | 是否保存結果到 Google Drive | False |
| `--no_timestamp` | 不在檔案名稱後加上時間戳記 | False |

## 🕐 時間戳記功能

### 自動時間戳記
腳本會自動在每個輸出文件名後加上時間戳記，格式為：`YYYYMMDD_HHMMSS`

**範例**：
- `eval_report.json` → `eval_report_20241201_143052.json`
- `confusion_matrix.png` → `confusion_matrix_20241201_143052.png`
- `roc_curves.png` → `roc_curves_20241201_143052.png`

### 優勢
1. **避免覆蓋**：每次運行都會生成新的檔案
2. **時間追蹤**：清楚知道每個結果的生成時間
3. **版本管理**：可以比較不同時間的評估結果
4. **組織整理**：按時間順序組織實驗結果

### 禁用時間戳記
如果不需要時間戳記，可以使用 `--no_timestamp` 參數：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --plot \
    --no_timestamp
```

## 💡 使用技巧

### 1. 數據集路徑
- 如果數據集在 Colab 本地：`/content/dataset`
- 如果數據集在 Drive：`/content/drive/MyDrive/dataset`
- 如果數據集在網上：先下載到 Colab

### 2. 模型權重路徑
- 如果模型在 Colab 本地：`/content/model.pth`
- 如果模型在 Drive：`/content/drive/MyDrive/models/model.pth`
- 如果模型在網上：先下載到 Colab

### 3. 快速測試建議
對於大數據集，建議先用 `--max_samples 1000` 進行快速測試，確認腳本運行正常後再進行完整評估。

### 4. 檔案管理建議
- 使用時間戳記來區分不同實驗的結果
- 定期清理舊的評估結果檔案
- 在 Drive 中創建有組織的資料夾結構

## 📊 輸出結果說明

### 控制台輸出：
- 時間戳記信息
- 檔案名稱預覽
- 整體準確率
- Macro/Weighted F1-Score
- 每個情緒類別的詳細指標
- 計算效率指標
- 文件保存位置
- 本次生成的文件列表

### JSON 報告包含：
- 時間戳記信息
- 評估時間
- 所有評估指標
- 混淆矩陣數據
- ROC 曲線數據
- 計算效率指標
- 預測結果和真實標籤

### 圖表文件：
- **confusion_matrix_YYYYMMDD_HHMMSS.png**：混淆矩陣熱力圖
- **roc_curves_YYYYMMDD_HHMMSS.png**：各類別 ROC 曲線

## ⚠️ 注意事項

1. **Google Drive 掛載**：首次使用時會要求授權訪問您的 Google Drive
2. **存儲空間**：確保 Drive 有足夠的存儲空間
3. **文件權限**：確保有權限在 Drive 中創建資料夾和文件
4. **網路連接**：穩定的網路連接有助於快速上傳文件到 Drive
5. **檔案管理**：時間戳記會增加檔案名稱長度，注意檔案系統的長度限制

## 🔍 故障排除

### 問題：無法掛載 Google Drive
**解決方案**：
- 檢查網路連接
- 重新授權 Google 帳戶
- 重啟 Colab 運行時

### 問題：權限錯誤
**解決方案**：
- 確保已授權 Colab 訪問 Drive
- 檢查 Drive 存儲空間是否充足

### 問題：圖片無法保存
**解決方案**：
- 檢查 matplotlib 後端設置
- 確保有寫入權限
- 檢查磁盤空間

### 問題：檔案名稱過長
**解決方案**：
- 使用 `--no_timestamp` 參數
- 縮短基礎檔案名稱
- 檢查檔案系統的長度限制

### 問題：套件安裝失敗
**解決方案**：
- 使用 `requirements_colab.txt` 而不是 `requirements.txt`
- 檢查 Python 版本兼容性
- 手動安裝核心套件
- 重啟 Colab 運行時

### 問題：PyTorch 版本不兼容
**解決方案**：
```python
# 檢查當前版本
import torch
print(f"PyTorch 版本：{torch.__version__}")

# 升級到兼容版本
!pip install --upgrade torch torchvision

# 或者安裝特定版本
!pip install torch==2.2.0 torchvision==0.17.0
```

### 問題：Google Drive 掛載衝突
**解決方案**：
```python
# 方法一：重新啟動 Colab 運行時
# Runtime -> Restart runtime

# 方法二：檢查 Drive 是否已經掛載
import os
if os.path.exists('/content/drive/MyDrive'):
    print("Drive 已經掛載")
else:
    print("Drive 未掛載")

# 方法三：使用 Python 代碼執行而不是 shell 命令
import subprocess
import sys

cmd = [
    sys.executable, 'eval_dataset_colab.py',
    '--data_dir', '/content/drive/MyDrive/datasets/AffectNet/Test',
    '--weights', '/content/drive/MyDrive/Models/ResEmoteNet/affectnet7_model.pth',
    '--max_samples', '1000',
    '--plot',
    '--save_to_drive',
    '--drive_folder', 'Quick_Test_Results'
]

result = subprocess.run(cmd, capture_output=True, text=True)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

### 問題：Drive 掛載後腳本仍嘗試掛載
**解決方案**：
腳本已經更新，會自動檢測已掛載的 Drive。如果仍有問題，請：
1. 確保使用最新版本的腳本
2. 檢查腳本中的 `mount_google_drive()` 函數是否正確
3. 使用 `--no_timestamp` 參數測試

## 📞 支援

如果遇到問題，請檢查：
1. 依賴包是否正確安裝
2. 文件路徑是否正確
3. 數據集格式是否符合要求
4. 模型文件是否完整
5. 時間戳記是否正確生成
6. Python 版本是否兼容

## 🆘 緊急故障排除

如果所有方法都失敗，請嘗試：

```python
# 1. 重啟 Colab 運行時
# Runtime -> Restart runtime

# 2. 重新安裝所有套件
!pip install --upgrade pip
!pip install torch torchvision numpy scikit-learn opencv-python Pillow matplotlib seaborn pandas tqdm

# 3. 檢查環境
import sys
print(f"Python: {sys.version}")
import torch
print(f"PyTorch: {torch.__version__}")
import numpy as np
print(f"NumPy: {np.__version__}")
```
