# ResEmoteNet 資料集評估工具升級版 - 人臉偵測功能

## 概述

這是 `eval_dataset_colab.py` 的升級版本，新增了人臉偵測和人臉裁切功能。升級版可以自動偵測影像中的人臉，裁切出人臉區域，然後進行情緒辨識，提高了評估的準確性和實用性。

## 主要新功能

### 🔍 人臉偵測
- **Haar Cascade**: 使用 OpenCV 的 Haar Cascade 分類器，快速且輕量
- **RetinaFace**: 使用 RetinaFace 模型，提供更精確的人臉偵測
- 自動選擇最大人臉（如果影像中有多個人臉）

### ✂️ 人臉裁切
- 自動裁切偵測到的人臉區域
- 支援邊界填充（padding）設定
- 保持影像比例和品質

### 📊 智能資料處理
- 自動過濾未偵測到人臉的影像
- 支援最大樣本數限制
- 保持原有的評估指標計算

## 安裝需求

### 基本套件
```bash
pip install torch torchvision opencv-python pillow numpy matplotlib seaborn scikit-learn tqdm
```

### 可選套件（用於 RetinaFace）
```bash
pip install retina-face
```

### 專案檔案
確保以下檔案存在於專案目錄中：
- `haarcascade_frontalface_default.xml` - Haar Cascade 模型檔案
- `approach/ResEmoteNet.py` - ResEmoteNet 模型定義

## 使用方法

### 基本用法（無人臉偵測）
```bash
python eval_dataset_colab_with_face_detection.py \
    --data_dir /path/to/test/data \
    --weights /path/to/model.pth
```

### 啟用人臉偵測（使用 Haar Cascade）
```bash
python eval_dataset_colab_with_face_detection.py \
    --data_dir /path/to/test/data \
    --weights /path/to/model.pth \
    --face_detection \
    --face_method haar \
    --face_padding 10 \
    --min_face_size 40
```

### 啟用人臉偵測（使用 RetinaFace）
```bash
python eval_dataset_colab_with_face_detection.py \
    --data_dir /path/to/test/data \
    --weights /path/to/model.pth \
    --face_detection \
    --face_method retinaface \
    --face_padding 20 \
    --min_face_size 50
```

### 限制測試樣本數
```bash
python eval_dataset_colab_with_face_detection.py \
    --data_dir /path/to/test/data \
    --weights /path/to/model.pth \
    --face_detection \
    --max_samples 1000
```

### 啟用圖表生成和 Google Drive 保存
```bash
python eval_dataset_colab_with_face_detection.py \
    --data_dir /path/to/test/data \
    --weights /path/to/model.pth \
    --face_detection \
    --face_method retinaface \
    --plot \
    --save_to_drive \
    --drive_folder "My_Results"
```

## 參數說明

### 基本參數
- `--data_dir`: 測試資料目錄路徑（必需）
- `--weights`: 模型權重檔案路徑（必需）
- `--batch_size`: 批次大小（預設：16）
- `--num_workers`: 資料載入工作程序數（預設：4）
- `--max_samples`: 最大測試樣本數限制（可選）

### 人臉偵測參數
- `--face_detection`: 啟用人臉偵測和裁切（可選）
- `--face_method`: 人臉偵測方法（可選：'haar', 'retinaface'，預設：'haar'）
- `--face_padding`: 人臉邊界填充像素（預設：0）
- `--min_face_size`: 最小人臉尺寸（預設：40）

### 輸出選項
- `--plot`: 生成圖表（混淆矩陣和 ROC 曲線）
- `--save_to_drive`: 保存結果到 Google Drive
- `--drive_folder`: Google Drive 中的結果資料夾名稱（預設：'ResEmoteNet_Results'）

## 資料目錄結構

測試資料應該按照以下結構組織：

```
test_data/
├── happy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── sad/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── anger/
│   └── ...
└── ... (其他情緒類別)
```

## 工作流程

### 1. 資料掃描
- 掃描指定目錄下的所有情緒類別資料夾
- 檢查每個影像檔案的有效性

### 2. 人臉偵測（如果啟用）
- 對每個影像進行人臉偵測
- 如果偵測到多個人臉，選擇最大的
- 記錄未偵測到人臉的影像

### 3. 人臉裁切
- 根據偵測到的人臉邊界框裁切影像
- 應用邊界填充（如果設定）
- 轉換為模型所需的格式

### 4. 模型評估
- 載入訓練好的 ResEmoteNet 模型
- 對裁切後的人臉影像進行情緒辨識
- 計算各種評估指標

### 5. 結果輸出
- 顯示整體準確率和各類別指標
- 顯示人臉偵測相關信息
- 輸出評估耗時
- 生成 JSON 格式的詳細報告
- 可選生成混淆矩陣和 ROC 曲線圖表
- 支援保存到 Google Drive

## 輸出範例

```
🚀 使用設備：cuda
🔄 正在載入模型...
✅ 模型載入完成
🔄 正在創建資料載入器...
🔍 啟用人臉偵測，使用 haar 方法
📊 資料集掃描完成：找到 1500 個有效樣本
📊 測試資料集：1500 個樣本
🏷️  類別映射：{0: 'happy', 1: 'surprise', 2: 'sad', 3: 'anger', 4: 'disgust', 5: 'fear', 6: 'neutral'}

🚀 開始評估...
🔍 評估中: 100%|██████████| 94/94 [00:45<00:00,  2.08it/s]
📊 正在計算評估指標...

============================================================
詳細評估結果
============================================================

整體準確率：0.8234
Macro F1-Score：0.8156

🔍 人臉偵測設置：
  方法：haar
  邊界填充：10 像素
  最小人臉尺寸：40 像素

各類別詳細指標：
--------------------------------------------------------------------------------
類別           準確率    Precision  Recall   F1-Score   樣本數  
--------------------------------------------------------------------------------
happy         0.8500    0.8600     0.8500   0.8550     200     
sad           0.7800    0.7900     0.7800   0.7850     200     
anger         0.8100    0.8200     0.8100   0.8150     200     
disgust       0.7900    0.8000     0.7900   0.7950     200     
fear          0.7500    0.7600     0.7500   0.7550     200     
surprise      0.8200    0.8300     0.8200   0.8250     200     
neutral       0.8500    0.8600     0.8500   0.8550     200     

評估完成！耗時：45.23 秒
```

## 注意事項

### 1. 人臉偵測限制
- Haar Cascade 方法較快但準確度較低
- RetinaFace 方法較準確但需要額外安裝
- 如果無法偵測到人臉，該影像會被跳過

### 2. 效能考量
- 人臉偵測會增加處理時間
- 建議根據硬體配置調整 `num_workers` 參數
- 大量影像處理時建議使用 `max_samples` 限制

### 3. 記憶體使用
- 人臉偵測會增加記憶體使用量
- 如果遇到記憶體不足，可以減少 `batch_size`

### 4. 錯誤處理
- 程式會自動跳過無法處理的影像
- 會顯示警告信息但不會中斷執行
- 建議檢查警告信息以了解資料品質

## 故障排除

### 1. Haar Cascade 檔案找不到
```
⚠️  Haar Cascade 初始化失敗
✅ 使用專案內 Haar Cascade 檔案
```
解決方案：確保 `haarcascade_frontalface_default.xml` 檔案在專案目錄中

### 2. RetinaFace 未安裝
```
⚠️  RetinaFace 未安裝，請執行：pip install retina-face
🔄 自動切換到 Haar Cascade 方法
```
解決方案：安裝 RetinaFace 或使用 Haar Cascade 方法

### 3. 未偵測到人臉
```
⚠️  未在 image.jpg 中偵測到人臉
```
解決方案：
- 檢查影像是否包含清晰的人臉
- 調整 `min_face_size` 參數
- 嘗試不同的 `face_method`

## 與原版比較

| 功能 | 原版 | 升級版 |
|------|------|--------|
| 人臉偵測 | ❌ | ✅ |
| 人臉裁切 | ❌ | ✅ |
| 多種偵測方法 | ❌ | ✅ |
| 自動過濾無效影像 | ❌ | ✅ |
| 邊界填充控制 | ❌ | ✅ |
| 最小人臉尺寸設定 | ❌ | ✅ |
| 評估指標 | 完整 | ✅ |
| 圖表生成 | ✅ | ✅ |
| Google Drive 支援 | ✅ | ✅ |
| ROC 曲線分析 | ✅ | ✅ |
| 效率指標計算 | ✅ | ✅ |

## 未來改進方向

1. **更多偵測方法**: 支援 Dlib、MTCNN 等
2. **人臉對齊**: 自動對齊人臉角度
3. **品質評估**: 評估人臉影像品質
4. **批次處理**: 支援多個資料集批次處理
5. **進階視覺化**: 支援更多圖表類型和互動功能
6. **雲端整合**: 支援更多雲端儲存服務（如 OneDrive、Dropbox）
7. **即時監控**: 支援即時進度追蹤和結果預覽
8. **批次報告**: 支援多個模型或資料集的批次評估報告

## 貢獻

如果您有改進建議或發現問題，歡迎提出 Issue 或 Pull Request。

## 授權

本專案遵循與原 ResEmoteNet 專案相同的授權條款。
