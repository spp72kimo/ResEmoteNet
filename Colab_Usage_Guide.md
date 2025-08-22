# ResEmoteNet 在 Google Colab 中的使用指南

## 🚀 快速開始

### 1. 上傳腳本到 Colab
將修改後的 `eval_dataset_colab.py` 腳本上傳到 Colab 環境中。

### 2. 安裝依賴包
```python
!pip install -r requirements.txt
```

### 3. 執行評估腳本

#### 基本用法（只保存到本地）：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --plot
```

#### 保存到 Google Drive：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --plot \
    --save_to_drive \
    --drive_folder "My_Emotion_Results"
```

#### 快速測試（限制樣本數）：
```bash
!python eval_dataset_colab.py \
    --data_dir /path/to/your/dataset \
    --weights /path/to/your/model.pth \
    --max_samples 1000 \
    --plot \
    --save_to_drive \
    --drive_folder "Quick_Test_Results"
```

## 📁 輸出文件結構

### 本地輸出：
```
Colab 工作目錄/
├── eval_report.json          # 詳細評估報告
├── confusion_matrix.png      # 混淆矩陣圖
└── roc_curves.png          # ROC 曲線圖
```

### Google Drive 輸出：
```
Google Drive/MyDrive/
└── ResEmoteNet_Results/     # 或您自定義的資料夾名稱
    ├── eval_report.json     # 詳細評估報告
    ├── confusion_matrix.png # 混淆矩陣圖
    └── roc_curves.png      # ROC 曲線圖
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

## 📊 輸出結果說明

### 控制台輸出：
- 整體準確率
- Macro/Weighted F1-Score
- 每個情緒類別的詳細指標
- 計算效率指標
- 文件保存位置

### JSON 報告包含：
- 所有評估指標
- 混淆矩陣數據
- ROC 曲線數據
- 計算效率指標
- 預測結果和真實標籤

### 圖表文件：
- **confusion_matrix.png**：混淆矩陣熱力圖
- **roc_curves.png**：各類別 ROC 曲線

## ⚠️ 注意事項

1. **Google Drive 掛載**：首次使用時會要求授權訪問您的 Google Drive
2. **存儲空間**：確保 Drive 有足夠的存儲空間
3. **文件權限**：確保有權限在 Drive 中創建資料夾和文件
4. **網路連接**：穩定的網路連接有助於快速上傳文件到 Drive

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

## 📞 支援

如果遇到問題，請檢查：
1. 依賴包是否正確安裝
2. 文件路徑是否正確
3. 數據集格式是否符合要求
4. 模型文件是否完整
