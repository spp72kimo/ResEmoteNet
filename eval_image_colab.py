import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from approach.ResEmoteNet import ResEmoteNet
import matplotlib.pyplot as plt

# 使用 Colab 的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 情緒標籤
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

# 載入模型
model = ResEmoteNet().to(device)
checkpoint = torch.load('best_model.pth', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 圖像預處理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_emotion(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    scores = probabilities.cpu().numpy().flatten()
    return scores

def analyze_image(image_path):
    # 載入圖片
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 人臉檢測
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    
    # 分析每個檢測到的人臉
    for (x, y, w, h) in faces:
        # 繪製邊界框
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 裁剪人臉區域
        face_roi = image_rgb[y:y+h, x:x+w]
        pil_face = Image.fromarray(face_roi)
        
        # 情緒預測
        emotion_scores = detect_emotion(pil_face)
        
        # 顯示結果
        max_emotion_idx = np.argmax(emotion_scores)
        max_emotion = emotions[max_emotion_idx]
        confidence = emotion_scores[max_emotion_idx]
        
        # 在圖片上標註情緒
        cv2.putText(image_rgb, f"{max_emotion}: {confidence:.2f}", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 顯示所有情緒分數
        print(f"\n檢測到人臉 - 位置: ({x}, {y})")
        print("情緒分析結果:")
        for i, emotion in enumerate(emotions):
            print(f"  {emotion}: {emotion_scores[i]:.3f}")
        print(f"主要情緒: {max_emotion} (信心度: {confidence:.3f})")
    
    # 顯示結果圖片
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.title("ResEmoteNet 情緒分析結果")
    plt.axis('off')
    plt.show()
    
    return image_rgb

# 使用範例
# 1. 上傳圖片到 Colab
from google.colab import files
uploaded = files.upload()

# 2. 分析上傳的圖片
for filename in uploaded.keys():
    print(f"\n分析圖片: {filename}")
    result = analyze_image(filename)