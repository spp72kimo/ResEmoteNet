import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from approach.ResEmoteNet import ResEmoteNet
import argparse
import matplotlib.pyplot as plt

# 使用 Colab 的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# 情緒標籤
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']

model = None

def load_model(weights_path: str):
    global model
    model = ResEmoteNet().to(device)
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model

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
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
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
    
    if len(faces) == 0:
        print("未偵測到人臉。將輸出原始圖片。")
    return image_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='輸入圖片路徑')
    parser.add_argument('--weights', required=True, help='模型權重 .pth 路徑')
    parser.add_argument('--out', default='result.png', help='輸出結果圖片路徑')
    args = parser.parse_args()

    load_model(args.weights)
    result_img = analyze_image(args.image)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.title("ResEmoteNet 情緒分析結果")
    plt.axis('off')
    plt.savefig(args.out, bbox_inches='tight')
    print(f"結果已儲存: {args.out}")