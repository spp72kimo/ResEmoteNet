#!/usr/bin/env python3
"""
資料集診斷腳本
用於檢查資料集結構和識別問題
"""

import os
import sys
import argparse
from pathlib import Path

def check_dataset_structure(data_dir: str):
    """檢查資料集結構"""
    print(f"🔍 診斷資料集：{data_dir}")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ 資料目錄不存在：{data_dir}")
        return False
    
    if not os.path.isdir(data_dir):
        print(f"❌ 指定路徑不是資料夾：{data_dir}")
        return False
    
    # 檢查資料目錄內容
    try:
        items = os.listdir(data_dir)
        print(f"📁 資料目錄內容：{items}")
        
        if not items:
            print("❌ 資料目錄為空")
            return False
        
        # 檢查每個項目
        valid_folders = []
        invalid_items = []
        
        for item in items:
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                sub_items = os.listdir(item_path)
                print(f"\n📂 {item}/ ({len(sub_items)} 個項目)")
                
                if len(sub_items) > 0:
                    # 檢查是否包含影像檔案
                    image_files = [f for f in sub_items if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    print(f"  🖼️  影像檔案：{len(image_files)} 張")
                    
                    if len(image_files) > 0:
                        # 顯示前幾個影像檔案作為示例
                        sample_images = image_files[:5]
                        print(f"    範例：{sample_images}")
                        if len(image_files) > 5:
                            print(f"    ... 還有 {len(image_files) - 5} 張影像")
                        
                        valid_folders.append((item, len(image_files)))
                    else:
                        print(f"  ⚠️  警告：資料夾中沒有影像檔案")
                        invalid_items.append(item)
                else:
                    print(f"  ⚠️  警告：資料夾為空")
                    invalid_items.append(item)
            else:
                print(f"📄 {item} (檔案，不是資料夾)")
                invalid_items.append(item)
        
        print(f"\n📊 診斷結果總結")
        print("=" * 60)
        print(f"✅ 有效的情緒類別資料夾：{len(valid_folders)} 個")
        for folder, count in valid_folders:
            print(f"  📂 {folder}: {count} 張影像")
        
        if invalid_items:
            print(f"\n⚠️  有問題的項目：{len(invalid_items)} 個")
            for item in invalid_items:
                print(f"  ❌ {item}")
        
        # 檢查情緒類別映射
        print(f"\n🏷️  情緒類別映射檢查")
        print("-" * 40)
        
        # 標準情緒類別
        standard_emotions = {
            'happy': 0, 'surprise': 1, 'sad': 2, 'anger': 3,
            'disgust': 4, 'fear': 5, 'neutral': 6
        }
        
        # 擴展的情緒類別映射
        extended_emotions = {
            # 標準英文名稱
            'happy': 0, 'surprise': 1, 'sad': 2, 'anger': 3,
            'disgust': 4, 'fear': 5, 'neutral': 6,
            # 常見變體
            'happiness': 0, 'happ': 0, 'joy': 0,
            'surprised': 1, 'surpr': 1,
            'sadness': 2, 'sad_': 2,
            'angry': 3, 'ang': 3,
            'disgusted': 4, 'disg': 4,
            'fearful': 5, 'fear_': 5,
            'neutral_': 6, 'neut': 6,
            # 數字編號
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            # 中文名稱
            '快樂': 0, '開心': 0, '高興': 0,
            '驚訝': 1, '驚奇': 1,
            '悲傷': 2, '傷心': 2,
            '憤怒': 3, '生氣': 3,
            '厭惡': 4, '噁心': 4,
            '恐懼': 5, '害怕': 5,
            '中性': 6, '平靜': 6
        }
        
        mapped_emotions = {}
        unmapped_emotions = []
        
        for folder, _ in valid_folders:
            emotion_name = folder.lower().strip()
            
            # 嘗試映射
            if emotion_name in extended_emotions:
                emotion_idx = extended_emotions[emotion_name]
                emotion_label = list(standard_emotions.keys())[emotion_idx]
                mapped_emotions[folder] = (emotion_idx, emotion_label)
                print(f"  ✅ {folder} -> {emotion_idx} ({emotion_label})")
            else:
                # 嘗試部分匹配
                matched = False
                for key, value in extended_emotions.items():
                    if key in emotion_name or emotion_name in key:
                        emotion_label = list(standard_emotions.keys())[value]
                        mapped_emotions[folder] = (value, emotion_label)
                        print(f"  🔄 {folder} -> {value} ({emotion_label}) [部分匹配: {key}]")
                        matched = True
                        break
                
                if not matched:
                    unmapped_emotions.append(folder)
                    print(f"  ❌ {folder} -> 無法映射")
        
        if unmapped_emotions:
            print(f"\n⚠️  無法映射的情緒類別：{unmapped_emotions}")
            print("建議：檢查資料夾命名是否與預期一致")
        
        print(f"\n📋 建議的資料夾命名：")
        for idx, name in standard_emotions.items():
            print(f"  {idx}: {name}")
        
        return len(valid_folders) > 0 and len(unmapped_emotions) == 0
        
    except Exception as e:
        print(f"❌ 診斷過程中發生錯誤：{e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='資料集結構診斷工具')
    parser.add_argument('--data_dir', type=str, required=True, help='資料集目錄路徑')
    
    args = parser.parse_args()
    
    success = check_dataset_structure(args.data_dir)
    
    if success:
        print(f"\n🎉 資料集診斷完成，結構正常！")
        return 0
    else:
        print(f"\n⚠️  資料集診斷完成，發現問題，請檢查上述信息！")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
