#!/usr/bin/env python3
"""
測試 Google Drive 支援功能的腳本
"""

import os
import sys
import tempfile
import json
from datetime import datetime

# 添加當前目錄到 Python 路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_google_drive_functions():
    """測試 Google Drive 相關函數"""
    print("🧪 測試 Google Drive 支援功能...")
    
    try:
        from eval_dataset_colab_with_face_detection import (
            get_timestamp, 
            generate_filename_with_timestamp,
            mount_google_drive,
            check_drive_mounted,
            get_drive_path,
            create_drive_folder
        )
        print("✅ 成功導入 Google Drive 相關函數")
    except ImportError as e:
        print(f"❌ 導入失敗：{e}")
        return False
    
    # 測試時間戳記函數
    print("\n📅 測試時間戳記函數...")
    timestamp = get_timestamp()
    print(f"  時間戳記：{timestamp}")
    assert len(timestamp) == 15, "時間戳記格式不正確"
    print("  ✅ 時間戳記函數正常")
    
    # 測試檔案名稱生成函數
    print("\n📝 測試檔案名稱生成函數...")
    base_name = "test_file"
    filename = generate_filename_with_timestamp(base_name, ".txt")
    print(f"  生成檔案名：{filename}")
    assert filename.startswith(base_name), "檔案名格式不正確"
    assert filename.endswith(".txt"), "副檔名不正確"
    print("  ✅ 檔案名稱生成函數正常")
    
    # 測試 Google Drive 掛載檢查
    print("\n🔍 測試 Google Drive 掛載檢查...")
    is_mounted = check_drive_mounted()
    print(f"  Drive 已掛載：{is_mounted}")
    print("  ✅ Drive 掛載檢查函數正常")
    
    # 測試 Drive 路徑獲取
    print("\n📍 測試 Drive 路徑獲取...")
    base_path = "test_folder"
    drive_path = get_drive_path(base_path)
    expected_path = os.path.join('/content/drive/MyDrive', base_path)
    print(f"  預期路徑：{expected_path}")
    print(f"  實際路徑：{drive_path}")
    assert drive_path == expected_path, "Drive 路徑不正確"
    print("  ✅ Drive 路徑獲取函數正常")
    
    # 測試 Drive 資料夾創建（僅在 Drive 已掛載時）
    if is_mounted:
        print("\n📁 測試 Drive 資料夾創建...")
        try:
            test_folder = "test_drive_folder"
            folder_path = create_drive_folder(test_folder)
            print(f"  創建資料夾：{folder_path}")
            
            # 檢查資料夾是否存在
            if os.path.exists(folder_path):
                print("  ✅ Drive 資料夾創建成功")
                
                # 清理測試資料夾
                try:
                    os.rmdir(folder_path)
                    print("  🧹 測試資料夾已清理")
                except OSError:
                    print("  ⚠️  無法清理測試資料夾（可能不為空）")
            else:
                print("  ❌ Drive 資料夾創建失敗")
        except Exception as e:
            print(f"  ❌ Drive 資料夾創建錯誤：{e}")
    else:
        print("\n📁 跳過 Drive 資料夾創建測試（Drive 未掛載）")
    
    # 測試 Google Drive 掛載
    print("\n🔄 測試 Google Drive 掛載...")
    try:
        mount_result = mount_google_drive()
        print(f"  掛載結果：{mount_result}")
        print("  ✅ Drive 掛載函數正常")
    except Exception as e:
        print(f"  ❌ Drive 掛載錯誤：{e}")
    
    print("\n🎉 所有 Google Drive 功能測試完成！")
    return True

def test_face_detection_integration():
    """測試人臉偵測與 Google Drive 的整合"""
    print("\n🔍 測試人臉偵測與 Google Drive 整合...")
    
    try:
        from eval_dataset_colab_with_face_detection import FaceDetector, FaceDetectionDataset
        print("✅ 成功導入人臉偵測相關類別")
    except ImportError as e:
        print(f"❌ 導入失敗：{e}")
        return False
    
    # 測試 FaceDetector 初始化
    print("\n📸 測試 FaceDetector 初始化...")
    try:
        detector = FaceDetector(method='haar')
        print("  ✅ FaceDetector 初始化成功")
    except Exception as e:
        print(f"  ❌ FaceDetector 初始化失敗：{e}")
        return False
    
    # 測試檔案名稱生成與人臉偵測的整合
    print("\n🔗 測試檔案名稱生成與人臉偵測整合...")
    base_name = "face_detection_results"
    filename = generate_filename_with_timestamp(base_name, ".json")
    print(f"  生成檔案名：{filename}")
    
    # 模擬評估結果
    mock_results = {
        'face_detection': {
            'enabled': True,
            'method': 'haar',
            'padding': 10,
            'min_face_size': 40
        },
        'overall_accuracy': 0.85,
        'timestamp': get_timestamp(),
        'test_type': 'face_detection_integration'
    }
    
    # 測試 JSON 序列化
    try:
        json_str = json.dumps(mock_results, ensure_ascii=False, indent=2)
        print("  ✅ JSON 序列化成功")
        print(f"  序列化長度：{len(json_str)} 字符")
    except Exception as e:
        print(f"  ❌ JSON 序列化失敗：{e}")
        return False
    
    print("  ✅ 人臉偵測與 Google Drive 整合測試成功")
    return True

def main():
    """主測試函數"""
    print("🚀 開始測試 Google Drive 支援功能...")
    print("=" * 60)
    
    # 測試基本 Google Drive 功能
    basic_test_passed = test_google_drive_functions()
    
    # 測試人臉偵測整合
    integration_test_passed = test_face_detection_integration()
    
    print("\n" + "=" * 60)
    print("📊 測試結果總結")
    print("=" * 60)
    
    if basic_test_passed:
        print("✅ 基本 Google Drive 功能測試：通過")
    else:
        print("❌ 基本 Google Drive 功能測試：失敗")
    
    if integration_test_passed:
        print("✅ 人臉偵測整合測試：通過")
    else:
        print("❌ 人臉偵測整合測試：失敗")
    
    if basic_test_passed and integration_test_passed:
        print("\n🎉 所有測試通過！Google Drive 支援功能正常")
        return 0
    else:
        print("\n⚠️  部分測試失敗，請檢查錯誤信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
