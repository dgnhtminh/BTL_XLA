"""
Script để chọn frames đại diện cho annotation
Chọn 200 frames từ 800 frames gốc theo chiến lược thông minh
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import random

def calculate_image_stats(img_path):
    """Tính toán các thống kê của ảnh"""
    img = np.array(Image.open(img_path))
    
    return {
        'path': img_path,
        'mean': img.mean(),
        'std': img.std(),
        'brightness': img.mean(),
        'contrast': img.std()
    }

def select_diverse_frames(frames_dir, n_select=200, output_file='selected_frames.txt'):
    """
    Chọn frames đa dạng từ dataset
    
    Chiến lược:
    1. Chia dataset thành các bins theo độ sáng
    2. Chọn đều từ mỗi bin
    3. Đảm bảo coverage từ đầu đến cuối video
    """
    
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob('frame_*.png'))
    
    print(f"📁 Tìm thấy {len(frame_files)} frames")
    print(f"🎯 Cần chọn {n_select} frames cho annotation")
    
    if len(frame_files) <= n_select:
        print("⚠️ Số frames ít hơn số cần chọn, sẽ chọn tất cả!")
        selected = frame_files
    else:
        # Chiến lược 1: Chọn đều theo thời gian (30% = 60 frames)
        n_temporal = int(n_select * 0.3)
        step = len(frame_files) // n_temporal
        temporal_selected = [frame_files[i] for i in range(0, len(frame_files), step)][:n_temporal]
        
        print(f"\n1️⃣ Chọn {len(temporal_selected)} frames đều theo thời gian...")
        
        # Lấy frames còn lại
        remaining_frames = [f for f in frame_files if f not in temporal_selected]
        
        # Chiến lược 2: Tính stats cho frames còn lại và chọn đa dạng (70% = 140 frames)
        print(f"2️⃣ Phân tích {len(remaining_frames)} frames còn lại...")
        stats = [calculate_image_stats(f) for f in tqdm(remaining_frames[:500])]  # Giới hạn để nhanh hơn
        
        # Chia thành bins theo brightness
        n_bins = 10
        brightnesses = [s['brightness'] for s in stats]
        bins = np.linspace(min(brightnesses), max(brightnesses), n_bins + 1)
        
        # Chọn đều từ mỗi bin
        n_per_bin = (n_select - len(temporal_selected)) // n_bins
        diverse_selected = []
        
        for i in range(n_bins):
            bin_stats = [s for s in stats if bins[i] <= s['brightness'] < bins[i+1]]
            if len(bin_stats) > 0:
                # Chọn ngẫu nhiên từ bin này
                selected_from_bin = random.sample(bin_stats, min(n_per_bin, len(bin_stats)))
                diverse_selected.extend([s['path'] for s in selected_from_bin])
        
        print(f"3️⃣ Chọn {len(diverse_selected)} frames đa dạng theo độ sáng...")
        
        # Kết hợp
        selected = temporal_selected + diverse_selected
        
        # Nếu chưa đủ, chọn thêm random
        if len(selected) < n_select:
            remaining = [f for f in frame_files if f not in selected]
            extra = random.sample(remaining, min(n_select - len(selected), len(remaining)))
            selected.extend(extra)
        
        selected = selected[:n_select]
    
    # Sắp xếp theo tên file
    selected = sorted(selected)
    
    # Lưu danh sách
    with open(output_file, 'w') as f:
        for frame in selected:
            f.write(f"{frame.name}\n")
    
    print(f"\n✅ Đã chọn {len(selected)} frames!")
    print(f"📝 Danh sách lưu tại: {output_file}")
    
    # Chia thành train/val (75%/25%)
    n_train = int(len(selected) * 0.75)
    train_frames = selected[:n_train]
    val_frames = selected[n_train:]
    
    print(f"\n📊 Phân chia:")
    print(f"   Training: {len(train_frames)} frames")
    print(f"   Validation: {len(val_frames)} frames")
    
    # Lưu danh sách train/val
    with open('selected_frames_train.txt', 'w') as f:
        for frame in train_frames:
            f.write(f"{frame.name}\n")
    
    with open('selected_frames_val.txt', 'w') as f:
        for frame in val_frames:
            f.write(f"{frame.name}\n")
    
    print(f"\n📁 Danh sách chi tiết:")
    print(f"   - selected_frames_train.txt ({len(train_frames)} frames)")
    print(f"   - selected_frames_val.txt ({len(val_frames)} frames)")
    
    return selected, train_frames, val_frames

def copy_selected_frames(selected_frames, frames_dir, output_dir):
    """Copy frames đã chọn sang thư mục mới"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\n📋 Copying {len(selected_frames)} frames to {output_dir}...")
    
    for frame in tqdm(selected_frames):
        shutil.copy(frame, output_dir / frame.name)
    
    print(f"✅ Hoàn tất!")

if __name__ == '__main__':
    # Cấu hình
    FRAMES_DIR = 'frames'
    N_SELECT = 200  # Tổng số frames cần annotation
    
    print("="*60)
    print("🎯 SCRIPT CHỌN FRAMES ĐỂ ANNOTATION")
    print("="*60)
    
    # Chọn frames
    selected, train_frames, val_frames = select_diverse_frames(
        FRAMES_DIR, 
        n_select=N_SELECT,
        output_file='selected_frames.txt'
    )
    
    # Tùy chọn: Copy sang thư mục riêng để dễ annotation
    print("\n" + "="*60)
    print("💡 HƯỚNG DẪN TIẾP THEO:")
    print("="*60)
    print("\n1. Mở file 'selected_frames_train.txt' và 'selected_frames_val.txt'")
    print("2. Dùng ImageJ/Fiji hoặc tool annotation để label các frames này")
    print("3. Lưu masks vào:")
    print("   - my_dataset/train/images/ và my_dataset/train/masks/")
    print("   - my_dataset/val/images/ và my_dataset/val/masks/")
    print("\n🚀 Sau khi annotation xong, chạy lại notebook training!")
