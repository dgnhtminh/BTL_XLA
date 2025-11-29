"""
Script sử dụng model hiện tại (AP=0.72) để pre-label frames
Giúp tăng tốc annotation: chỉ cần sửa thay vì vẽ từ đầu!

Workflow:
1. Model predict trên selected frames
2. Lưu predictions dưới dạng masks
3. Mở trong ImageJ/Fiji để sửa
4. Tiết kiệm: ~50-60% thời gian annotation!
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from csbdeep.utils import normalize
from stardist.models import StarDist2D

def load_selected_frames(list_file, frames_dir):
    """Load danh sách frames từ file txt"""
    frames_dir = Path(frames_dir)
    
    with open(list_file, 'r') as f:
        frame_names = [line.strip() for line in f if line.strip()]
    
    frame_paths = [frames_dir / name for name in frame_names]
    
    # Kiểm tra files tồn tại
    existing = [p for p in frame_paths if p.exists()]
    
    print(f"📁 Found {len(existing)}/{len(frame_paths)} frames")
    
    if len(existing) < len(frame_paths):
        missing = set(frame_paths) - set(existing)
        print(f"⚠️ Missing {len(missing)} frames:")
        for m in list(missing)[:5]:
            print(f"   - {m.name}")
        if len(missing) > 5:
            print(f"   ... and {len(missing)-5} more")
    
    return existing

def pre_label_frames(frame_paths, model, output_dir, split='train'):
    """
    Dùng model để pre-label frames
    
    Args:
        frame_paths: List các đường dẫn frames
        model: StarDist model đã train
        output_dir: Thư mục output (my_dataset)
        split: 'train' hoặc 'val'
    """
    output_dir = Path(output_dir)
    
    # Tạo thư mục
    images_dir = output_dir / split / 'images'
    masks_dir = output_dir / split / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PRE-LABELING: {len(frame_paths)} frames for {split.upper()}")
    print(f"{'='*60}\n")
    
    print(f"📁 Output directories:")
    print(f"   Images: {images_dir}")
    print(f"   Masks:  {masks_dir}")
    
    # Process frames
    for frame_path in tqdm(frame_paths, desc=f"Pre-labeling {split}"):
        # Load image
        img = np.array(Image.open(frame_path))
        
        # Normalize
        img_norm = normalize(img, 1, 99.8, axis=(0, 1))
        
        # Predict
        labels, details = model.predict_instances(
            img_norm,
            n_tiles=model._guess_n_tiles(img_norm),
            show_tile_progress=False
        )
        
        # Save image
        Image.fromarray(img).save(images_dir / frame_path.name)
        
        # Save mask (as 16-bit for ImageJ compatibility)
        labels_16bit = labels.astype(np.uint16)
        Image.fromarray(labels_16bit).save(masks_dir / frame_path.name)
    
    print(f"\n✅ Pre-labeling completed!")
    print(f"   Images: {len(list(images_dir.glob('*')))} files")
    print(f"   Masks:  {len(list(masks_dir.glob('*')))} files")

def main():
    print("="*60)
    print("🤖 PRE-LABEL SCRIPT")
    print("="*60)
    print("\nSử dụng model hiện tại để tạo draft annotations")
    print("Bạn sẽ chỉ cần SỬA thay vì VẼ TỪ ĐẦU!")
    print("\nTimesave: ~50-60% thời gian annotation")
    print("="*60 + "\n")
    
    # Configuration
    FRAMES_DIR = 'frames'
    MODEL_NAME = 'stardist_my_data'
    MODEL_BASEDIR = 'models'
    OUTPUT_DIR = 'my_dataset'
    
    TRAIN_LIST = 'selected_frames_train.txt'
    VAL_LIST = 'selected_frames_val.txt'
    
    # Check files exist
    if not Path(TRAIN_LIST).exists():
        print(f"❌ Error: {TRAIN_LIST} not found!")
        print(f"💡 Run 'select_frames_for_annotation.py' first!")
        return
    
    if not Path(VAL_LIST).exists():
        print(f"❌ Error: {VAL_LIST} not found!")
        print(f"💡 Run 'select_frames_for_annotation.py' first!")
        return
    
    # Load model
    print(f"📦 Loading model: {MODEL_NAME}...")
    try:
        model = StarDist2D(None, name=MODEL_NAME, basedir=MODEL_BASEDIR)
        print(f"✅ Model loaded!")
        print(f"   Config: n_rays={model.config.n_rays}, grid={model.config.grid}")
        print(f"   Thresholds: nms={model.thresholds.nms:.3f}, prob={model.thresholds.prob:.3f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"💡 Make sure you have trained the model first!")
        return
    
    # Load frame lists
    print(f"\n{'='*60}")
    print(f"LOADING FRAME LISTS")
    print(f"{'='*60}\n")
    
    train_frames = load_selected_frames(TRAIN_LIST, FRAMES_DIR)
    val_frames = load_selected_frames(VAL_LIST, FRAMES_DIR)
    
    print(f"\n✅ Total: {len(train_frames)} train + {len(val_frames)} val = {len(train_frames)+len(val_frames)} frames")
    
    # Confirm
    print(f"\n{'='*60}")
    print(f"⚠️ IMPORTANT: This will create/overwrite files in:")
    print(f"   {OUTPUT_DIR}/train/")
    print(f"   {OUTPUT_DIR}/val/")
    print(f"{'='*60}\n")
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Pre-label train set
    pre_label_frames(train_frames, model, OUTPUT_DIR, split='train')
    
    # Pre-label val set
    pre_label_frames(val_frames, model, OUTPUT_DIR, split='val')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ PRE-LABELING COMPLETED!")
    print(f"{'='*60}\n")
    
    print(f"📊 Summary:")
    print(f"   Training set: {len(train_frames)} frames")
    print(f"   Validation set: {len(val_frames)} frames")
    print(f"   Total: {len(train_frames)+len(val_frames)} frames")
    
    print(f"\n📁 Output location:")
    print(f"   {OUTPUT_DIR}/train/images/  (images)")
    print(f"   {OUTPUT_DIR}/train/masks/   (masks to edit)")
    print(f"   {OUTPUT_DIR}/val/images/    (images)")
    print(f"   {OUTPUT_DIR}/val/masks/     (masks to edit)")
    
    print(f"\n{'='*60}")
    print(f"💡 NEXT STEPS:")
    print(f"{'='*60}\n")
    
    print(f"1. Open ImageJ/Fiji")
    print(f"2. Load images from: {OUTPUT_DIR}/train/images/")
    print(f"3. Load corresponding masks from: {OUTPUT_DIR}/train/masks/")
    print(f"4. EDIT the masks (add/remove/fix cells)")
    print(f"5. Save edited masks back to: {OUTPUT_DIR}/train/masks/")
    print(f"6. Repeat for validation set")
    print(f"7. Run training notebook: 1_training_my_data_IMPROVED.ipynb")
    
    print(f"\n⏱️ Estimated time saved:")
    print(f"   Without pre-labeling: ~2-3 min/frame × {len(train_frames)+len(val_frames)} = {(len(train_frames)+len(val_frames))*2.5/60:.1f} hours")
    print(f"   With pre-labeling: ~1-1.5 min/frame × {len(train_frames)+len(val_frames)} = {(len(train_frames)+len(val_frames))*1.25/60:.1f} hours")
    print(f"   Time saved: ~{(len(train_frames)+len(val_frames))*1.25/60:.1f} hours! 🎉")
    
    print(f"\n{'='*60}")
    print(f"💡 PRO TIP:")
    print(f"{'='*60}\n")
    print(f"Model AP=0.72 means ~72% of predictions are good!")
    print(f"Focus on:")
    print(f"   - Cells the model MISSED (false negatives)")
    print(f"   - Cells the model WRONGLY detected (false positives)")
    print(f"   - Cells not properly SEPARATED (merge errors)")
    print(f"\nDon't re-draw good predictions! Just fix the errors.")
    
    print(f"\n🚀 Ready to go! Good luck with annotation!")

if __name__ == '__main__':
    main()
