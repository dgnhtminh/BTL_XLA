"""
Script để làm sạch masks trong my_dataset/
Cải thiện chất lượng annotation trước khi training

Chức năng:
1. Remove noise objects (< min_size pixels)
2. Fill holes trong cells
3. Smooth boundaries
4. Backup data cũ trước khi overwrite

Usage:
    python clean_masks.py
    python clean_masks.py --min-size 30  # Custom threshold
    python clean_masks.py --no-backup    # Không backup
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import shutil
from scipy import ndimage
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops
import json
from datetime import datetime

def improve_mask_quality(mask, min_size=20, fill_holes=True, smooth=True):
    """
    Cải thiện chất lượng mask annotation
    
    Args:
        mask: Instance mask (mỗi cell có label khác nhau)
        min_size: Kích thước tối thiểu của object (pixels)
                  20 pixels ≈ cell có đường kính ~5 pixels
        fill_holes: Fill các lỗ trống trong cells
        smooth: Làm mượt boundaries
    
    Returns:
        Cleaned mask
    """
    if mask.max() == 0:
        return mask
    
    # Tạo mask mới
    mask_cleaned = np.zeros_like(mask)
    current_label = 1
    
    # Xử lý từng cell riêng biệt
    for cell_id in range(1, mask.max() + 1):
        # Extract cell
        cell_mask = (mask == cell_id)
        
        # Skip nếu quá nhỏ (có thể là noise)
        if cell_mask.sum() < min_size:
            continue
        
        # Fill holes
        if fill_holes:
            cell_mask = ndimage.binary_fill_holes(cell_mask)
        
        # Smooth boundary
        if smooth:
            # Closing: fill small gaps, smooth edges
            cell_mask = binary_closing(cell_mask, disk(1))
        
        # Add to cleaned mask
        mask_cleaned[cell_mask] = current_label
        current_label += 1
    
    return mask_cleaned

def backup_directory(src_dir, backup_suffix='_backup'):
    """Backup thư mục trước khi modify"""
    src_path = Path(src_dir)
    backup_path = src_path.parent / (src_path.name + backup_suffix)
    
    if backup_path.exists():
        print(f"⚠️  Backup already exists: {backup_path}")
        response = input("    Overwrite existing backup? (y/N): ")
        if response.lower() != 'y':
            print("    Skipping backup...")
            return None
        shutil.rmtree(backup_path)
    
    print(f"📦 Creating backup: {backup_path}")
    shutil.copytree(src_path, backup_path)
    return backup_path

def analyze_mask(mask):
    """Phân tích mask để lấy statistics"""
    props = regionprops(mask)
    
    stats = {
        'n_cells': len(props),
        'cell_sizes': [p.area for p in props],
        'cells_with_holes': sum(1 for p in props if p.euler_number < 1),
        'very_small_cells': sum(1 for p in props if p.area < 20),
    }
    
    if stats['cell_sizes']:
        stats['min_size'] = min(stats['cell_sizes'])
        stats['max_size'] = max(stats['cell_sizes'])
        stats['avg_size'] = np.mean(stats['cell_sizes'])
    else:
        stats['min_size'] = 0
        stats['max_size'] = 0
        stats['avg_size'] = 0
    
    return stats

def process_masks_in_directory(masks_dir, min_size=20, fill_holes=True, smooth=True, dry_run=False):
    """
    Process tất cả masks trong một thư mục
    
    Args:
        masks_dir: Đường dẫn đến thư mục chứa masks
        min_size: Threshold cho noise removal
        fill_holes: Fill holes trong cells
        smooth: Smooth boundaries
        dry_run: Nếu True, chỉ analyze không modify files
    
    Returns:
        Dictionary với statistics
    """
    masks_dir = Path(masks_dir)
    mask_files = sorted(masks_dir.glob('*.png'))
    
    if not mask_files:
        print(f"⚠️  No mask files found in {masks_dir}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Processing: {masks_dir}")
    print(f"Found: {len(mask_files)} mask files")
    print(f"{'='*70}\n")
    
    stats_before = {
        'total_cells': 0,
        'total_noise': 0,
        'total_with_holes': 0,
        'files_processed': 0,
    }
    
    stats_after = stats_before.copy()
    
    # Process each mask
    for mask_file in tqdm(mask_files, desc=f"Processing {masks_dir.name}"):
        # Load mask
        mask = np.array(Image.open(mask_file))
        
        # Analyze before
        before = analyze_mask(mask)
        stats_before['total_cells'] += before['n_cells']
        stats_before['total_noise'] += before['very_small_cells']
        stats_before['total_with_holes'] += before['cells_with_holes']
        stats_before['files_processed'] += 1
        
        # Clean mask
        mask_cleaned = improve_mask_quality(mask, min_size, fill_holes, smooth)
        
        # Analyze after
        after = analyze_mask(mask_cleaned)
        stats_after['total_cells'] += after['n_cells']
        stats_after['total_noise'] += after['very_small_cells']
        stats_after['total_with_holes'] += after['cells_with_holes']
        stats_after['files_processed'] += 1
        
        # Save cleaned mask (if not dry run)
        if not dry_run:
            # Convert to uint16 for compatibility
            mask_cleaned_uint16 = mask_cleaned.astype(np.uint16)
            Image.fromarray(mask_cleaned_uint16).save(mask_file)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY for {masks_dir.name}")
    print(f"{'='*70}")
    print(f"Files processed:     {stats_before['files_processed']}")
    print(f"\nCells:")
    print(f"  Before cleaning:   {stats_before['total_cells']}")
    print(f"  After cleaning:    {stats_after['total_cells']}")
    print(f"  Change:            {stats_after['total_cells'] - stats_before['total_cells']:+d}")
    print(f"\nNoise objects removed:")
    print(f"  Very small (<{min_size}px): {stats_before['total_noise'] - stats_after['total_noise']}")
    print(f"\nCells with holes fixed:")
    print(f"  Before:            {stats_before['total_with_holes']}")
    print(f"  After:             {stats_after['total_with_holes']}")
    print(f"  Fixed:             {stats_before['total_with_holes'] - stats_after['total_with_holes']}")
    print(f"{'='*70}\n")
    
    return {
        'before': stats_before,
        'after': stats_after,
        'directory': str(masks_dir),
    }

def main():
    parser = argparse.ArgumentParser(
        description='Clean masks in my_dataset/ to improve annotation quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clean_masks.py                    # Clean with default settings
  python clean_masks.py --min-size 30      # Use custom threshold
  python clean_masks.py --no-backup        # Skip backup
  python clean_masks.py --dry-run          # Preview changes only
  python clean_masks.py --dataset custom/  # Custom dataset path
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='my_dataset',
        help='Path to dataset directory (default: my_dataset)'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=20,
        help='Minimum cell size in pixels (default: 20)'
    )
    
    parser.add_argument(
        '--no-fill-holes',
        action='store_true',
        help='Skip filling holes in cells'
    )
    
    parser.add_argument(
        '--no-smooth',
        action='store_true',
        help='Skip smoothing boundaries'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup (use with caution!)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("🧹 MASK CLEANING SCRIPT")
    print("="*70)
    print(f"Dataset:     {args.dataset}")
    print(f"Min size:    {args.min_size} pixels")
    print(f"Fill holes:  {not args.no_fill_holes}")
    print(f"Smooth:      {not args.no_smooth}")
    print(f"Backup:      {not args.no_backup}")
    print(f"Dry run:     {args.dry_run}")
    print("="*70 + "\n")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    
    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset directory not found: {dataset_path}")
        return 1
    
    # Check train/val directories
    train_masks = dataset_path / 'train' / 'masks'
    val_masks = dataset_path / 'val' / 'masks'
    
    if not train_masks.exists():
        print(f"❌ Error: Training masks directory not found: {train_masks}")
        return 1
    
    if not val_masks.exists():
        print(f"⚠️  Warning: Validation masks directory not found: {val_masks}")
        val_masks = None
    
    # Confirm before proceeding (if not dry run)
    if not args.dry_run and not args.no_backup:
        print("⚠️  This will modify mask files in place (after backup).")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Create backups
    backup_paths = []
    if not args.no_backup and not args.dry_run:
        print("\n📦 Creating backups...")
        
        train_backup = backup_directory(train_masks.parent)
        if train_backup:
            backup_paths.append(train_backup)
        
        if val_masks:
            val_backup = backup_directory(val_masks.parent)
            if val_backup:
                backup_paths.append(val_backup)
        
        print("✅ Backups created!\n")
    
    # Process masks
    results = {}
    
    # Train masks
    print("\n" + "="*70)
    print("PROCESSING TRAINING MASKS")
    print("="*70)
    results['train'] = process_masks_in_directory(
        train_masks,
        min_size=args.min_size,
        fill_holes=not args.no_fill_holes,
        smooth=not args.no_smooth,
        dry_run=args.dry_run
    )
    
    # Validation masks
    if val_masks:
        print("\n" + "="*70)
        print("PROCESSING VALIDATION MASKS")
        print("="*70)
        results['val'] = process_masks_in_directory(
            val_masks,
            min_size=args.min_size,
            fill_holes=not args.no_fill_holes,
            smooth=not args.no_smooth,
            dry_run=args.dry_run
        )
    
    # Save summary report
    if not args.dry_run:
        report = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'min_size': args.min_size,
                'fill_holes': not args.no_fill_holes,
                'smooth': not args.no_smooth,
            },
            'backups': [str(p) for p in backup_paths],
            'results': results,
        }
        
        report_file = dataset_path / 'mask_cleaning_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ CLEANING COMPLETED!")
    print("="*70)
    
    if args.dry_run:
        print("\n🔍 This was a DRY RUN - no files were modified")
        print("   Run without --dry-run to apply changes")
    else:
        total_removed = 0
        total_before = 0
        total_after = 0
        
        for split, stats in results.items():
            if stats:
                total_before += stats['before']['total_cells']
                total_after += stats['after']['total_cells']
                total_removed += stats['before']['total_noise'] - stats['after']['total_noise']
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total cells before:  {total_before}")
        print(f"   Total cells after:   {total_after}")
        print(f"   Noise removed:       {total_removed}")
        print(f"   Change:              {total_after - total_before:+d}")
        
        if backup_paths:
            print(f"\n📦 Backups created at:")
            for p in backup_paths:
                print(f"   - {p}")
            print(f"\n💡 To restore from backup:")
            print(f"   Remove current folders and rename backups")
        
        print(f"\n🚀 Next step: Train model with cleaned masks!")
    
    print("="*70 + "\n")
    
    return 0

if __name__ == '__main__':
    exit(main())
