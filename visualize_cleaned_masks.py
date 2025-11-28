"""
Script để visualize kết quả cleaning masks
So sánh trước/sau cleaning

Usage:
    python visualize_cleaned_masks.py
    python visualize_cleaned_masks.py --n-samples 5
    python visualize_cleaned_masks.py --save-figures
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import argparse
from stardist import random_label_cmap

def load_and_compare(image_path, mask_original_path, mask_cleaned_path):
    """Load và so sánh ảnh, mask gốc, mask cleaned"""
    
    # Load image
    img = np.array(Image.open(image_path))
    
    # Load masks
    mask_orig = np.array(Image.open(mask_original_path))
    mask_clean = np.array(Image.open(mask_cleaned_path))
    
    # Get statistics
    props_orig = regionprops(mask_orig)
    props_clean = regionprops(mask_clean)
    
    stats = {
        'n_cells_before': len(props_orig),
        'n_cells_after': len(props_clean),
        'sizes_before': [p.area for p in props_orig],
        'sizes_after': [p.area for p in props_clean],
        'noise_removed': len(props_orig) - len(props_clean),
    }
    
    return img, mask_orig, mask_clean, stats

def visualize_comparison(img, mask_orig, mask_clean, stats, title_prefix="", save_path=None):
    """Visualize comparison between original and cleaned masks"""
    
    lbl_cmap = random_label_cmap()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Before cleaning
    axes[0, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_orig, cmap=lbl_cmap)
    axes[0, 1].set_title(f'BEFORE Cleaning\n({stats["n_cells_before"]} cells)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[0, 2].imshow(mask_orig, cmap=lbl_cmap, alpha=0.5)
    axes[0, 2].set_title('Overlay BEFORE')
    axes[0, 2].axis('off')
    
    if stats['sizes_before']:
        axes[0, 3].hist(stats['sizes_before'], bins=30, edgecolor='black', color='red', alpha=0.7)
        axes[0, 3].axvline(20, color='orange', linestyle='--', linewidth=2, label='Min threshold')
        axes[0, 3].set_title('Size Distribution BEFORE')
        axes[0, 3].set_xlabel('Cell Size (pixels)')
        axes[0, 3].set_ylabel('Count')
        axes[0, 3].legend()
    
    # Row 2: After cleaning
    axes[1, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')
    
    change = stats['n_cells_after'] - stats['n_cells_before']
    axes[1, 1].imshow(mask_clean, cmap=lbl_cmap)
    axes[1, 1].set_title(f'AFTER Cleaning ✨\n({stats["n_cells_after"]} cells, {change:+d})')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(img, cmap='gray' if img.ndim == 2 else None)
    axes[1, 2].imshow(mask_clean, cmap=lbl_cmap, alpha=0.5)
    axes[1, 2].set_title('Overlay AFTER')
    axes[1, 2].axis('off')
    
    if stats['sizes_after']:
        axes[1, 3].hist(stats['sizes_after'], bins=30, edgecolor='black', color='green', alpha=0.7)
        axes[1, 3].axvline(20, color='orange', linestyle='--', linewidth=2, label='Min threshold')
        axes[1, 3].set_title('Size Distribution AFTER')
        axes[1, 3].set_xlabel('Cell Size (pixels)')
        axes[1, 3].set_ylabel('Count')
        axes[1, 3].legend()
    
    plt.suptitle(f'{title_prefix}Mask Cleaning Comparison\n'
                 f'Removed: {stats["noise_removed"]} noise objects',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize mask cleaning results')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='my_dataset',
        help='Path to dataset directory (default: my_dataset)'
    )
    
    parser.add_argument(
        '--backup-suffix',
        type=str,
        default='_backup',
        help='Backup directory suffix (default: _backup)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=3,
        help='Number of samples to visualize (default: 3)'
    )
    
    parser.add_argument(
        '--save-figures',
        action='store_true',
        help='Save figures to file'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'both'],
        help='Which split to visualize (default: train)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("👁️  MASK CLEANING VISUALIZATION")
    print("="*70)
    print(f"Dataset:      {args.dataset}")
    print(f"Backup suffix: {args.backup_suffix}")
    print(f"Samples:      {args.n_samples}")
    print(f"Split:        {args.split}")
    print("="*70 + "\n")
    
    dataset_path = Path(args.dataset)
    
    # Determine which splits to process
    splits = []
    if args.split in ['train', 'both']:
        splits.append('train')
    if args.split in ['val', 'both']:
        splits.append('val')
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Visualizing {split.upper()} split")
        print(f"{'='*70}\n")
        
        # Paths
        images_dir = dataset_path / split / 'images'
        masks_cleaned_dir = dataset_path / split / 'masks'
        masks_backup_dir = dataset_path / (split + args.backup_suffix) / 'masks'
        
        # Check paths exist
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            continue
        
        if not masks_cleaned_dir.exists():
            print(f"❌ Cleaned masks directory not found: {masks_cleaned_dir}")
            continue
        
        if not masks_backup_dir.exists():
            print(f"⚠️  Backup masks directory not found: {masks_backup_dir}")
            print(f"   Cannot compare - backup not found or cleaning not performed yet.")
            continue
        
        # Get file lists
        image_files = sorted(images_dir.glob('*'))
        
        if not image_files:
            print(f"⚠️  No images found in {images_dir}")
            continue
        
        # Sample random images
        import random
        n_samples = min(args.n_samples, len(image_files))
        sampled_files = random.sample(image_files, n_samples)
        
        print(f"Visualizing {n_samples} random samples from {len(image_files)} images...\n")
        
        # Process each sample
        for i, img_file in enumerate(sampled_files, 1):
            print(f"[{i}/{n_samples}] Processing {img_file.name}...")
            
            # Corresponding mask files
            mask_cleaned_file = masks_cleaned_dir / img_file.name
            mask_backup_file = masks_backup_dir / img_file.name
            
            if not mask_cleaned_file.exists():
                print(f"   ⚠️  Cleaned mask not found: {mask_cleaned_file.name}")
                continue
            
            if not mask_backup_file.exists():
                print(f"   ⚠️  Backup mask not found: {mask_backup_file.name}")
                continue
            
            # Load and compare
            img, mask_orig, mask_clean, stats = load_and_compare(
                img_file, mask_backup_file, mask_cleaned_file
            )
            
            # Visualize
            save_path = None
            if args.save_figures:
                save_path = dataset_path / f'comparison_{split}_{img_file.stem}.png'
            
            visualize_comparison(
                img, mask_orig, mask_clean, stats,
                title_prefix=f"{split.upper()} - {img_file.name}\n",
                save_path=save_path
            )
            
            # Print stats
            print(f"   Before: {stats['n_cells_before']} cells")
            print(f"   After:  {stats['n_cells_after']} cells")
            print(f"   Change: {stats['n_cells_after'] - stats['n_cells_before']:+d}")
            print(f"   Noise removed: {stats['noise_removed']}")
            print()
    
    print("="*70)
    print("✅ Visualization completed!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
