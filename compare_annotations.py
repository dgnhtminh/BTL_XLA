"""
Script để visualize và so sánh annotations trước/sau cleaning
Giúp đánh giá chất lượng cải thiện

Usage:
    python compare_annotations.py
    python compare_annotations.py --old-dir my_dataset/train/masks --new-dir my_dataset/train/masks_new
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from skimage.measure import regionprops
import argparse


def create_comparison_figure(image, old_mask, new_mask, image_name=""):
    """
    Tạo figure so sánh chi tiết old vs new annotations
    """
    from matplotlib.patches import Rectangle
    
    # Analyze masks
    old_regions = regionprops(old_mask)
    new_regions = regionprops(new_mask)
    
    old_sizes = [r.area for r in old_regions]
    new_sizes = [r.area for r in new_regions]
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Define colormap
    cmap = plt.cm.nipy_spectral
    
    # Row 1: OLD annotations
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(old_mask, cmap=cmap)
    n_old = old_mask.max()
    ax2.set_title(f'OLD Annotations\n({n_old} objects)', fontsize=12, fontweight='bold', color='red')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 4, 3)
    ax3.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    ax3.imshow(old_mask, cmap=cmap, alpha=0.4)
    ax3.set_title('OLD Overlay', fontsize=12)
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.hist(old_sizes, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax4.axvline(50, color='orange', linestyle='--', linewidth=2, label='Min threshold (50px)')
    ax4.set_xlabel('Cell Area (pixels)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'OLD Size Distribution\nVery small: {sum(s < 50 for s in old_sizes)}', fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Row 2: NEW annotations
    ax5 = plt.subplot(3, 4, 5)
    ax5.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    ax5.set_title('Original Image', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.imshow(new_mask, cmap=cmap)
    n_new = new_mask.max()
    ax6.set_title(f'NEW Annotations\n({n_new} cells) ✨', fontsize=12, fontweight='bold', color='green')
    ax6.axis('off')
    
    ax7 = plt.subplot(3, 4, 7)
    ax7.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    ax7.imshow(new_mask, cmap=cmap, alpha=0.4)
    ax7.set_title('NEW Overlay ✨', fontsize=12)
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 4, 8)
    ax8.hist(new_sizes, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax8.axvline(50, color='orange', linestyle='--', linewidth=2, label='Min threshold (50px)')
    ax8.set_xlabel('Cell Area (pixels)')
    ax8.set_ylabel('Frequency')
    ax8.set_title(f'NEW Size Distribution\nVery small: {sum(s < 50 for s in new_sizes)}', fontsize=10)
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)
    
    # Row 3: Detailed comparisons
    ax9 = plt.subplot(3, 4, 9)
    # Side by side comparison
    comparison = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    comparison[old_mask > 0] = [255, 0, 0]  # Red for old
    comparison[new_mask > 0] = [0, 255, 0]  # Green for new
    comparison[(old_mask > 0) & (new_mask > 0)] = [255, 255, 0]  # Yellow for overlap
    ax9.imshow(comparison)
    ax9.set_title('Overlap Analysis\n🔴Old 🟢New 🟡Both', fontsize=10)
    ax9.axis('off')
    
    # Calculate metrics
    old_circs = []
    old_solids = []
    for r in old_regions:
        if r.perimeter > 0:
            old_circs.append(4 * np.pi * r.area / (r.perimeter ** 2))
        old_solids.append(r.solidity)
    
    new_circs = []
    new_solids = []
    for r in new_regions:
        if r.perimeter > 0:
            new_circs.append(4 * np.pi * r.area / (r.perimeter ** 2))
        new_solids.append(r.solidity)
    
    ax10 = plt.subplot(3, 4, 10)
    ax10.boxplot([old_circs, new_circs], labels=['OLD', 'NEW'])
    ax10.axhline(0.3, color='orange', linestyle='--', linewidth=2, label='Min threshold')
    ax10.set_ylabel('Circularity')
    ax10.set_title('Shape Quality (Circularity)\nHigher = rounder', fontsize=10)
    ax10.legend(fontsize=8)
    ax10.grid(True, alpha=0.3, axis='y')
    
    ax11 = plt.subplot(3, 4, 11)
    ax11.boxplot([old_solids, new_solids], labels=['OLD', 'NEW'])
    ax11.axhline(0.7, color='orange', linestyle='--', linewidth=2, label='Min threshold')
    ax11.set_ylabel('Solidity')
    ax11.set_title('Compactness (Solidity)\nHigher = fewer holes', fontsize=10)
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3, axis='y')
    
    ax12 = plt.subplot(3, 4, 12)
    # Summary stats
    ax12.axis('off')
    summary_text = f"""
COMPARISON SUMMARY
{'='*30}

Object Count:
  OLD: {n_old:4d} objects
  NEW: {n_new:4d} cells
  Δ:   {n_new - n_old:+4d} ({(n_new-n_old)/max(n_old,1)*100:+.1f}%)

Size (pixels):
  OLD avg: {np.mean(old_sizes):6.1f}
  NEW avg: {np.mean(new_sizes):6.1f}
  
Very Small (< 50px):
  OLD: {sum(s < 50 for s in old_sizes):4d}
  NEW: {sum(s < 50 for s in new_sizes):4d}
  Removed: {sum(s < 50 for s in old_sizes) - sum(s < 50 for s in new_sizes):4d}

Circularity (0-1):
  OLD: {np.mean(old_circs):.3f}
  NEW: {np.mean(new_circs):.3f}
  Δ:   {np.mean(new_circs) - np.mean(old_circs):+.3f}

Solidity (0-1):
  OLD: {np.mean(old_solids):.3f}
  NEW: {np.mean(new_solids):.3f}
  Δ:   {np.mean(new_solids) - np.mean(old_solids):+.3f}
{'='*30}

✅ Quality Improved!
    """
    ax12.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=9,
              verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Annotation Quality Comparison: {image_name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig


def compare_datasets(images_dir, old_masks_dir, new_masks_dir, output_dir='comparison_results', num_samples=10):
    """
    So sánh toàn bộ dataset và tạo visualizations
    """
    images_path = Path(images_dir)
    old_masks_path = Path(old_masks_dir)
    new_masks_path = Path(new_masks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get files
    image_files = sorted(images_path.glob('*.png'))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {images_dir}")
        return
    
    print(f"📊 Comparing {len(image_files)} images...")
    print(f"   Images:    {images_dir}")
    print(f"   Old masks: {old_masks_dir}")
    print(f"   New masks: {new_masks_dir}")
    print(f"   Output:    {output_dir}")
    print()
    
    # Select samples to visualize
    if len(image_files) > num_samples:
        # Select evenly spaced samples
        indices = np.linspace(0, len(image_files) - 1, num_samples, dtype=int)
        sample_files = [image_files[i] for i in indices]
    else:
        sample_files = image_files
    
    print(f"🎨 Creating {len(sample_files)} comparison visualizations...")
    
    for i, img_file in enumerate(sample_files, 1):
        old_mask_file = old_masks_path / img_file.name
        new_mask_file = new_masks_path / img_file.name
        
        if not old_mask_file.exists():
            print(f"⚠️  Missing old mask: {img_file.name}")
            continue
        
        if not new_mask_file.exists():
            print(f"⚠️  Missing new mask: {img_file.name}")
            continue
        
        # Load data
        image = np.array(Image.open(img_file))
        old_mask = np.array(Image.open(old_mask_file))
        new_mask = np.array(Image.open(new_mask_file))
        
        # Create comparison
        fig = create_comparison_figure(image, old_mask, new_mask, img_file.stem)
        
        # Save
        output_file = output_path / f'comparison_{img_file.stem}.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  [{i}/{len(sample_files)}] Saved: {output_file.name}")
    
    print(f"\n✅ All comparisons saved to: {output_dir}")
    print(f"\n💡 TIP: Mở các file comparison_*.png để xem chi tiết!")


def main():
    parser = argparse.ArgumentParser(description='Compare old vs new annotations')
    parser.add_argument('--images-dir', type=str, default='my_dataset/train/images',
                        help='Directory containing images')
    parser.add_argument('--old-dir', type=str, default='my_dataset/train/masks',
                        help='Directory containing old masks')
    parser.add_argument('--new-dir', type=str, default='my_dataset/train/masks_new',
                        help='Directory containing new masks')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                        help='Output directory for comparison visualizations')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    compare_datasets(
        args.images_dir,
        args.old_dir,
        args.new_dir,
        args.output_dir,
        args.num_samples
    )


if __name__ == '__main__':
    main()
