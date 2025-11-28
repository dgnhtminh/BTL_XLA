"""
Script demo để test annotation system trên 1 image
Giúp bạn hiểu workflow trước khi chạy toàn bộ

Usage:
    python demo_single_image.py
    python demo_single_image.py --image my_dataset/train/images/frame_0049.png
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Import advanced annotator
try:
    from advanced_annotation import AdvancedAnnotator
except ImportError:
    print("❌ Cannot import advanced_annotation.py")
    print("   Make sure advanced_annotation.py is in the same directory")
    sys.exit(1)


def demo_single_image(image_path, show_steps=True):
    """
    Demo annotation process trên 1 ảnh
    
    Args:
        image_path: Path to image file
        show_steps: Show step-by-step visualization
    """
    print("="*70)
    print("DEMO: ADVANCED ANNOTATION SYSTEM")
    print("="*70)
    print()
    
    # Load image
    print(f"📂 Loading image: {image_path}")
    image_file = Path(image_path)
    
    if not image_file.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    image = np.array(Image.open(image_file))
    print(f"   Shape: {image.shape}")
    print(f"   Dtype: {image.dtype}")
    print()
    
    # Create annotator
    print("🔧 Creating annotator with settings:")
    annotator = AdvancedAnnotator(
        min_cell_size=50,
        max_cell_size=5000,
        min_circularity=0.3,
        gaussian_sigma=1.5,
        distance_threshold=0.4,
    )
    print(f"   min_cell_size: {annotator.min_cell_size} pixels")
    print(f"   max_cell_size: {annotator.max_cell_size} pixels")
    print(f"   min_circularity: {annotator.min_circularity}")
    print(f"   gaussian_sigma: {annotator.gaussian_sigma}")
    print()
    
    # Annotate with visualization
    print("🎯 Running annotation...")
    mask, fig = annotator.annotate_image(image, visualize=True)
    
    n_cells = mask.max()
    print(f"✅ Annotation completed!")
    print(f"   Detected {n_cells} cells")
    print()
    
    # Save results
    output_dir = Path('demo_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save mask
    mask_file = output_dir / f'{image_file.stem}_mask.png'
    Image.fromarray(mask.astype(np.uint16)).save(mask_file)
    print(f"💾 Saved mask to: {mask_file}")
    
    # Save visualization
    viz_file = output_dir / f'{image_file.stem}_pipeline.png'
    fig.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved pipeline visualization to: {viz_file}")
    
    # Create detailed overlay
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='nipy_spectral')
    axes[1].set_title(f'Detected Cells ({n_cells} cells)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[2].imshow(mask, cmap='nipy_spectral', alpha=0.4)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    overlay_file = output_dir / f'{image_file.stem}_overlay.png'
    plt.savefig(overlay_file, dpi=150, bbox_inches='tight')
    print(f"💾 Saved overlay to: {overlay_file}")
    
    # Show statistics
    from skimage.measure import regionprops
    regions = regionprops(mask)
    
    sizes = [r.area for r in regions]
    circularities = []
    solidities = []
    
    for r in regions:
        if r.perimeter > 0:
            circ = 4 * np.pi * r.area / (r.perimeter ** 2)
            circularities.append(circ)
        solidities.append(r.solidity)
    
    print()
    print("="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Number of cells:        {len(regions)}")
    print(f"\nCell Size (pixels):")
    print(f"  Mean:                 {np.mean(sizes):.1f}")
    print(f"  Std:                  {np.std(sizes):.1f}")
    print(f"  Min:                  {np.min(sizes):.1f}")
    print(f"  Max:                  {np.max(sizes):.1f}")
    print(f"\nCircularity (0-1, higher=rounder):")
    print(f"  Mean:                 {np.mean(circularities):.3f}")
    print(f"  Std:                  {np.std(circularities):.3f}")
    print(f"  Min:                  {np.min(circularities):.3f}")
    print(f"  Max:                  {np.max(circularities):.3f}")
    print(f"\nSolidity (0-1, higher=more compact):")
    print(f"  Mean:                 {np.mean(solidities):.3f}")
    print(f"  Std:                  {np.std(solidities):.3f}")
    print(f"  Min:                  {np.min(solidities):.3f}")
    print(f"  Max:                  {np.max(solidities):.3f}")
    print("="*70)
    print()
    
    # Show plots
    if show_steps:
        print("📊 Displaying visualizations...")
        print("   Close the windows to continue")
        plt.show()
    else:
        plt.close('all')
    
    print()
    print("✅ DEMO COMPLETED!")
    print()
    print("📁 Output files in: demo_output/")
    print(f"   - {mask_file.name}")
    print(f"   - {viz_file.name}")
    print(f"   - {overlay_file.name}")
    print()
    print("🎯 NEXT STEPS:")
    print("   1. Check the output visualizations")
    print("   2. If satisfied, run full workflow:")
    print("      python run_annotation_workflow.py")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Demo annotation on a single image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_single_image.py
  python demo_single_image.py --image my_dataset/train/images/frame_0049.png
  python demo_single_image.py --image my_dataset/train/images/frame_0049.png --no-show
        """
    )
    parser.add_argument('--image', type=str, 
                        default='my_dataset/train/images/frame_0049.png',
                        help='Path to image file')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (save only)')
    
    args = parser.parse_args()
    
    # Check if opencv is installed
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV is not installed!")
        print("   Install it with: pip install opencv-python")
        sys.exit(1)
    
    # Check if skimage is installed
    try:
        import skimage
    except ImportError:
        print("❌ scikit-image is not installed!")
        print("   Install it with: pip install scikit-image")
        sys.exit(1)
    
    # Run demo
    demo_single_image(args.image, show_steps=not args.no_show)


if __name__ == '__main__':
    main()
