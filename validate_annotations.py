"""
Script validation để kiểm tra chất lượng annotations
So sánh annotations cũ vs mới và đưa ra metrics

Usage:
    python validate_annotations.py
    python validate_annotations.py --old-dir my_dataset/train/masks --new-dir my_dataset/train/masks_new
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import pandas as pd
import argparse
from scipy import ndimage


class AnnotationValidator:
    """
    Class để validate và so sánh annotations
    """
    
    def __init__(self):
        self.results = []
    
    def analyze_mask(self, mask, label=""):
        """
        Phân tích chi tiết một mask
        
        Returns:
            Dictionary chứa metrics
        """
        if mask.max() == 0:
            return {
                'label': label,
                'n_cells': 0,
                'avg_size': 0,
                'std_size': 0,
                'min_size': 0,
                'max_size': 0,
                'very_small_cells': 0,
                'very_large_cells': 0,
                'avg_circularity': 0,
                'cells_with_holes': 0,
                'avg_solidity': 0,
            }
        
        regions = regionprops(mask)
        
        # Collect metrics
        sizes = [r.area for r in regions]
        circularities = []
        solidities = []
        holes_count = 0
        
        for r in regions:
            # Circularity
            if r.perimeter > 0:
                circ = 4 * np.pi * r.area / (r.perimeter ** 2)
                circularities.append(circ)
            
            # Solidity
            solidities.append(r.solidity)
            
            # Has holes?
            if r.euler_number < 1:
                holes_count += 1
        
        metrics = {
            'label': label,
            'n_cells': len(regions),
            'avg_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'very_small_cells': sum(1 for s in sizes if s < 50),
            'very_large_cells': sum(1 for s in sizes if s > 5000),
            'avg_circularity': np.mean(circularities) if circularities else 0,
            'cells_with_holes': holes_count,
            'avg_solidity': np.mean(solidities) if solidities else 0,
        }
        
        return metrics
    
    def compare_masks(self, old_mask, new_mask, image_name=""):
        """
        So sánh 2 masks (old vs new)
        
        Returns:
            Dictionary chứa comparison metrics
        """
        old_metrics = self.analyze_mask(old_mask, label="old")
        new_metrics = self.analyze_mask(new_mask, label="new")
        
        comparison = {
            'image': image_name,
            'old_n_cells': old_metrics['n_cells'],
            'new_n_cells': new_metrics['n_cells'],
            'cell_change': new_metrics['n_cells'] - old_metrics['n_cells'],
            'cell_change_pct': ((new_metrics['n_cells'] - old_metrics['n_cells']) / max(old_metrics['n_cells'], 1)) * 100,
            
            'old_avg_size': old_metrics['avg_size'],
            'new_avg_size': new_metrics['avg_size'],
            'size_change': new_metrics['avg_size'] - old_metrics['avg_size'],
            
            'old_very_small': old_metrics['very_small_cells'],
            'new_very_small': new_metrics['very_small_cells'],
            'small_removed': old_metrics['very_small_cells'] - new_metrics['very_small_cells'],
            
            'old_circularity': old_metrics['avg_circularity'],
            'new_circularity': new_metrics['avg_circularity'],
            'circularity_improvement': new_metrics['avg_circularity'] - old_metrics['avg_circularity'],
            
            'old_holes': old_metrics['cells_with_holes'],
            'new_holes': new_metrics['cells_with_holes'],
            'holes_fixed': old_metrics['cells_with_holes'] - new_metrics['cells_with_holes'],
            
            'old_solidity': old_metrics['avg_solidity'],
            'new_solidity': new_metrics['avg_solidity'],
        }
        
        return comparison
    
    def validate_dataset(self, old_dir, new_dir, output_dir='validation_results'):
        """
        Validate toàn bộ dataset
        
        Args:
            old_dir: Directory chứa old masks
            new_dir: Directory chứa new masks
            output_dir: Directory để save results
        """
        old_path = Path(old_dir)
        new_path = Path(new_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Get all mask files
        old_files = sorted(old_path.glob('*.png'))
        new_files = sorted(new_path.glob('*.png'))
        
        if len(old_files) == 0:
            print(f"❌ Không tìm thấy masks cũ trong {old_dir}")
            return
        
        if len(new_files) == 0:
            print(f"❌ Không tìm thấy masks mới trong {new_dir}")
            return
        
        print(f"🔍 Validating {len(old_files)} masks...")
        print(f"   Old dir: {old_dir}")
        print(f"   New dir: {new_dir}")
        print()
        
        # Compare all masks
        comparisons = []
        
        for old_file in tqdm(old_files, desc="Comparing"):
            new_file = new_path / old_file.name
            
            if not new_file.exists():
                print(f"⚠️  Missing new mask for {old_file.name}")
                continue
            
            # Load masks
            old_mask = np.array(Image.open(old_file))
            new_mask = np.array(Image.open(new_file))
            
            # Compare
            comp = self.compare_masks(old_mask, new_mask, old_file.stem)
            comparisons.append(comp)
        
        # Convert to DataFrame
        df = pd.DataFrame(comparisons)
        
        # Save detailed results
        csv_file = output_path / 'comparison_detailed.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Detailed results saved to: {csv_file}")
        
        # Print summary statistics
        self._print_summary(df)
        
        # Create visualizations
        self._create_plots(df, output_path)
        
        # Generate report
        self._generate_report(df, output_path)
        
        return df
    
    def _print_summary(self, df):
        """In summary statistics"""
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\n📊 CELL COUNT:")
        print(f"   Old total:     {df['old_n_cells'].sum():>6} cells")
        print(f"   New total:     {df['new_n_cells'].sum():>6} cells")
        print(f"   Change:        {df['cell_change'].sum():>6} cells ({df['cell_change'].sum()/df['old_n_cells'].sum()*100:+.1f}%)")
        print(f"   Avg per image: {df['new_n_cells'].mean():.1f} cells")
        
        print(f"\n🧹 NOISE REMOVAL:")
        print(f"   Very small cells removed: {df['small_removed'].sum():>5} ({df['small_removed'].sum()/df['old_very_small'].sum()*100:.1f}%)")
        print(f"   Old very small cells:     {df['old_very_small'].sum():>5}")
        print(f"   New very small cells:     {df['new_very_small'].sum():>5}")
        
        print(f"\n⭕ SHAPE QUALITY:")
        print(f"   Old avg circularity: {df['old_circularity'].mean():.3f}")
        print(f"   New avg circularity: {df['new_circularity'].mean():.3f}")
        print(f"   Improvement:         {df['circularity_improvement'].mean():+.3f}")
        
        print(f"\n🕳️  HOLES FIXED:")
        print(f"   Old cells with holes: {df['old_holes'].sum():>4}")
        print(f"   New cells with holes: {df['new_holes'].sum():>4}")
        print(f"   Holes fixed:          {df['holes_fixed'].sum():>4}")
        
        print(f"\n📏 CELL SIZE:")
        print(f"   Old avg size: {df['old_avg_size'].mean():.1f} ± {df['old_avg_size'].std():.1f} pixels")
        print(f"   New avg size: {df['new_avg_size'].mean():.1f} ± {df['new_avg_size'].std():.1f} pixels")
        
        print(f"\n🎯 SOLIDITY (Compactness):")
        print(f"   Old avg solidity: {df['old_solidity'].mean():.3f}")
        print(f"   New avg solidity: {df['new_solidity'].mean():.3f}")
        print(f"   Improvement:      {(df['new_solidity'].mean() - df['old_solidity'].mean()):+.3f}")
        
        print(f"{'='*70}\n")
    
    def _create_plots(self, df, output_dir):
        """Tạo visualization plots"""
        output_path = Path(output_dir)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Cell count comparison
        ax1 = plt.subplot(2, 3, 1)
        x = np.arange(len(df))
        ax1.plot(x, df['old_n_cells'], 'o-', label='Old', alpha=0.7, color='red')
        ax1.plot(x, df['new_n_cells'], 'o-', label='New', alpha=0.7, color='green')
        ax1.set_xlabel('Image Index')
        ax1.set_ylabel('Number of Cells')
        ax1.set_title('Cell Count: Old vs New')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cell size distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist([df['old_avg_size'], df['new_avg_size']], 
                 bins=30, label=['Old', 'New'], alpha=0.7, color=['red', 'green'])
        ax2.set_xlabel('Average Cell Size (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Cell Size Distribution')
        ax2.legend()
        
        # 3. Circularity improvement
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(df['old_circularity'], df['new_circularity'], alpha=0.5)
        min_val = min(df['old_circularity'].min(), df['new_circularity'].min())
        max_val = max(df['old_circularity'].max(), df['new_circularity'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='No change')
        ax3.set_xlabel('Old Circularity')
        ax3.set_ylabel('New Circularity')
        ax3.set_title('Circularity: Old vs New')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Very small cells removed
        ax4 = plt.subplot(2, 3, 4)
        ax4.bar(['Old', 'New'], 
                [df['old_very_small'].sum(), df['new_very_small'].sum()],
                color=['red', 'green'], alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title('Very Small Cells (< 50 pixels)')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Holes fixed
        ax5 = plt.subplot(2, 3, 5)
        ax5.bar(['Old', 'New'], 
                [df['old_holes'].sum(), df['new_holes'].sum()],
                color=['red', 'green'], alpha=0.7)
        ax5.set_ylabel('Count')
        ax5.set_title('Cells with Holes')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Solidity comparison
        ax6 = plt.subplot(2, 3, 6)
        ax6.boxplot([df['old_solidity'].dropna(), df['new_solidity'].dropna()],
                    labels=['Old', 'New'])
        ax6.set_ylabel('Solidity')
        ax6.set_title('Solidity Distribution')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = output_path / 'validation_plots.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to: {plot_file}")
    
    def _generate_report(self, df, output_dir):
        """Tạo text report"""
        output_path = Path(output_dir)
        report_file = output_path / 'validation_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ANNOTATION QUALITY VALIDATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total images analyzed: {len(df)}\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*70 + "\n\n")
            
            f.write("Cell Count:\n")
            f.write(f"  Old total:     {df['old_n_cells'].sum()} cells\n")
            f.write(f"  New total:     {df['new_n_cells'].sum()} cells\n")
            f.write(f"  Change:        {df['cell_change'].sum()} cells ({df['cell_change'].sum()/df['old_n_cells'].sum()*100:+.1f}%)\n\n")
            
            f.write("Quality Improvements:\n")
            f.write(f"  Very small cells removed: {df['small_removed'].sum()}\n")
            f.write(f"  Holes fixed:              {df['holes_fixed'].sum()}\n")
            f.write(f"  Avg circularity change:   {df['circularity_improvement'].mean():+.3f}\n")
            f.write(f"  Avg solidity improvement: {(df['new_solidity'].mean() - df['old_solidity'].mean()):+.3f}\n\n")
            
            f.write("TOP 10 MOST IMPROVED IMAGES (by noise removal):\n")
            f.write("-"*70 + "\n")
            top_improved = df.nlargest(10, 'small_removed')[['image', 'old_n_cells', 'new_n_cells', 'small_removed']]
            f.write(top_improved.to_string(index=False))
            f.write("\n\n")
            
            f.write("IMAGES WITH POTENTIAL ISSUES (very few cells detected):\n")
            f.write("-"*70 + "\n")
            low_cells = df[df['new_n_cells'] < 10][['image', 'new_n_cells', 'old_n_cells']]
            if len(low_cells) > 0:
                f.write(low_cells.to_string(index=False))
            else:
                f.write("No issues found.\n")
            f.write("\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"✅ Report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate annotation quality')
    parser.add_argument('--old-dir', type=str, default='my_dataset/train/masks',
                        help='Directory containing old masks')
    parser.add_argument('--new-dir', type=str, default='my_dataset/train/masks_new',
                        help='Directory containing new masks')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                        help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Create validator
    validator = AnnotationValidator()
    
    # Run validation
    validator.validate_dataset(args.old_dir, args.new_dir, args.output_dir)


if __name__ == '__main__':
    main()
