"""
Quick Start Script - Chạy toàn bộ annotation workflow
Tự động hóa tất cả các bước từ annotation → validation → comparison

Usage:
    python run_annotation_workflow.py
    python run_annotation_workflow.py --quick  # Chạy nhanh với ít samples
"""

import subprocess
import sys
from pathlib import Path
import argparse
import time


class AnnotationWorkflow:
    """Class để orchestrate toàn bộ annotation workflow"""
    
    def __init__(self, quick_mode=False):
        self.quick_mode = quick_mode
        self.base_dir = Path('my_dataset')
        
    def print_step(self, step_num, title):
        """In header cho mỗi bước"""
        print("\n" + "="*70)
        print(f"STEP {step_num}: {title}")
        print("="*70 + "\n")
    
    def run_command(self, cmd, description):
        """Chạy command với error handling"""
        print(f"🚀 {description}...")
        print(f"   Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"\n✅ {description} - COMPLETED\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} - FAILED")
            print(f"   Error: {e}")
            return False
    
    def check_directories(self):
        """Kiểm tra directories có tồn tại không"""
        print("🔍 Checking directories...")
        
        required_dirs = [
            self.base_dir / 'train' / 'images',
            self.base_dir / 'train' / 'masks',
            self.base_dir / 'val' / 'images',
            self.base_dir / 'val' / 'masks',
        ]
        
        all_exist = True
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"   ✓ {dir_path}")
            else:
                print(f"   ✗ {dir_path} - NOT FOUND")
                all_exist = False
        
        if not all_exist:
            print("\n❌ Some directories are missing!")
            print("   Please make sure you have:")
            print("   - my_dataset/train/images/")
            print("   - my_dataset/train/masks/")
            print("   - my_dataset/val/images/")
            print("   - my_dataset/val/masks/")
            return False
        
        print("\n✅ All directories found\n")
        return True
    
    def step1_create_annotations(self):
        """Step 1: Tạo annotations mới"""
        self.print_step(1, "CREATE NEW ANNOTATIONS")
        
        # Parameters
        visualize = 3 if self.quick_mode else 5
        
        # Training set
        cmd_train = [
            sys.executable, 'advanced_annotation.py',
            '--images-dir', 'my_dataset/train/images',
            '--masks-dir', 'my_dataset/train/masks_new',
            '--min-size', '50',
            '--max-size', '5000',
            '--min-circularity', '0.3',
            '--visualize', str(visualize),
        ]
        
        if not self.run_command(cmd_train, "Annotate training set"):
            return False
        
        # Validation set
        cmd_val = [
            sys.executable, 'advanced_annotation.py',
            '--images-dir', 'my_dataset/val/images',
            '--masks-dir', 'my_dataset/val/masks_new',
            '--min-size', '50',
            '--max-size', '5000',
            '--min-circularity', '0.3',
            '--visualize', str(visualize),
        ]
        
        if not self.run_command(cmd_val, "Annotate validation set"):
            return False
        
        return True
    
    def step2_compare_annotations(self):
        """Step 2: So sánh old vs new"""
        self.print_step(2, "COMPARE OLD vs NEW ANNOTATIONS")
        
        # Parameters
        num_samples = 5 if self.quick_mode else 10
        
        # Training set
        cmd_train = [
            sys.executable, 'compare_annotations.py',
            '--images-dir', 'my_dataset/train/images',
            '--old-dir', 'my_dataset/train/masks',
            '--new-dir', 'my_dataset/train/masks_new',
            '--output-dir', 'comparison_results_train',
            '--num-samples', str(num_samples),
        ]
        
        if not self.run_command(cmd_train, "Compare training annotations"):
            return False
        
        # Validation set
        cmd_val = [
            sys.executable, 'compare_annotations.py',
            '--images-dir', 'my_dataset/val/images',
            '--old-dir', 'my_dataset/val/masks',
            '--new-dir', 'my_dataset/val/masks_new',
            '--output-dir', 'comparison_results_val',
            '--num-samples', str(num_samples // 2 + 1),
        ]
        
        if not self.run_command(cmd_val, "Compare validation annotations"):
            return False
        
        return True
    
    def step3_validate_quality(self):
        """Step 3: Validate annotation quality"""
        self.print_step(3, "VALIDATE ANNOTATION QUALITY")
        
        # Training set
        cmd_train = [
            sys.executable, 'validate_annotations.py',
            '--old-dir', 'my_dataset/train/masks',
            '--new-dir', 'my_dataset/train/masks_new',
            '--output-dir', 'validation_results_train',
        ]
        
        if not self.run_command(cmd_train, "Validate training annotations"):
            return False
        
        # Validation set
        cmd_val = [
            sys.executable, 'validate_annotations.py',
            '--old-dir', 'my_dataset/val/masks',
            '--new-dir', 'my_dataset/val/masks_new',
            '--output-dir', 'validation_results_val',
        ]
        
        if not self.run_command(cmd_val, "Validate validation annotations"):
            return False
        
        return True
    
    def print_summary(self):
        """In summary và hướng dẫn tiếp theo"""
        self.print_step(4, "WORKFLOW COMPLETED!")
        
        print("📊 RESULTS GENERATED:")
        print()
        print("1. NEW ANNOTATIONS:")
        print("   - my_dataset/train/masks_new/")
        print("   - my_dataset/val/masks_new/")
        print("   - my_dataset/train/visualizations/")
        print("   - my_dataset/val/visualizations/")
        print()
        print("2. COMPARISONS:")
        print("   - comparison_results_train/comparison_*.png")
        print("   - comparison_results_val/comparison_*.png")
        print()
        print("3. VALIDATION REPORTS:")
        print("   - validation_results_train/validation_report.txt")
        print("   - validation_results_train/validation_plots.png")
        print("   - validation_results_train/comparison_detailed.csv")
        print("   - validation_results_val/ (same structure)")
        print()
        print("="*70)
        print()
        print("🎯 NEXT STEPS:")
        print()
        print("1. REVIEW RESULTS:")
        print("   • Open validation_results_*/validation_report.txt")
        print("   • Check validation_results_*/validation_plots.png")
        print("   • Browse comparison_results_*/comparison_*.png")
        print()
        print("2. IF QUALITY IS GOOD:")
        print("   • Backup old masks:")
        print("     mv my_dataset/train/masks my_dataset/train/masks_old")
        print("     mv my_dataset/val/masks my_dataset/val/masks_old")
        print("   • Deploy new masks:")
        print("     mv my_dataset/train/masks_new my_dataset/train/masks")
        print("     mv my_dataset/val/masks_new my_dataset/val/masks")
        print("   • Upload to Google Drive and run COLAB training")
        print()
        print("3. IF ADJUSTMENTS NEEDED:")
        print("   • Edit parameters in advanced_annotation.py")
        print("   • Run workflow again: python run_annotation_workflow.py")
        print()
        print("="*70)
        print()
        print("✨ Happy Training! Your annotations should be much better now!")
        print()
    
    def run(self):
        """Chạy toàn bộ workflow"""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("ADVANCED ANNOTATION WORKFLOW")
        print("="*70)
        print()
        if self.quick_mode:
            print("⚡ Running in QUICK MODE (fewer samples)")
        else:
            print("🐢 Running in FULL MODE (all checks)")
        print()
        
        # Check directories
        if not self.check_directories():
            return False
        
        # Step 1: Create annotations
        if not self.step1_create_annotations():
            print("\n❌ Workflow FAILED at Step 1")
            return False
        
        # Step 2: Compare
        if not self.step2_compare_annotations():
            print("\n❌ Workflow FAILED at Step 2")
            return False
        
        # Step 3: Validate
        if not self.step3_validate_quality():
            print("\n❌ Workflow FAILED at Step 3")
            return False
        
        # Summary
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        self.print_summary()
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run complete annotation workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_annotation_workflow.py              # Full mode
  python run_annotation_workflow.py --quick      # Quick mode (fewer samples)
        """
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode - fewer visualization samples')
    
    args = parser.parse_args()
    
    # Run workflow
    workflow = AnnotationWorkflow(quick_mode=args.quick)
    success = workflow.run()
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
