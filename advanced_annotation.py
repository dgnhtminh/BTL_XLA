"""
Advanced Annotation Tool với Adaptive Watershed
Phiên bản cải tiến để tạo annotations chất lượng cao

Cải tiến so với phương pháp cũ:
- Sử dụng Distance Transform để tách cells dính nhau
- Adaptive thresholding thay vì threshold cố định
- Morphological operations để làm sạch và mượt boundaries
- Loại bỏ noise và artifacts một cách thông minh
- Quality validation để đảm bảo chất lượng annotations

Author: Advanced Annotation System
Date: 2025-11-28
"""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from skimage import morphology, filters, measure, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
import json
from datetime import datetime


class AdvancedAnnotator:
    """
    Class để tạo annotations chất lượng cao sử dụng kỹ thuật advanced
    """
    
    def __init__(self, 
                 min_cell_size=50,      # Tăng từ 20 -> 50 pixels
                 max_cell_size=5000,    # Giới hạn trên để loại artifacts
                 min_circularity=0.3,   # Cell phải có hình dạng tương đối tròn
                 gaussian_sigma=1.5,    # Smoothing trước khi segment
                 distance_threshold=0.4, # Threshold cho peak detection
                 ):
        """
        Args:
            min_cell_size: Kích thước tối thiểu của cell (pixels)
            max_cell_size: Kích thước tối đa của cell (pixels)
            min_circularity: Circularity tối thiểu (0-1, 1 = perfect circle)
            gaussian_sigma: Sigma cho Gaussian blur
            distance_threshold: Threshold cho local maxima trong distance transform
        """
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.min_circularity = min_circularity
        self.gaussian_sigma = gaussian_sigma
        self.distance_threshold = distance_threshold
        
        self.stats = {
            'total_processed': 0,
            'cells_detected': 0,
            'noise_removed': 0,
            'cells_merged': 0,
        }
    
    def preprocess_image(self, image):
        """
        Tiền xử lý ảnh để cải thiện segmentation
        
        Args:
            image: Input image (grayscale or RGB)
        
        Returns:
            Preprocessed image (grayscale, normalized)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Normalize to 0-255
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (0, 0), self.gaussian_sigma)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # để cải thiện contrast cục bộ
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        return enhanced
    
    def create_binary_mask(self, image):
        """
        Tạo binary mask sử dụng adaptive thresholding
        
        Args:
            image: Preprocessed grayscale image
        
        Returns:
            Binary mask (uint8)
        """
        # Adaptive threshold - tốt hơn global threshold
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # INV vì cells là dark
            blockSize=21,  # Kích thước vùng local
            C=5  # Constant trừ đi từ mean
        )
        
        # Morphological operations để làm sạch
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Close small gaps
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium)
        
        # Fill holes trong cells
        binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
        
        return binary
    
    def watershed_segmentation(self, binary_mask):
        """
        Sử dụng watershed với distance transform để tách cells
        
        Args:
            binary_mask: Binary mask
        
        Returns:
            Instance segmentation mask
        """
        # Distance transform
        # Mỗi pixel = khoảng cách đến background gần nhất
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Smooth distance map
        distance = ndimage.gaussian_filter(distance, sigma=1)
        
        # Find local maxima (cell centers)
        # Sử dụng threshold động dựa trên distance max
        max_dist = distance.max()
        min_distance = int(max(3, max_dist * self.distance_threshold))
        
        local_max = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=binary_mask,
            exclude_border=False
        )
        
        # Create markers for watershed
        markers = np.zeros_like(binary_mask, dtype=np.int32)
        markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
        markers = ndimage.label(markers)[0]
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary_mask)
        
        return labels
    
    def filter_objects(self, labels):
        """
        Lọc và làm sạch các objects theo criteria
        
        Args:
            labels: Instance segmentation mask
        
        Returns:
            Filtered mask
        """
        # Measure properties of each object
        regions = measure.regionprops(labels)
        
        # Create new mask
        filtered_mask = np.zeros_like(labels)
        current_label = 1
        
        noise_count = 0
        
        for region in regions:
            area = region.area
            perimeter = region.perimeter
            
            # Calculate circularity: 4π * area / perimeter²
            # Circle = 1.0, square = 0.785, line = 0
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
            
            # Filter criteria
            is_valid = True
            
            # Size filter
            if area < self.min_cell_size or area > self.max_cell_size:
                is_valid = False
                noise_count += 1
            
            # Shape filter - loại bỏ objects quá dài/mảnh
            if circularity < self.min_circularity:
                is_valid = False
                noise_count += 1
            
            # Solidity filter - loại bỏ objects có nhiều lỗ/rất không đều
            if region.solidity < 0.7:  # solidity = area / convex_hull_area
                is_valid = False
                noise_count += 1
            
            # Keep valid objects
            if is_valid:
                filtered_mask[labels == region.label] = current_label
                current_label += 1
        
        self.stats['noise_removed'] += noise_count
        
        return filtered_mask
    
    def post_process_mask(self, mask):
        """
        Post-processing để cải thiện boundaries
        
        Args:
            mask: Filtered instance mask
        
        Returns:
            Post-processed mask
        """
        processed_mask = np.zeros_like(mask)
        
        for label_id in range(1, mask.max() + 1):
            # Extract single cell
            cell = (mask == label_id).astype(np.uint8)
            
            # Smooth boundary với morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)
            
            # Fill any remaining holes
            cell = ndimage.binary_fill_holes(cell).astype(np.uint8)
            
            # Add to processed mask
            processed_mask[cell > 0] = label_id
        
        return processed_mask
    
    def annotate_image(self, image, visualize=False):
        """
        Pipeline đầy đủ để annotate một ảnh
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            visualize: Nếu True, return visualization
        
        Returns:
            mask: Instance segmentation mask
            viz: (optional) Visualization figure
        """
        # Step 1: Preprocess
        preprocessed = self.preprocess_image(image)
        
        # Step 2: Create binary mask
        binary = self.create_binary_mask(preprocessed)
        
        # Step 3: Watershed segmentation
        labels = self.watershed_segmentation(binary)
        
        # Step 4: Filter objects
        filtered = self.filter_objects(labels)
        
        # Step 5: Post-process
        final_mask = self.post_process_mask(filtered)
        
        # Update stats
        self.stats['total_processed'] += 1
        self.stats['cells_detected'] += final_mask.max()
        
        if visualize:
            return final_mask, self._create_visualization(image, preprocessed, binary, labels, final_mask)
        
        return final_mask
    
    def _create_visualization(self, original, preprocessed, binary, watershed_result, final):
        """Tạo visualization để debug"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')
        
        # Preprocessed
        axes[0, 1].imshow(preprocessed, cmap='gray')
        axes[0, 1].set_title('2. Preprocessed (CLAHE)')
        axes[0, 1].axis('off')
        
        # Binary
        axes[0, 2].imshow(binary, cmap='gray')
        axes[0, 2].set_title('3. Binary Mask')
        axes[0, 2].axis('off')
        
        # Watershed result
        axes[1, 0].imshow(watershed_result, cmap='nipy_spectral')
        n_watershed = watershed_result.max()
        axes[1, 0].set_title(f'4. Watershed ({n_watershed} objects)')
        axes[1, 0].axis('off')
        
        # Final result
        axes[1, 1].imshow(final, cmap='nipy_spectral')
        n_final = final.max()
        axes[1, 1].set_title(f'5. Final ({n_final} cells) [-{n_watershed-n_final}]')
        axes[1, 1].axis('off')
        
        # Overlay
        axes[1, 2].imshow(original, cmap='gray')
        axes[1, 2].imshow(final, cmap='nipy_spectral', alpha=0.4)
        axes[1, 2].set_title('6. Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def annotate_dataset(self, images_dir, masks_dir, visualize_samples=3):
        """
        Annotate toàn bộ dataset
        
        Args:
            images_dir: Thư mục chứa images
            masks_dir: Thư mục output cho masks
            visualize_samples: Số lượng samples để visualize
        """
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        masks_path.mkdir(exist_ok=True, parents=True)
        
        # Get all images
        image_files = sorted(images_path.glob('*.png'))
        
        if len(image_files) == 0:
            print(f"❌ Không tìm thấy ảnh trong {images_dir}")
            return
        
        print(f"🎯 Bắt đầu annotate {len(image_files)} images...")
        print(f"   Settings:")
        print(f"     - min_cell_size: {self.min_cell_size} pixels")
        print(f"     - max_cell_size: {self.max_cell_size} pixels")
        print(f"     - min_circularity: {self.min_circularity}")
        print(f"     - gaussian_sigma: {self.gaussian_sigma}")
        print()
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'cells_detected': 0,
            'noise_removed': 0,
            'cells_merged': 0,
        }
        
        # Process all images
        for i, img_file in enumerate(tqdm(image_files, desc="Annotating")):
            # Load image
            image = np.array(Image.open(img_file))
            
            # Annotate
            visualize = (i < visualize_samples)
            if visualize:
                mask, fig = self.annotate_image(image, visualize=True)
                # Save visualization
                viz_path = masks_path.parent / 'visualizations'
                viz_path.mkdir(exist_ok=True)
                fig.savefig(viz_path / f'{img_file.stem}_pipeline.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                mask = self.annotate_image(image, visualize=False)
            
            # Save mask
            mask_file = masks_path / img_file.name
            Image.fromarray(mask.astype(np.uint16)).save(mask_file)
        
        # Print statistics
        print(f"\n{'='*60}")
        print("ANNOTATION STATISTICS")
        print(f"{'='*60}")
        print(f"Total images processed:  {self.stats['total_processed']}")
        print(f"Total cells detected:    {self.stats['cells_detected']}")
        print(f"Noise objects removed:   {self.stats['noise_removed']}")
        print(f"Average cells per image: {self.stats['cells_detected']/self.stats['total_processed']:.1f}")
        print(f"{'='*60}")
        
        # Save config - Convert numpy types to Python types
        config = {
            'date': datetime.now().isoformat(),
            'parameters': {
                'min_cell_size': int(self.min_cell_size),
                'max_cell_size': int(self.max_cell_size),
                'min_circularity': float(self.min_circularity),
                'gaussian_sigma': float(self.gaussian_sigma),
                'distance_threshold': float(self.distance_threshold),
            },
            'statistics': {
                'total_processed': int(self.stats['total_processed']),
                'cells_detected': int(self.stats['cells_detected']),
                'noise_removed': int(self.stats['noise_removed']),
                'cells_merged': int(self.stats['cells_merged']),
            }
        }
        
        config_file = masks_path.parent / 'annotation_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to: {config_file}")


def main():
    """Main function để chạy annotation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Annotation Tool')
    parser.add_argument('--images-dir', type=str, default='my_dataset/train/images',
                        help='Directory containing images')
    parser.add_argument('--masks-dir', type=str, default='my_dataset/train/masks_new',
                        help='Output directory for masks')
    parser.add_argument('--min-size', type=int, default=50,
                        help='Minimum cell size (pixels)')
    parser.add_argument('--max-size', type=int, default=5000,
                        help='Maximum cell size (pixels)')
    parser.add_argument('--min-circularity', type=float, default=0.3,
                        help='Minimum circularity (0-1)')
    parser.add_argument('--visualize', type=int, default=5,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create annotator
    annotator = AdvancedAnnotator(
        min_cell_size=args.min_size,
        max_cell_size=args.max_size,
        min_circularity=args.min_circularity,
    )
    
    # Run annotation
    annotator.annotate_dataset(
        args.images_dir,
        args.masks_dir,
        visualize_samples=args.visualize
    )


if __name__ == '__main__':
    main()
