# 🔬 Cell Detection & Segmentation using StarDist

Dự án phát hiện và phân đoạn tế bào tự động từ ảnh kính hiển vi sử dụng deep learning model **StarDist**.

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Kết quả](#-kết-quả)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

## 🎯 Giới thiệu

Dự án này thực hiện **instance segmentation** cho tế bào trong ảnh kính hiển vi, giúp:
- Tự động phát hiện và đếm số lượng tế bào
- Phân đoạn chính xác từng tế bào riêng lẻ
- Xử lý trường hợp tế bào chồng lấp nhau

**Công nghệ sử dụng:**
- **StarDist 2D**: Mô hình deep learning dựa trên star-convex polygons
- **TensorFlow/Keras**: Framework training model
- **Data Augmentation**: Tăng cường dữ liệu đa dạng (rotation, flip, elastic, brightness, contrast)

## ✨ Tính năng

- 🔍 **Phát hiện tế bào**: Tự động detect tất cả tế bào trong ảnh
- 🎨 **Phân đoạn chính xác**: Tạo mask riêng biệt cho từng tế bào
- 📊 **Đánh giá hiệu suất**: Tính toán metrics (AP, Precision, Recall, F1-score)
- 🎲 **Chọn frames thông minh**: Lựa chọn dữ liệu đa dạng cho annotation
- 📈 **Visualization**: Hiển thị kết quả dự đoán với overlay masks

## 📁 Cấu trúc thư mục

```
stardist_project/
├── 0_select_frames_for_annotation.py  # Script chọn frames cho annotation
├── 0_pre_label_frames.py              # Script tiền xử lý labels
├── 1_COLAB_TRAINING_IMPROVED.ipynb    # Notebook training model
├── 2_prediction_with_metrics.ipynb    # Notebook inference và đánh giá
├── selected_frames_train.txt          # Danh sách frames training
├── selected_frames_val.txt            # Danh sách frames validation
├── frames/                            # Thư mục chứa frames gốc
├── my_dataset/                        # Dataset đã annotation
│   ├── train/
│   │   ├── images/                    # 174 ảnh training
│   │   └── masks/                     # 174 masks training
│   └── val/
│       ├── images/                    # 59 ảnh validation
│       └── masks/                     # 59 masks validation
├── models/                            # Thư mục chứa trained models
│   └── stardist_my_data_v2_improved/
│       ├── config.json
│       ├── weights_best.h5
│       ├── thresholds.json
│       └── training_summary.txt
├── predictions/                       # Kết quả dự đoán
│   ├── masks/                         # Predicted masks
│   ├── overlays/                      # Visualization overlays
│   ├── detection_metrics.csv         # Metrics tổng hợp
│   └── detailed_objects.csv          # Chi tiết từng object
└── stardist/                         # Source code StarDist (modified)
```

## 🔧 Cài đặt

### Yêu cầu
- Python 3.7+
- TensorFlow 2.x
- CUDA (khuyến nghị cho training nhanh)

### Cài đặt thư viện

```bash
pip install tensorflow
pip install stardist
pip install csbdeep
pip install numpy pandas matplotlib pillow scikit-image tqdm
```

Hoặc sử dụng requirements.txt:

```bash
pip install -r requirements.txt
```

## 🚀 Sử dụng

### 1. Chuẩn bị dữ liệu

**Chọn frames cho annotation:**
```bash
python 0_select_frames_for_annotation.py
```

Script này sẽ:
- Chọn 200 frames đa dạng từ 800 frames gốc
- Chia thành 150 frames training + 50 frames validation
- Sử dụng chiến lược chọn thông minh (temporal + diversity)

### 2. Training model

Mở và chạy `1_COLAB_TRAINING_IMPROVED.ipynb`:
- Cấu hình model với 64 rays, patch size 256x256
- Sử dụng strong augmentation
- Training 150 epochs với early stopping
- Lưu best model vào `models/`

**Cấu hình training:**
- Batch size: 4
- Learning rate: 0.0003
- Steps per epoch: 200
- Augmentation: rotation + flip + elastic + brightness + contrast + noise

### 3. Dự đoán và đánh giá

Mở và chạy `2_prediction_with_metrics.ipynb`:
- Load trained model
- Dự đoán trên validation set
- Tính toán metrics (AP, Precision, Recall, F1)
- Export kết quả và visualization

## 📊 Kết quả

### Hiệu suất model

Model **stardist_my_data_v2_improved** đạt được:

| Metric | Giá trị |
|--------|---------|
| **AP@0.5** | 0.802 |
| **AP@0.6** | 0.814 |
| **AP@0.7** | 0.832 |
| **AP@0.8** | 0.862 |
| **AP@0.9** | 0.916 |
| **Đánh giá** | ⭐⭐ VERY GOOD |

### Dataset

- **Training**: 174 ảnh
- **Validation**: 59 ảnh
- **Total**: 233 ảnh đã annotation

### Ví dụ kết quả

Model có khả năng:
- ✅ Phát hiện chính xác tế bào với IoU cao
- ✅ Phân biệt tế bào chồng lấp
- ✅ Xử lý tốt biến đổi về độ sáng/tương phản
- ✅ Segmentation chính xác biên tế bào

## 📖 Tài liệu tham khảo

### Papers
- [StarDist - Object Detection with Star-convex Shapes](https://arxiv.org/abs/1806.03535)
- [Cell Detection with Star-convex Polygons](https://arxiv.org/abs/2006.14109)

### Code & Documentation
- [StarDist GitHub](https://github.com/stardist/stardist)
- [StarDist Documentation](https://stardist.net/)

Dự án Xử lý Ảnh - Phát hiện và Phân đoạn Tế bào

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

⭐ Nếu thấy hữu ích, hãy star repo này!
