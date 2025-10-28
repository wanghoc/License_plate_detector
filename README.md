# 🚗 Hệ thống Nhận diện Biển số Xe # - License Plate Recognition

Nhận diện biển số xe Việt Nam qua webcam sử dụng YOLOv8 OBB để **nhận diện biển số xe** và **EasyOCR** để **đọc ký tự** trên biển số qua webcam.

## 🚀 CÁCH CHẠY## 🎯 Tính năng chính

### Cách 1: Double-click file✅ **Nhận diện biển số xe** (OBB - Oriented Bounding Box)

````✅ **Đọc ký tự trên biển số** (OCR - Optical Character Recognition)

run.bat✅ **Xác thực format biển số** Việt Nam

```✅ **Hiển thị real-time** qua webcam

✅ **Lưu ảnh kết quả** và ảnh biển số đã crop

### Cách 2: Command line

```bash## 🔧 Yêu cầu hệ thống

Python39\python.exe license_plate_detector.py

```- Python 3.8 trở lên

- Webcam

## 📸 Tính năng- Model đã train: `plate_detector_best.pt`



✅ Nhận diện biển số xe real-time  ## 📦 Cài đặt

✅ Đọc ký tự trên biển số (OCR)

✅ **Tự động lưu ảnh** khi phát hiện biển số hợp lệ  1. Cài đặt các thư viện cần thiết:

✅ Mỗi biển số chỉ lưu 1 lần

✅ Lưu vào thư mục `img/` (full + crop)```bash

pip install -r requirements.txt

## 🎮 Sử dụng```



1. Chạy `run.bat`**Lưu ý**: Lần đầu chạy, EasyOCR sẽ tự động tải model OCR (khoảng 150MB), có thể mất vài phút.

2. Đợi khởi tạo (lần đầu mất 2-3 phút)

3. Đưa biển số xe trước camera## 🚀 Sử dụng

4. Biển số hợp lệ sẽ **TỰ ĐỘNG LƯU**

5. Nhấn 'q' để thoát### Script chính (Khuyến nghị):



## 📂 Files```bash

python license_plate_detector.py

````

├── run.bat ← Chạy chương trình (double-click)

├── license_plate_detector.py ← Code chính### Script cũ (chỉ detect, không OCR):

├── plate_detector_best.pt ← Model

├── requirements.txt```bash

├── img/ ← Ảnh tự động lưu tại đâypython test_webcam.py

│ ├── full*30A12345*\*.jpg```

│ └── plate*30A12345*\*.jpg

└── train_model/ ← Dữ liệu training## 📖 Hướng dẫn sử dụng

````

Khi chương trình chạy:

## 🤖 Models

- **Đưa biển số xe** trước camera

1. **Nhận diện**: YOLOv8n-OBB (`plate_detector_best.pt`)- **Nhấn 'q' hoặc ESC**: Thoát chương trình

2. **OCR**: EasyOCR (tự động tải ~150MB lần đầu)- **Nhấn 's'**: Lưu ảnh full frame với kết quả

- **Nhấn 'c'**: Lưu ảnh biển số đã crop

## ⚠️ Lưu ý

## 📊 Thông tin hiển thị

- Dùng Python 3.9 (đã config sẵn)

- Lần đầu chạy tải OCR model- **Số lượng biển số** phát hiện được

- Camera cần cấp quyền- **Ký tự trên biển số** (OCR)

- Ảnh lưu tự động vào `img/`- **Độ tin cậy** (Confidence) %

- **Trạng thái**: HỢP LỆ / KHÔNG RÕ

## 🔧 Fix lỗi OpenCV- **Bounding box** màu xanh quanh biển số



```bash## 🗂️ Cấu trúc dự án

C:\Users\quang\Python39\python.exe -m pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y

C:\Users\quang\Python39\python.exe -m pip install opencv-contrib-python==4.8.1.78```

C:\Users\quang\Python39\python.exe -m pip install "numpy<2.0"test model/

```├── plate_detector_best.pt          # Model YOLOv8 OBB đã train

├── license_plate_detector.py       # Script chính (có OCR)

---├── test_webcam.py                  # Script cũ (không OCR)

**Version 2.1** | 28/10/2025├── requirements.txt                # Danh sách thư viện

├── README.md                       # File này
└── train_model/                    # Dự án train model
    └── runs/
        └── plate_detector_obb5/
            └── weights/
                └── best.pt         # Model gốc
````

## 🔍 Model đã train

- **Loại model**: YOLOv8n-OBB (Oriented Bounding Box)
- **Số classes**: 2 (license_plate)
- **Nguồn**: `train_model/runs/plate_detector_obb5/weights/best.pt`
- **Đã copy ra**: `plate_detector_best.pt`

## 🛠️ Troubleshooting

### Không mở được webcam

- Kiểm tra quyền truy cập camera
- Thử đổi `camera_index=0` thành `1` hoặc `2` trong code

### Lỗi model

- Đảm bảo file `plate_detector_best.pt` tồn tại
- Kiểm tra model trong `train_model/runs/plate_detector_obb5/weights/best.pt`

### Lỗi thư viện

```bash
pip install -r requirements.txt --upgrade
```

### EasyOCR chậm

- Lần đầu chạy cần tải model (~150MB)
- Nếu có GPU NVIDIA: cài `torch` với CUDA support để tăng tốc

### OCR không chính xác

- Điều chỉnh ánh sáng cho biển số rõ hơn
- Đưa biển số gần camera hơn
- Giữ biển số thẳng, không nghiêng quá

## 📝 Lưu ý

- Model OBB hỗ trợ nhận diện đối tượng xoay góc
- Độ tin cậy mặc định: 25% (có thể điều chỉnh trong code)
- Format biển số VN: `30A-12345`, `51F1-12345`, etc.
- Ảnh được lưu với timestamp: `detection_20231028_143045_1.jpg`
- Ảnh crop: `plate_30A12345_20231028_143045.jpg`

## 🎓 Model Information

Model này được train để nhận diện biển số xe máy và ô tô Việt Nam với các đặc điểm:

- Hỗ trợ biển số nghiêng, xoay góc
- Hoạt động trong nhiều điều kiện ánh sáng
- Tốc độ xử lý real-time

---

**Phát triển bởi**: wanghoc - wanghoc.id.vn   
**Lưu ý**: Việc nhận diện ký tự từ biển số hiện đang sử dụng EasyOCR nên hiệu suất nhận diện chưa cao.



