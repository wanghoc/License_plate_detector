"""
Hệ thống nhận diện biển số xe hoàn chỉnh
- Nhận diện biển số qua YOLOv8 OBB
- Đọc ký tự qua EasyOCR
"""
import cv2
from ultralytics import YOLO
import easyocr
import re
import numpy as np
from datetime import datetime

class LicensePlateDetector:
    def __init__(self, plate_model_path='plate_detector_best.pt'):
        """
        Khởi tạo hệ thống nhận diện biển số
        Args:
            plate_model_path: Đường dẫn đến model YOLOv8 OBB đã train
        """
        print("=" * 50)
        print("ĐANG KHỞI TẠO HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE")
        print("=" * 50)
        
        # Load model nhận diện biển số
        print("\n[1/2] Đang tải model nhận diện biển số...")
        try:
            self.plate_model = YOLO(plate_model_path)
            print(f"✓ Tải model thành công: {plate_model_path}")
        except Exception as e:
            print(f"✗ Lỗi khi tải model: {e}")
            raise
        
        # Khởi tạo OCR reader
        print("\n[2/2] Đang khởi tạo OCR reader...")
        print("    (Lần đầu chạy sẽ tải model OCR, có thể mất vài phút...)")
        try:
            self.ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)
            print("✓ OCR reader sẵn sàng!")
        except Exception as e:
            print(f"✗ Lỗi khi khởi tạo OCR: {e}")
            raise
        
        print("\n" + "=" * 50)
        print("HỆ THỐNG ĐÃ SẴN SÀNG!")
        print("=" * 50 + "\n")
        
        # Biến lưu lịch sử
        self.plate_history = []
    
    def validate_plate_format(self, text):
        """
        Kiểm tra format biển số Việt Nam
        Format: 30A-12345, 51F1-12345, 29A12345, etc.
        """
        text = text.upper().replace(' ', '').replace('-', '')
        # Pattern cho biển số VN: 2 số + 1-2 chữ + 4-5 số
        pattern = r'\d{2}[A-Z]{1,2}\d{4,5}'
        match = re.search(pattern, text)
        return match is not None, text
    
    def format_plate_text(self, text):
        """Format lại text biển số cho đẹp"""
        text = text.upper().replace(' ', '').replace('-', '')
        # Tìm pattern: 2 số + chữ + số
        match = re.search(r'(\d{2})([A-Z]{1,2})(\d{4,5})', text)
        if match:
            return f"{match.group(1)}{match.group(2)}-{match.group(3)}"
        return text
    
    def process_frame(self, frame, conf_threshold=0.3):
        """
        Xử lý 1 frame: detect biển số và OCR
        Args:
            frame: Frame ảnh từ webcam
            conf_threshold: Ngưỡng confidence cho detection
        Returns:
            annotated_frame: Frame đã vẽ kết quả
            plates_info: Danh sách thông tin biển số phát hiện
        """
        annotated_frame = frame.copy()
        plates_info = []
        
        # Detect biển số
        results = self.plate_model(frame, conf=conf_threshold, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Xử lý OBB (Oriented Bounding Box)
            if result.obb is not None and len(result.obb) > 0:
                for idx, obb in enumerate(result.obb):
                    try:
                        # Lấy confidence
                        conf = float(obb.conf[0]) if obb.conf is not None else 0.0
                        
                        # Lấy class
                        cls = int(obb.cls[0]) if obb.cls is not None else 0
                        class_name = self.plate_model.names[cls] if cls < len(self.plate_model.names) else "plate"
                        
                        # Lấy tọa độ OBB (4 điểm)
                        points = obb.xyxyxyxy[0].cpu().numpy().astype(int)
                        
                        # Tính bounding box thường
                        x_coords = points[:, 0]
                        y_coords = points[:, 1]
                        x1, x2 = x_coords.min(), x_coords.max()
                        y1, y2 = y_coords.min(), y_coords.max()
                        
                        # Đảm bảo trong giới hạn frame
                        h, w = frame.shape[:2]
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        
                        # Crop biển số
                        if x2 > x1 and y2 > y1:
                            cropped_plate = frame[y1:y2, x1:x2]
                            
                            # Tiền xử lý ảnh cho OCR
                            gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                            # Tăng độ tương phản
                            gray = cv2.equalizeHist(gray)
                            # Resize nếu ảnh quá nhỏ
                            if gray.shape[1] < 200:
                                scale = 200 / gray.shape[1]
                                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                            
                            # OCR
                            ocr_results = self.ocr_reader.readtext(gray, detail=0)
                            plate_text_raw = ''.join(ocr_results).replace(' ', '')
                            
                            # Validate và format
                            is_valid, plate_text = self.validate_plate_format(plate_text_raw)
                            plate_text_formatted = self.format_plate_text(plate_text) if is_valid else plate_text_raw
                            
                            # Vẽ OBB polygon
                            cv2.polylines(annotated_frame, [points], True, (0, 255, 0), 2)
                            
                            # Vẽ background cho text
                            label = f"{class_name}: {conf*100:.1f}%"
                            if plate_text_formatted:
                                label += f" | {plate_text_formatted}"
                            
                            # Tính kích thước text
                            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            
                            # Vẽ background
                            cv2.rectangle(annotated_frame, (x1, y1 - text_h - 10), 
                                        (x1 + text_w + 10, y1), (0, 255, 0), -1)
                            
                            # Vẽ text
                            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                            
                            # Lưu thông tin
                            plate_info = {
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'class': class_name,
                                'text': plate_text_formatted,
                                'raw_text': plate_text_raw,
                                'is_valid': is_valid,
                                'cropped': cropped_plate,
                                'frame': frame.copy()  # Lưu frame gốc
                            }
                            plates_info.append(plate_info)
                            
                    except Exception as e:
                        print(f"Lỗi xử lý OBB {idx}: {e}")
                        continue
        
        return annotated_frame, plates_info
    
    def run_webcam(self, camera_index=0, conf_threshold=0.3):
        """
        Chạy detection qua webcam
        Args:
            camera_index: Index của camera (0 = camera mặc định)
            conf_threshold: Ngưỡng confidence
        """
        print("\n" + "=" * 50)
        print("BẮT ĐẦU NHẬN DIỆN QUA WEBCAM")
        print("=" * 50)
        print(f"Confidence threshold: {conf_threshold}")
        print("\nHƯỚNG DẪN:")
        print("  - Đưa biển số xe trước camera")
        print("  - Biển số HỢP LỆ sẽ TỰ ĐỘNG LƯU vào thư mục img/")
        print("  - Nhấn 'q' hoặc ESC: Thoát")
        print("=" * 50 + "\n")
        
        # Mở webcam
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("✗ Không thể mở webcam!")
            print("Đang thử lại...")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print("✗ Vẫn không mở được webcam!")
                return
        
        print("✓ Webcam đã mở!")
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Đọc frame test
        ret, test_frame = cap.read()
        if not ret:
            print("✗ Không thể đọc frame!")
            cap.release()
            return
        
        print(f"✓ Kích thước frame: {test_frame.shape[1]}x{test_frame.shape[0]}")
        print("✓ Bắt đầu nhận diện...\n")
        
        # Tạo window
        window_name = 'Nhan dien bien so xe - License Plate Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        saved_plates = set()  # Lưu các biển số đã chụp
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("✗ Không đọc được frame!")
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, plates_info = self.process_frame(frame, conf_threshold)
            
            # Hiển thị info
            info_y = 30
            
            if len(plates_info) > 0:
                # Header
                cv2.putText(annotated_frame, f"Phat hien: {len(plates_info)} bien so", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                info_y += 35
                
                # Chi tiết từng biển
                for i, plate in enumerate(plates_info):
                    status = "HOP LE" if plate['is_valid'] else "KHONG RO"
                    color = (0, 255, 0) if plate['is_valid'] else (0, 165, 255)
                    
                    text = f"  {i+1}. Bien so: {plate['text']} ({plate['confidence']*100:.1f}%) - {status}"
                    cv2.putText(annotated_frame, text, 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    info_y += 28
                    
                    # Print console (mỗi 30 frames)
                    if frame_count % 30 == 0:
                        print(f"[Frame {frame_count}] Phát hiện: {plate['text']} - "
                              f"Conf: {plate['confidence']*100:.1f}% - {status}")
                    
                    # Tự động lưu ảnh nếu biển số hợp lệ và chưa lưu
                    if plate['is_valid'] and plate['text'] not in saved_plates:
                        saved_plates.add(plate['text'])
                        
                        # Lưu ảnh full frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        plate_clean = plate['text'].replace('-', '').replace(' ', '')
                        
                        # Lưu full frame với annotation vào img/img_full
                        full_filename = f'img/img_full/{plate_clean}_{timestamp}.jpg'
                        cv2.imwrite(full_filename, annotated_frame)
                        
                        # Lưu ảnh biển số crop vào img/img_crop
                        crop_filename = f'img/img_crop/{plate_clean}_{timestamp}.jpg'
                        cv2.imwrite(crop_filename, plate['cropped'])
                        
                        print(f"\n✓✓✓ TỰ ĐỘNG LƯU BIỂN SỐ MỚI: {plate['text']}")
                        print(f"    - Full: {full_filename}")
                        print(f"    - Crop: {crop_filename}\n")
                    
            else:
                cv2.putText(annotated_frame, "Khong phat hien bien so", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Hướng dẫn
            help_text = "Nhan 'q'=Thoat | Bien so HOP LE se tu dong luu vao thu muc img/"
            cv2.putText(annotated_frame, help_text, 
                       (10, annotated_frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Hiển thị
            cv2.imshow(window_name, annotated_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Thoát
                print("\nĐang thoát...")
                break
        
        # Dọn dẹp
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Đã đóng webcam!")
        print("=" * 50)

def main():
    """Hàm main"""
    try:
        # Khởi tạo detector
        detector = LicensePlateDetector('plate_detector_best.pt')
        
        # Chạy webcam
        detector.run_webcam(camera_index=0, conf_threshold=0.25)
        
    except KeyboardInterrupt:
        print("\n\n✓ Đã dừng bởi người dùng!")
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
