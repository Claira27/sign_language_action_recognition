# Sign Language Action Recognition with Mediapipe + LSTM

Hệ thống nhận diện hành động ngôn ngữ ký hiệu bằng mô hình LSTM sử dụng **landmark tọa độ tay** trích xuất từ thư viện **MediaPipe** của Google. Dữ liệu đầu vào là chuỗi landmark của **right hand** và **left hand** với độ dài cố định 30 frame mỗi video. Mục tiêu là phân loại 21 hành động khác nhau thuộc ngôn ngữ ký hiệu thông dụng.

**Không sử dụng face** và **pose** landmark để giảm nhiễu và trọng số mô hình, vì dữ liệu thu được từ góc camera không chứa các khớp khác trong post và về cơ bản landmark mặt và pose không chủ đích khác biệt( không đóng góp và có thể chiếm trọng số lớn). Nhưng có thể tự tạo dữ liệu theo file collect_data.py bằng việc điều chỉnh biến actions và thay đổi dòng concatenate landmark tùy chỉnh trong hàm extract_landmarks().

## Thư viện sử dụng

- [`mediapipe`](https://google.github.io/mediapipe/) – Trích xuất landmark bàn tay (Hand Landmarks)
- `tensorflow` – Mô hình LSTM và huấn luyện
- `numpy`, `opencv-python` – Xử lý dữ liệu và hình ảnh
- `scikit-learn` – Tiền xử lý và chia tập dữ liệu

## Cấu trúc dữ liệu

- Mỗi action có **100 video**, mỗi video gồm **30 frame**.
- Mỗi frame là 1 mảng `.npy` chứa landmark bàn tay:
  - `left_hand`: 21 điểm × 3 chiều = **63 giá trị**
  - `right_hand`: 21 điểm × 3 chiều = **63 giá trị**
  - → Tổng cộng: **126 giá trị mỗi frame**
- Dữ liệu lưu theo cấu trúc:
data/
hello/
0/
0.npy
...
29.npy
...
99/

## Mô hình huấn luyện
```
Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 30, 64)              │          48,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 30, 128)             │          98,816 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (LSTM)                        │ (None, 64)                  │          49,408 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │           4,160 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 21)                  │             693 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 612,161 (2.34 MB)
 Trainable params: 204,053 (797.08 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 408,108 (1.56 MB)
```
có thể tùy chỉnh lớp trong Sequential()

## File chính
LSTM_train.ipynb	Huấn luyện mô hình
detect.py	Dự đoán hành động từ webcam
collect_data.py	Tạo dữ liệu  
requirements.txt	Danh sách thư viện cần thiết

