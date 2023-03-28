
import cv2
import numpy as np

from lib_detection import load_model, detect_lp, im2single

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = "test.jpg"

# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Đọc file ảnh đầu vào
Ivehicle = cv2.imread(img_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

output_img = LpImg[0]
output_img *= 255  # or any coefficient
output_img = output_img.astype(np.uint8)
plate = cv2.imwrite('output_test.jpg', output_img)   # ten img_path la gi thi output la the