import cv2
import numpy as np

# Đọc hai hình ảnh
img1 = cv2.imread('ref.jpg', 0)
img2 = cv2.imread('align.jpg', 0)

# Phát hiện điểm đặc trưng và đối sánh
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Ước tính ma trận biến đổi
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Biến đổi hình ảnh
height, width = img2.shape
img1_aligned = cv2.warpPerspective(img1, M, (width, height))

# Kết hợp hình ảnh
result = cv2.addWeighted(img1_aligned, 0.5, img2, 0.5, 0)

# Hiển thị kết quả
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()