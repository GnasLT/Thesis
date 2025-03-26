import cv2
import numpy as np

def align_images(img1, img2):
    # 1. Phát hiện các điểm đặc trưng
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 2. Ghép các điểm đặc trưng
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 3. Ước tính phép biến đổi hình học
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 4. Biến đổi ảnh
    h, w = img1.shape[:2]
    img2_aligned = cv2.warpPerspective(img2, M, (w, h))

    return img2_aligned

# Đọc ảnh
img1 = cv2.imread('r.jpg')
img2 = cv2.imread('l.jpg')

# Căn chỉnh ảnh
img_aligned = align_images(img1, img2)

# Hiển thị ảnh đã căn chỉnh
cv2.imshow("Aligned Image", img_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()