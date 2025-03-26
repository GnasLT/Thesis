import cv2
import numpy as np

def align_images(image1_path, image2_path):
    """
    Căn chỉnh hai hình ảnh có góc chụp gần giống nhau.

    Args:
        image1_path: Đường dẫn đến hình ảnh thứ nhất.
        image2_path: Đường dẫn đến hình ảnh thứ hai.

    Returns:
        Hình ảnh đã được căn chỉnh.
    """

    # 1. Đọc và hiển thị hình ảnh
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Không thể đọc hình ảnh.")
        return None

    # 2. Tìm các điểm đặc trưng
    # Sử dụng ORB (Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 3. Ghép các điểm đặc trưng
    # Sử dụng Brute-Force matcher với Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sắp xếp các matches theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)
    numGoodMatches = int(len(matches) * 0.1)
    matches = matches[:numGoodMatches]

    # Lấy ra các điểm tương ứng
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Tìm ma trận biến đổi hình học sử dụng RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 4. Biến đổi hình ảnh
    h, w = img1.shape
    img1_aligned = cv2.warpPerspective(img1, M, (2*h, w))

    return img1_aligned

if __name__ == '__main__':
    # Thay đổi đường dẫn đến hình ảnh của bạn
    image1_path = "a.jpg"  # Đường dẫn đến hình ảnh thứ nhất
    image2_path = "b.jpg"  # Đường dẫn đến hình ảnh thứ hai

    aligned_image = align_images(image1_path, image2_path)

    if aligned_image is not None:
        # Hiển thị hình ảnh đã được căn chỉnh
        cv2.imshow("Aligned Image", aligned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()