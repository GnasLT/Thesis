import cv2
import numpy as np

def combine_images(img1_path, img2_path):
    # Đọc hình ảnh
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Khởi tạo ORB
    orb = cv2.ORB_create()

    # Tìm keypoints và descriptors với ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Đối sánh đặc trưng
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sắp xếp các matches theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)

    # Lọc ngoại lệ bằng RANSAC
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        img2_warp = cv2.warpPerspective(img1, M, (img2.shape[1] + img1.shape[1], img2.shape[0]))
        img2_warp[0:img2.shape[0], 0:img2.shape[1]] = img2
        img2_warp = cv2.resize(img2_warp,(700,500))
        return img2_warp
    else:
        print("Không đủ matches để thực hiện RANSAC.")
        return None

# Sử dụng hàm
combined_img = combine_images("28325c.jpeg", "28325_3.jpeg")

if combined_img is not None:
    cv2.imshow("Combined Image", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()