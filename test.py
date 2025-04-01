import cv2
from matplotlib import pyplot as plt
import numpy as np

def filter_matches_with_mask(matches, mask):
    filtered_matches = []
    for i, m in enumerate(matches):
        if mask[i] == 1:  # mask là mảng NumPy, giá trị True được biểu diễn bằng 1
            filtered_matches.append(m)
    return filtered_matches

img_nir = cv2.imread('28325b.jpeg',cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.imread('28325_2.jpeg',cv2.COLOR_BGR2GRAY) 

height, width, channels = img_nir.shape
img_rgb = cv2.resize(img_rgb,(width,height),cv2.INTER_CUBIC)

orb = cv2.ORB_create(500,1.6,10,33)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img_nir,None)
kp2, des2 = orb.detectAndCompute(img_rgb,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance) 

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

filtered_matches = filter_matches_with_mask(matches, mask)
print(f"Số lượng matches ban đầu: {len(matches)}")
print(f"Số lượng matches sau khi lọc: {len(filtered_matches)}")
    

# Lấy kích thước ảnh đầu tiên
height, width = img_rgb.shape[:2]
img_rgb_aligned = cv2.warpPerspective(img_nir, H, (width, height))

# Hiển thị ảnh
#cv2.imshow("Image 2 Aligned", img_rgb_aligned)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

img3 = cv2.drawMatches(img_nir,kp1,img_rgb,kp2,filtered_matches[:],None, flags=2)
plt.imshow(img3),plt.show()