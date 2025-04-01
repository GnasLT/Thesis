import cv2
from matplotlib import pyplot as plt
import numpy as np

def filter_matches_with_mask(matches, mask):
    filtered_matches = []
    for i, m in enumerate(matches):
        if mask[i] == 1:  # mask là mảng NumPy, giá trị True được biểu diễn bằng 1
            filtered_matches.append(m)
    return filtered_matches
def split_red(rgb):
    _,_,r = cv2.split(rgb)
    return r

def align_image(img_nir,img_rgb): #sift
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_nir, None)
    kp2, des2 = sift.detectAndCompute(img_rgb, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    print(f"Số lượng matches ban đầu: {len(good_matches)}")
    # Lấy tọa độ các điểm tương ứng
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # Tìm ma trận homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    filtered_matches = filter_matches_with_mask(good_matches, mask)
    print(f"Số lượng matches sau khi lọc: {len(filtered_matches)}")
    img_warp = cv2.warpPerspective(img_nir, H, (width,height))
    img3 = cv2.drawMatches(img_nir,kp1,img_rgb,kp2,filtered_matches[:],None, flags=2)
    return img_warp

nir_path ='28325b.jpeg'
rgb_path ='28325_2.jpeg'

img_nir = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED) 
img_rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED) 

#resize hinh anh bang pp noi suy cubic

height, width, channels = img_nir.shape # h = 2464, w = 3280

img_rgb = cv2.resize(img_rgb,(width,height),cv2.INTER_CUBIC)
image_align = align_image(img_nir, img_rgb)
cv2.imwrite('28325nir.jpeg',image_align)

#nir = cv2.imread('28325nir.jpeg',cv2.COLOR_BGR2GRAY)
red = split_red(img_rgb)
if len(image_align.shape) == 3: # nếu nir là ảnh màu chuyển về ảnh xám.
    b,g,nir = cv2.split(image_align)
print(red.shape)
print(nir.shape)

np.seterr(divide='ignore', invalid='ignore')

ndvi = (nir.astype(float) - red.astype(float) ) / (nir + red+1e-8)
ndvi_normalized = ((ndvi + 1) / 2) * 255  # Chuyển từ -1,1 sang 0,255
ndvi_normalized = ndvi_normalized.astype(np.uint8)
ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
print(ndvi[1618][1477])
ndvi = np.random.uniform(-1, 1, size=(2464, 3280))
plt.imshow(ndvi_colored)
#plt.imshow(nir),
plt.show()