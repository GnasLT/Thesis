import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load images
image1 = cv2.imread('28325c.jpeg',cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('28325_3.jpeg',cv2.IMREAD_GRAYSCALE)

# Convert to grayscale
orb = cv2.ORB_create(500,1.5,8,35)
sift = cv2.SIFT_create(500)

# Detect keypoints and compute descriptors for both images
keypoints1 = orb.detect(image1, None)
keypoints2 = orb.detect(image2, None)

keypoints1, descriptors1 =sift.compute(image1,keypoints1)
keypoints2, descriptors2 = sift.compute(image2, keypoints2)
# Create a Brute-Force Matcher
bf = cv2.BFMatcher()

# Match descriptors
matches = bf.knnMatch(descriptors1, descriptors2,k=2)

# Apply ratio test to filter matches
good_matches = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good_matches.append(m)

# Draw matches
matched_image = cv2.drawMatches(image1, keypoints1, 
                        image2, keypoints2, good_matches[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
"""
src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
img_warp = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))

cv2.imshow("Warped image", img_warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
# Display the original image with keypoints marked
plt.figure(figsize = (14, 10))
plt.imshow(matched_image) 
plt.title('Image mathching with Brute-Force method')
plt.show()