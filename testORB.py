import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb_example(image_path1, image_path2):
    # Load images
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    orb = cv2.ORB_create(500,1.3,10,31)

    # Tìm keypoints và descriptors với ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Đối sánh đặc trưng
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sắp xếp các matches theo khoảng cách
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw top matches
    print(len(matches))
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result using matplotlib
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title("ORB Feature Matching")
    plt.show()

# Example usage (replace with your image paths)
image_path1 = '28325b.jpeg'
image_path2 = '28325_2.jpeg'

orb_example(image_path1, image_path2)