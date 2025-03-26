import cv2
import numpy as np

def calculate_ndvi_from_keypoints(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    if img1 is None or img2 is None:
        raise ValueError("Could not open or find the images.")

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # BFMatcher with hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    matched_keypoints1 = []
    matched_keypoints2 = []
    for match in matches:
        matched_keypoints1.append(kp1[match.queryIdx].pt)
        matched_keypoints2.append(kp2[match.trainIdx].pt)

    # Calculate NDVI at matched keypoints
    ndvi_values = []
    for pt1, pt2 in zip(matched_keypoints1, matched_keypoints2):
        x, y = map(int, pt1)  # Convert to integers for pixel access.
        try:
            red = img1[y, x] #access the pixel value
            nir = img2[y, x]
            if red + nir == 0: # handle division by zero
                ndvi = 0.0
            else:
                ndvi = (nir - red) / (nir + red)
            ndvi_values.append(ndvi)
        except IndexError:
            # Handle cases where keypoints are outside the image boundaries
            print(f"Keypoint ({x}, {y}) is out of image bounds. Skipping.")
            continue

    return ndvi_values, matched_keypoints1, matched_keypoints2

# Example usage:
# Replace with your image paths
image1_path = "red_band.tif"  # Example: red band image
image2_path = "nir_band.tif"  # Example: near-infrared band image

try:
    ndvi_values, kp1, kp2 = calculate_ndvi_from_keypoints(image1_path, image2_path)

    if ndvi_values:
        print("NDVI values:", ndvi_values)
        print("Number of matched keypoints:", len(ndvi_values))
        #print("Matched keypoints 1:", kp1)
        #print("Matched keypoints 2:", kp2)

        # Optionally, you can visualize the matches
        # img1_color = cv2.imread(image1_path)
        # img2_color = cv2.imread(image2_path)
        # matches_to_draw = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kp1))] #create dummy matches to draw lines.
        # img_matches = cv2.drawMatches(img1_color, [cv2.KeyPoint(x=x, y=y, _size=1) for x, y in kp1], img2_color,[cv2.KeyPoint(x=x, y=y, _size=1) for x,y in kp2], matches_to_draw, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow("Matches", img_matches)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    else:
        print("No matched keypoints found.")

except ValueError as e:
    print(f"Error: {e}")
except FileNotFoundError:
    print("Error: One or both image files not found.")