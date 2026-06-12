import cv2

from skimage.feature import hog

def extract_hog_features(image_path):
    
    img=cv2.imread(image_path)

    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    features= hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2,2),
        block_norm="L2-Hys"
    )

    return features