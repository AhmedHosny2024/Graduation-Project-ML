import numpy as np
import cv2

def compute_integral_image(image):
    return np.cumsum(np.cumsum(image, axis=0), axis=1)

def compute_haar_feature(integral_image, feature_type, position, size):
    x, y = position
    w, h = size
    
    if feature_type == 'two-rectangle':
        mid_x = x + w // 2
        white = integral_image[y+h, mid_x] - integral_image[y, mid_x] - integral_image[y+h, x] + integral_image[y, x]
        black = integral_image[y+h, x+w] - integral_image[y, x+w] - integral_image[y+h, mid_x] + integral_image[y, mid_x]
        return white - black
    
    elif feature_type == 'three-rectangle':
        third_x = x + w // 3
        two_third_x = x + 2 * (w // 3)
        white1 = integral_image[y+h, third_x] - integral_image[y, third_x] - integral_image[y+h, x] + integral_image[y, x]
        black = integral_image[y+h, two_third_x] - integral_image[y, two_third_x] - integral_image[y+h, third_x] + integral_image[y, third_x]
        white2 = integral_image[y+h, x+w] - integral_image[y, x+w] - integral_image[y+h, two_third_x] + integral_image[y, two_third_x]
        return white1 - black + white2
    
    elif feature_type == 'four-rectangle':
        mid_x = x + w // 2
        mid_y = y + h // 2
        white1 = integral_image[mid_y, mid_x] - integral_image[y, mid_x] - integral_image[mid_y, x] + integral_image[y, x]
        black1 = integral_image[mid_y, x+w] - integral_image[y, x+w] - integral_image[mid_y, mid_x] + integral_image[y, mid_x]
        black2 = integral_image[y+h, mid_x] - integral_image[mid_y, mid_x] - integral_image[y+h, x] + integral_image[mid_y, x]
        white2 = integral_image[y+h, x+w] - integral_image[mid_y, x+w] - integral_image[y+h, mid_x] + integral_image[mid_y, mid_x]
        return white1 - black1 - black2 + white2
    
    elif feature_type == 'edge-horizontal':
        mid_y = y + h // 2
        white = integral_image[mid_y, x+w] - integral_image[mid_y, x] - integral_image[y, x+w] + integral_image[y, x]
        black = integral_image[y+h, x+w] - integral_image[y+h, x] - integral_image[mid_y, x+w] + integral_image[mid_y, x]
        return white - black
    
    elif feature_type == 'edge-vertical':
        mid_x = x + w // 2
        white = integral_image[y+h, mid_x] - integral_image[y, mid_x] - integral_image[y+h, x] + integral_image[y, x]
        black = integral_image[y+h, x+w] - integral_image[y, x+w] - integral_image[y+h, mid_x] + integral_image[y, mid_x]
        return white - black
    
    return 0

def extract_haar_features(image, window_size=(24, 24)):
    integral_image = compute_integral_image(image)
    integral_image = integral_image / (integral_image.max() - integral_image.min())  # Normalize integral image
    feature_types = ['two-rectangle', 'three-rectangle', 'four-rectangle', 'edge-horizontal', 'edge-vertical']
    features = []
    h, w = image.shape
    
    for y in range(0, h - window_size[1] + 1, window_size[1]):
        for x in range(0, w - window_size[0] + 1, window_size[0]):
            for feature_type in feature_types:
                feature_value = compute_haar_feature(integral_image, feature_type, (x, y), window_size)
                features.append(feature_value)
    
    return features

def extract_haar_features_3d(image, window_size=(24, 24)):
    features = []
    for channel in range(image.shape[2]):
        channel_features = extract_haar_features(image[:, :, channel], window_size)
        features.extend(channel_features)
    return features

if __name__ == "__main__":
    # Load X-ray image (3D)
    image_path = "datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)  # Example 3D image with 3 slices (RGB channels)
    print(image.shape)
    # Extract Haar features
    haar_features = extract_haar_features_3d(image, window_size=(24, 24))
    print(len(haar_features))
