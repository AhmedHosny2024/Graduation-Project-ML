import numpy as np
from FeatureExtraction.hog import extract_hog_features
from FeatureExtraction.haar import extract_haar_features

def extract_features(img,type):
    img = np.array(img)
    if type == 'hog':
        features = extract_hog_features(img)
        return features.flatten()
    else:
        features = extract_haar_features(img)
        return features
