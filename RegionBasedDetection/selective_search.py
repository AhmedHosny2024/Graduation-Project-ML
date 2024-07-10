import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern
from skimage.color import rgb2lab, rgb2gray
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def segment_image(image, scale=100, sigma=0.5, min_size=50):
    return felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)

def extract_region_features(region, image):
    minr, minc, maxr, maxc = region.bbox
    region_image = image[minr:maxr, minc:maxc]
    
    color_hist = np.histogram(region_image, bins=25, range=(0, 256))[0]
    color_hist = color_hist / color_hist.sum()  # Normalize
    
    texture = local_binary_pattern(rgb2gray(region_image), P=8, R=1, method='uniform')
    texture_hist = np.histogram(texture, bins=25, range=(0, 25))[0]
    texture_hist = texture_hist / texture_hist.sum()  # Normalize
    
    size = region.area
    fill = region.filled_area
    
    return np.concatenate([color_hist, texture_hist, [size, fill]])

def compute_similarity(region1, region2):
    return np.sum(np.abs(region1 - region2))

def merge_regions(region1, region2):
    return (region1 + region2) / 2

def selective_search(image, scale=100, sigma=0.5, min_size=50):
    segmented_image = segment_image(image, scale=scale, sigma=sigma, min_size=min_size)
    regions = regionprops(segmented_image)
    
    region_features = []
    for region in regions:
        features = extract_region_features(region, image)
        region_features.append(features)
    
    region_pairs = [(i, j) for i in range(len(region_features)) for j in range(i+1, len(region_features))]
    
    similarities = {}
    for (i, j) in region_pairs:
        similarity = compute_similarity(region_features[i], region_features[j])
        similarities[(i, j)] = similarity
    
    while len(similarities) > 0:
        i, j = min(similarities, key=similarities.get)
        
        if j >= len(region_features):
            similarities.pop((i, j))
            continue
        
        region_features[i] = merge_regions(region_features[i], region_features[j])
        region_features.pop(j)
        
        # Remove outdated similarities
        similarities = {(a, b): sim for (a, b), sim in similarities.items() if b != j and a != j}
        
        # Update similarities for the merged region
        new_similarities = {}
        for k in range(len(region_features)):
            if k != i:
                similarity = compute_similarity(region_features[i], region_features[k])
                new_similarities[(min(i, k), max(i, k))] = similarity
        
        similarities.update(new_similarities)
    
    # Collect bounding boxes of the regions
    regions_bboxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        regions_bboxes.append((minc, minr, maxc - minc, maxr - minr))
    
    return segmented_image, regions_bboxes

if __name__ == "__main__":
    # Load the image
    image_path = "datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    
    # Perform selective search
    segmented_image_test, regions = selective_search_ss(image, min_size=1000)
    
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(segmented_image_test)
    plt.axis('off')
    plt.show()

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    print("Number of regions:", len(regions))
    import selectivesearch as ss
    image=cv2.imread("datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg")
    segmented_image, regions = ss.selective_search(image, min_size=1000,scale=100, sigma=0.5)
    print("Number of regions:", len(regions))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()