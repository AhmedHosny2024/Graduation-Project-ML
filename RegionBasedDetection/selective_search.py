import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.measure import regionprops
import matplotlib.pyplot as plt

def segment_image(image, scale=100, sigma=0.5, min_size=50):
    """
    Segment an input image into regions using Felzenszwalb's method.
    
    Parameters:
    - image (numpy.ndarray): Input RGB image.
    - scale (int): Parameter controlling the segment size.
    - sigma (float): Width of Gaussian kernel for preprocessing.
    - min_size (int): Minimum component size for pruning.
    
    Returns:
    - numpy.ndarray: Segmented image with labeled regions.
    """
    return felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)

def extract_region_features(region, image):
    """
    Extracts features from a given region within the original image.
    
    Parameters:
    - region (skimage.measure._regionprops.RegionProperties): Properties of the region to extract features from.
    - image (numpy.ndarray): Original RGB image.
    
    Returns:
    - numpy.ndarray: Feature vector including color histogram, texture histogram, size, and fill ratio of the region.
    """
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
    """
    Computes a similarity score between two regions based on their feature vectors.
    
    Parameters:
    - region1 (numpy.ndarray): Feature vector of region 1.
    - region2 (numpy.ndarray): Feature vector of region 2.
    
    Returns:
    - float: Similarity score between the two regions.
    """
    return np.sum(np.abs(region1 - region2))

def merge_regions(region1, region2):
    """
    Merges two regions by averaging their feature vectors.
    
    Parameters:
    - region1 (numpy.ndarray): Feature vector of region 1.
    - region2 (numpy.ndarray): Feature vector of region 2.
    
    Returns:
    - numpy.ndarray: Merged feature vector of the two regions.
    """
    return (region1 + region2) / 2

def selective_search(image, scale=100, sigma=0.5, min_size=50, num_iterations=100):
    """
    Performs selective search on an input image to generate region proposals.
    
    Parameters:
    - image (numpy.ndarray): Input RGB image.
    - scale (int): Parameter controlling the segment size in Felzenszwalb's segmentation.
    - sigma (float): Width of Gaussian kernel for preprocessing in Felzenszwalb's segmentation.
    - min_size (int): Minimum component size for pruning in Felzenszwalb's segmentation.
    - num_iterations (int): Number of iterations for greedy region merging.
    
    Returns:
    - numpy.ndarray: Segmented image with labeled regions.
    - list: List of bounding boxes for the merged regions.
    """
    # Step 1: Segment the image into regions
    segmented_image = segment_image(image, scale=scale, sigma=sigma, min_size=min_size)
    regions = regionprops(segmented_image)
    
    # Step 2: Extract features for each region
    region_features = []
    region_bboxes = []
    for region in regions:
        features = extract_region_features(region, image)
        region_features.append(features)
        minr, minc, maxr, maxc = region.bbox
        region_bboxes.append([minc, minr, maxc, maxr])
    
    # Step 3: Perform greedy region merging for num_iterations
    for iteration in range(num_iterations):
        # Step 4: Calculate pairwise similarities between all regions
        region_pairs = [(i, j) for i in range(len(region_features)) for j in range(i+1, len(region_features))]
        similarities = {}
        for (i, j) in region_pairs:
            similarity = compute_similarity(region_features[i], region_features[j])
            similarities[(i, j)] = similarity
        
        # Step 5: Find the most similar pair of regions
        i, j = min(similarities, key=similarities.get)
        
        # Step 6: Merge the two regions into a single larger region
        merged_features = merge_regions(region_features[i], region_features[j])
        
        # Step 7: Update the region features and remove the original regions
        region_features[i] = merged_features
        region_features.pop(j)
        
        # Step 8: Update the bounding boxes of the merged regions
        minc = min(region_bboxes[i][0], region_bboxes[j][0])
        minr = min(region_bboxes[i][1], region_bboxes[j][1])
        maxc = max(region_bboxes[i][2], region_bboxes[j][2])
        maxr = max(region_bboxes[i][3], region_bboxes[j][3])
        region_bboxes[i] = [minc, minr, maxc, maxr]
        region_bboxes.pop(j)
    
    # Step 9: Collect final bounding boxes of the merged regions
    final_bboxes = []
    for bbox in region_bboxes:
        minc, minr, maxc, maxr = bbox
        final_bboxes.append((minc, minr, maxc - minc, maxr - minr))
    
    return segmented_image, final_bboxes

if __name__ == "__main__":
    # Load the image
    image_path = "datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"
    # image_path = "E:/Graduation Project/Graduation-Project-ML/datasets/mimic-cxr-jpg/files/p11/p11002268/s58301648/a57c42a3-a519a3eb-50a43237-c6d2eacb-fbae58b3.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    
    # Perform selective search
    segmented_image_test, regions = selective_search(image)
    
    # Display results
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    print("Number of regions:", len(regions))
