import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
def apply_kernel(image, kernel):
    """
    Applies a kernel to an image using convolution.
    
    Parameters:
    - image (numpy.ndarray): Input image.
    - kernel (numpy.ndarray): Convolution kernel.
    
    Returns:
    - numpy.ndarray: Convolved image.
    """
    k_height, k_width = kernel.shape
    img_height, img_width = image.shape
    pad_height, pad_width = k_height // 2, k_width // 2
    
    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Initialize the output image
    output = np.zeros_like(image)
    
    # Apply the kernel to each pixel in the image
    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i+k_height, j:j+k_width]
            output[i, j] = np.sum(region * kernel)
    
    return output

def sobel_operator(image):
    """
    Applies Sobel operator to an image to compute gradients in x and y directions.
    
    Parameters:
    - image (numpy.ndarray): Input image.
    
    Returns:
    - gx (numpy.ndarray): Gradient in x direction.
    - gy (numpy.ndarray): Gradient in y direction.
    """
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Apply Sobel kernels to the image
    gx = apply_kernel(image, sobel_x)
    gy = apply_kernel(image, sobel_y)

    return gx, gy

def compute_gradients(image):
    """
    Computes magnitude and orientation of gradients using Sobel operator.
    
    Parameters:
    - image (numpy.ndarray): Input image.
    
    Returns:
    - magnitude (numpy.ndarray): Magnitude of gradients.
    - angle (numpy.ndarray): Orientation of gradients (in degrees).
    """
    # Convert image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(image)
    
    # Compute gradients using Sobel operator
    gx ,gy = sobel_operator(gray)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180 / np.pi

    return magnitude, angle

def calculate_histogram(magnitude, angle, cell_size=(8, 8), bins=9):
    """
    Calculates histograms of gradient orientations for cells in the image.
    
    Parameters:
    - magnitude (numpy.ndarray): Magnitude of gradients.
    - angle (numpy.ndarray): Orientation of gradients (in degrees).
    - cell_size (tuple): Size of each cell (height, width).
    - bins (int): Number of bins in the histogram.
    
    Returns:
    - histogram (numpy.ndarray): Histogram of gradients for each cell.
    """
    cell_rows, cell_cols = cell_size
    angle_bins = np.linspace(0, 180, bins+1)
    
    h, w = magnitude.shape
    num_cells_y = h // cell_rows
    num_cells_x = w // cell_cols
    histogram = np.zeros((num_cells_y, num_cells_x, bins))
    
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            mag_cell = magnitude[i * cell_rows: (i + 1) * cell_rows,
                                 j * cell_cols: (j + 1) * cell_cols]
            ang_cell = angle[i * cell_rows: (i + 1) * cell_rows,
                             j * cell_cols: (j + 1) * cell_cols]
            
            hist, _ = np.histogram(ang_cell, bins=angle_bins, weights=mag_cell)
            histogram[i, j, :] = hist
    
    return histogram

def normalize_histogram(histogram, block_size=(2, 2)):
    """
    Normalizes histograms within each block.
    
    Parameters:
    - histogram (numpy.ndarray): Histogram of gradients for each cell.
    - block_size (tuple): Size of each block (num_cells_y, num_cells_x).
    
    Returns:
    - normalized_histogram (numpy.ndarray): Normalized histogram.
    """
    num_blocks_y = histogram.shape[0] - block_size[0] + 1
    num_blocks_x = histogram.shape[1] - block_size[1] + 1
    if num_blocks_y <= 0 or num_blocks_x <= 0:
        normalized_histogram = np.zeros((histogram.shape[0] * histogram.shape[1] * histogram.shape[2]))
    else:
        normalized_histogram = np.zeros((num_blocks_y, num_blocks_x, block_size[0], block_size[1], histogram.shape[2]))
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            block = histogram[y:y + block_size[0], x:x + block_size[1], :]
            eps = 1e-5
            normalized_block = block / np.sqrt(np.sum(block**2) + eps**2)
            normalized_histogram[y, x, :] = normalized_block
            
    return normalized_histogram.ravel()

def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), bins=9):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.
    
    Parameters:
    - image (numpy.ndarray): Input image.
    - cell_size (tuple): Size of each cell (height, width).
    - block_size (tuple): Size of each block (num_cells_y, num_cells_x).
    - bins (int): Number of bins in the histogram.
    
    Returns:
    - hog_features (numpy.ndarray): HOG feature vector.
    """
    magnitude, angle = compute_gradients(image)
    histogram = calculate_histogram(magnitude, angle, cell_size, bins)
    hog_features = normalize_histogram(histogram, block_size)
    return hog_features

# Example usage:
if __name__ == "__main__":
    image_path = "datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg"
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    print(image.shape)
    # calculate time
    start = time.time()
    hog_features = extract_hog_features(image)
    end= time.time()
    print("Time taken: ",end-start)
    print("HOG feature vector length:", len(hog_features))
    print(hog_features)
     # call built-in HOG descriptor
    # define each block as 4x4 cells of 64x64 pixels each
    cell_size = (8, 8)      # h x w in pixels
    block_size = (2, 2)         # h x w in cells
    nbins = 9  # number of orientation bins 
    # create a HOG object
    hog = cv2.HOGDescriptor()
    h = hog.compute(image)
    print(h)
    print(len(h))