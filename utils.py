import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

def median_blur(image, ksize=3):
    """
    Apply median blur to the image.
    
    Args:
        image: Input image (numpy array).
        ksize: Size of the kernel (must be an odd number).
        
    Returns:
        Blurred image (numpy array).
    """
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    
    # Padding the image to handle the borders
    pad_size = ksize // 2
    if image.ndim == 2:  # Grayscale image
        padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    elif image.ndim == 3:  # Color image
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Unsupported image format")
        
    # Initialize the output image
    blurred_image = np.zeros_like(image)
    
    # Get image dimensions
    height, width = image.shape

    # Apply the median filter
    for y in range(height):
        for x in range(width):
            # Extract the current window
            window = padded_image[y:y+ksize, x:x+ksize]
            
            # Compute the median of the window
            median_value = np.median(window)
            
            # Assign the median value to the output image
            blurred_image[y, x] = median_value

    return blurred_image

def gaussian_kernel(size, sigma):
    """
    Generate a Gaussian kernel.
    
    Args:
        size: Size of the kernel (must be odd).
        sigma: Standard deviation of the Gaussian distribution.
        
    Returns:
        Gaussian kernel (numpy array).
    """
    k = (size - 1) // 2
    x, y = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def apply_convolution(image, kernel):
    """
    Apply convolution to an image with a given kernel.
    
    Args:
        image: Input image (numpy array).
        kernel: Convolution kernel (numpy array).
        
    Returns:
        Convolved image (numpy array).
    """
    pad_size = kernel.shape[0] // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    height, width = image.shape
    convolved_image = np.zeros_like(image)
    
    for y in range(height):
        for x in range(width):
            window = padded_image[y:y+kernel.shape[0], x:x+kernel.shape[1]]
            convolved_image[y, x] = np.sum(window * kernel)
    
    return convolved_image

# bad implementation of non-local means denoising
# def non_local_means_denoising(image, h=30, patch_size=7, window_size=21):
#     """
#     Apply Non-Local Means Denoising to the image.
    
#     Args:
#         image: Input noisy image (numpy array).
#         h: Filter strength.
#         patch_size: Size of the patch.
#         window_size: Size of the window for searching similar patches.
        
#     Returns:
#         Denoised image (numpy array).
#     """
#     pad_size = patch_size // 2
#     padded_image = np.pad(image, pad_size, mode='reflect')
#     height, width = image.shape
#     denoised_image = np.zeros_like(image)
    
#     # Precompute Gaussian weights
#     gaussian_weights = gaussian_kernel(patch_size, patch_size / 6.4)
    
#     # Extract patches
#     patches = view_as_windows(padded_image, (patch_size, patch_size))
#     patches = patches.reshape(-1, patch_size, patch_size)
    
#     # Vectorize patch comparison
#     for y in range(height):
#         for x in range(width):
#             patch = patches[y * width + x]
#             patch_weights = np.exp(-np.sum((patches - patch) ** 2 * gaussian_weights, axis=(1, 2)) / h ** 2)
#             patch_weights = patch_weights.reshape(height, width)
            
#             # Compute the weighted average of the patches
#             denoised_image[y, x] = np.sum(patch_weights * image) / np.sum(patch_weights)
    
#     return denoised_image

def non_local_means_denoising(image, h=30, patch_size=7, window_size=21):
    
    """
    Apply Non-Local Means Denoising to the image using scikit-image.
    
    Args:
        image: Input noisy image (numpy array).
        h: Filter strength.
        patch_size: Size of the patch.
        window_size: Size of the window for searching similar patches.
        
    Returns:
        Denoised image (numpy array).
    """
    # Estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(image))
    
    # Apply Non-Local Means Denoising
    denoised_image = denoise_nl_means(image, h=h, patch_size=patch_size, patch_distance=window_size//2, 
                                      fast_mode=True, sigma=sigma_est)
    
    return denoised_image


def inpaint(image, mask, radius=3):
    """
    Inpaint the masked region of the image using a simple algorithm.
    
    Args:
        image: Input grayscale image (numpy array).
        mask: Binary mask indicating the region to be inpainted (numpy array).
        radius: Radius of the neighborhood to consider for inpainting.
        
    Returns:
        Inpainted image (numpy array).
    """
    # Convert image to float for precise calculations
    image = image.astype(np.float32)
    
    # Create a copy of the image to store the inpainted result
    inpainted_image = image.copy()
    
    # Get the coordinates of the masked region
    mask_coords = np.argwhere(mask)
    
    # Iterate until the masked region is filled
    while len(mask_coords) > 0:
        new_mask_coords = []
        
        for y, x in mask_coords:
            # Get the neighborhood of the current pixel
            y_min = max(0, y - radius)
            y_max = min(image.shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(image.shape[1], x + radius + 1)
            
            # Extract the neighborhood
            neighborhood = inpainted_image[y_min:y_max, x_min:x_max]
            neighborhood_mask = mask[y_min:y_max, x_min:x_max]
            
            # Compute the average value of the non-masked pixels in the neighborhood
            non_masked_values = neighborhood[neighborhood_mask == 0]
            if len(non_masked_values) > 0:
                inpainted_image[y, x] = np.mean(non_masked_values)
                mask[y, x] = 0
            else:
                new_mask_coords.append((y, x))
        
        mask_coords = new_mask_coords
    
    return inpainted_image.astype(np.uint8)

