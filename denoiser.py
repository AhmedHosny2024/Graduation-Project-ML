import cv2
from generate_noise import *
from utils import *
def remove_block_pixel_noise(image):
    """
    Remove Block-Pixel noise using median filtering.
    
    Args:
        image: Input noisy image (numpy array).
    
    Returns:
        Denoised image (numpy array).
    """
    denoised_image = median_blur(image, ksize=3)
    return denoised_image

def remove_convolve_noise(image, sigma=1.0):
    """
    Remove Convolve-Noise using Gaussian filtering and Non-Local Means Denoising.
    
    Args:
        image: Input noisy image (numpy array).
        sigma: Standard deviation of the Gaussian kernel used in the noise addition.
    
    Returns:
        Denoised image (numpy array).
    """
    # Step 1: Gaussian filtering to reduce blur
    kernel_size = int(2 * sigma + 1)
    gaussian_kernel_ = gaussian_kernel(kernel_size, sigma)
    gaussian_filtered_image = apply_convolution(image, gaussian_kernel_)

    # Step 2: Non-Local Means Denoising to remove Gaussian noise
    # denoised_image = cv2.fastNlMeansDenoising(gaussian_filtered_image, None, 30, 7, 21)
    
    # this is the implementation of the above line from scratch
    denoised_image = non_local_means_denoising(gaussian_filtered_image, h=30, patch_size=7, window_size=21)

    return denoised_image




def remove_keep_patch_noise(image):
    """
    Remove Keep-Patch noise using image inpainting.
    
    Args:
        image: Input noisy image (numpy array).
    
    Returns:
        Denoised image (numpy array).
    """
    mask = (image == 0).astype(np.uint8)  # Create mask where pixels are zero
    inpainted_image = inpaint(image, mask)
    return inpainted_image

def remove_extract_patch_noise(noisy_image):
    """
    Remove Extract-Patch noise using image inpainting.
    
    Args:
        noisy_image: Input noisy image (numpy array).
    
    Returns:
        Denoised image (numpy array).
    """
    # Create a mask where the non-zero pixels are the patch
    mask = (noisy_image == 0).astype(np.uint8)
    
    # Inpaint the image using the mask
    inpainted_image = cv2.inpaint(noisy_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return inpainted_image

def remove_pad_rotate_project_noise(noisy_image, rotation_angle):
    """
    Remove Pad-Rotate-Project noise from an image by reversing the rotation.
    
    Args:
        noisy_image: Input noisy image (numpy array).
        rotation_angle: The rotation angle used during noise addition.
    
    Returns:
        Denoised image (numpy array).
    """
    # Get image dimensions
    h, w = noisy_image.shape[:2]
    
    # Calculate padding size (assuming padding was symmetric)
    padding = max(h, w) // 2
    
    # Pad the noisy image again to ensure no cropping occurs during inverse rotation
    padded_noisy_image = cv2.copyMakeBorder(noisy_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Get inverse rotation matrix
    inverse_rotation_matrix = cv2.getRotationMatrix2D((padded_noisy_image.shape[1]//2, padded_noisy_image.shape[0]//2), -rotation_angle, 1)
    
    # Perform inverse rotation
    unrotated_image = cv2.warpAffine(padded_noisy_image, inverse_rotation_matrix, (padded_noisy_image.shape[1], padded_noisy_image.shape[0]))
    
    # Crop the image to remove padding
    denoised_image = unrotated_image[padding:padding+h, padding:padding+w]
    
    return denoised_image

def remove_line_strip_noise(image):
    """
    Remove Line-Strip noise using image inpainting.
    
    Args:
        image: Input noisy image (numpy array).
    
    Returns:
        Denoised image (numpy array).
    """
    mask = (image == 0.5).astype(np.uint8)  # Create mask where strips are added
    # inpainted_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    inpainted_image = inpaint(image, mask, radius=3)
    return inpainted_image

def remove_salt_and_pepper_noise(image):
    """
    Remove salt and pepper noise using median filtering.
    
    Args:
        image: Input noisy image (numpy array).
    
    Returns:
        Denoised image (numpy array).
    """
    denoised_image = median_blur(image, 3)
    return denoised_image

def remove_gaussian_projection_noise(image):
    """
    Remove Gaussian-Projection noise using Non-Local Means Denoising.
    
    Args:
        image: Input noisy image (numpy array).
    
    Returns:
        Denoised image (numpy array).
    """
    # denoised_image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    denoised_image = non_local_means_denoising(image, h=30, patch_size=7, window_size=21)
    return denoised_image

 
def denoise_image(image, method):
    if method == 'block-pixel':
        # very good solution
        return remove_block_pixel_noise(image)
    elif method == 'convolve':
        # sadly it make the image plurry
        return remove_convolve_noise(image)
    elif method == 'keep-patch':
        # maybe good
        return remove_keep_patch_noise(image)
    elif method == 'extract-patch':
        # maybe good
        return remove_extract_patch_noise(image)
    elif method == 'pad-rotate-project':
        # sadly it can't detect the rotation angle
        return remove_pad_rotate_project_noise(image, rotation_angle=2)
    elif method == 'line-strip':
        # excellent solution
        return remove_line_strip_noise(image)
    elif method == 'salt-and-pepper':
        # very good solution
        return remove_salt_and_pepper_noise(image)
    elif method == 'gaussian':
        # sadly it make the image plurry
        return remove_gaussian_projection_noise(image)
    elif method == 'no-noise':
        return image
    else:
        raise ValueError("Invalid method: {}".format(method) + "Choose from 'block-pixel', 'convolve', 'keep-patch', 'extract-patch', 'pad-rotate-project', 'line-strip', 'salt-and-pepper', 'gaussian'")
    

if __name__ == "__main__":
  # save all noise functions in dictionary
  noise_functions = {
      # 'block-pixel': add_block_pixel_noise,
      # 'convolve': add_convolve_noise,
      # 'keep-patch': add_keep_patch_noise,
      # 'extract-patch': add_extract_patch_noise,
      # 'pad-rotate-project': add_pad_rotate_project_noise,
      # 'line-strip': add_line_strip_noise,
      # 'salt-and-pepper': add_salt_and_pepper_noise,
      'gaussian': add_gaussian_projection_noise
  }

  for (key,func) in noise_functions.items():
      image = cv2.imread('image.jpg', cv2.IMREAD_UNCHANGED)
      image=np.array(image).astype("float32")
      image = cv2.resize(image, (512, 512))
      noise_image,image = func(image)
      clean_image = denoise_image(noise_image, method=key)

      ssim,psnr = eval_metrics(image, clean_image)

      print("Method: ", key, " SSIM: ", ssim, " PSNR: ", psnr)

      plt.figure(figsize=(10, 5))
      plt.subplot(1, 3, 1)
      plt.imshow(image, cmap="gray")
      plt.title("Original Image")
      plt.axis("off")
      plt.subplot(1, 3, 2)
      plt.imshow(noise_image, cmap="gray")
      plt.title("Noisy Image")
      plt.axis("off")
      plt.subplot(1, 3, 3)
      plt.imshow(clean_image, cmap="gray")
      plt.title("Clean Image")
      plt.axis("off")
      plt.show()
      plt.show()
      