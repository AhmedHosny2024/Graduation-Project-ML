from skimage.feature import hog
import numpy as np


# 1- got 0.1875 accuracy 
# Average SSIM:  0.7722181199388367
# Average PSNR:  28.970157366174934
def all_image(image):
  return image.flatten() 

# 2- got 0.4375 accuracy
# Average SSIM:  0.7970383516956439
# Average PSNR:  29.497383485113495
def Hog(image):
  fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True,feature_vector=True)
  return fd

# 3- got 0.25 accuracy
# Average SSIM: 0.9703221610497511 
# Average PSNR: 35.96126321499047
def Hog2(image):
  fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True,feature_vector=True)
  return fd

# 4- got 0.3125 accuracy
# Average SSIM:  0.7040074849462885
# Average PSNR:  26.0354890824695
def fourier_transform(image):
  f = np.fft.fft2(image)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20*np.log(np.abs(fshift))
  return magnitude_spectrum.flatten()

'''
when the model classifiy the image noise type wrong (which is happen many times as the classifier acuracy is low) 
the denoised image will be wrong (as the denoising function will be wrong as well) 
we got high SSIM and PSNR because the change in the image is small line line strip or block pixel noise
BUT it remove the medical information in the image or didn't remove the noise which is the main goal of the project
so we didn't use it
it get this result which train on 4 types of noise 150 image and test on 16 image
'''