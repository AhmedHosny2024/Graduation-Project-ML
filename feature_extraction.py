from skimage.feature import hog
import numpy as np

# SVM
#   Accuracy:  0.3333333333333333 
#   Average SSIM:  0.9706744820566566
#   Average PSNR:  36.36342098511046
# RANDOM FOREST
#   Accuracy:  0.4375
#   Average SSIM:  0.8302919768181796
#   Average PSNR:  30.89012875967285
def all_image(image):
  return image.flatten() 

# SVM
#   Accuracy:  0.3333333333333333
#   Average SSIM:  0.9706967410660425
#   Average PSNR:  36.412467623255
# RANDOM FOREST
#   Accuracy:  1.0   WOW 😎
#   Average SSIM:  0.9327131385372782
#   Average PSNR:  34.47807884234579
def Hog(image):
  fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True,feature_vector=True)
  return fd

# SVM
#   3Accuracy:  0.3333333333333333
#   Average SSIM:  0.6988707117158928
#   Average PSNR:  26.017515034225344
# RANDOM FOREST
#   Accuracy:  0.9583333333333334
#   Average SSIM:  0.9328699278993343
#   Average PSNR:  34.580314125480264
def Hog2(image):
  fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True,feature_vector=True)
  return fd

# SVM
#   Accuracy:  0.3333333333333333
#   Average SSIM:  0.9706847828242776
#   Average PSNR:  36.383200709957656
# RANDOM FOREST
#   Accuracy:  1.0   WOW 😎
#   Average SSIM:  0.9327725672987425
#   Average PSNR:  34.489576292298
def fourier_transform(image):
  f = np.fft.fft2(image)
  fshift = np.fft.fftshift(f)
  magnitude_spectrum = 20*np.log(np.abs(fshift))
  return magnitude_spectrum.flatten()

# SVM
#   Accuracy:  0.3333333333333333
#   Average SSIM:  0.9706054619935945
#   Average PSNR:  36.34067142764114
# RANDOM FOREST 
#   Accuracy:  1.0
#   Average SSIM:  0.9327485700951085
#   Average PSNR:  34.4827113404983
def mix_features(image):
  return np.concatenate((Hog(image),fourier_transform(image)),axis=0)

'''
when the model classifiy the image noise type wrong the denoised image will be wrong (as the denoising function will be wrong as well) 
we got high SSIM and PSNR because the change in the image is block pixel noise
BUT it remove the medical information in the image or didn't remove the noise which is the main goal of the project
so we didn't use it
it get this result which train on 4 types of noise 450 image and test on 48 image so the classifier get high accuracy but the medical information is lost as i save the denoised image and the noised image and the original image
if we have more than one type of noise in the image the denoier function of one type of noise may affect the other type of noise (generate more noise in the image)
so we didn't use it and use end to end model 
'''