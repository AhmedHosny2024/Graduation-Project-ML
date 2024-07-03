import joblib
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from generate_noise import *
from denoiser import *
from feature_extraction import *

transform =  A.Compose(
                [
                    A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
                ]
            )
transform2 =  A.Compose(
                [
                    A.PadIfNeeded(min_height=512, min_width=512,border_mode= cv2.BORDER_CONSTANT,value=0),
                ]
            )

base_path ='C:/Users/engah/Downloads/GP/Graduation-Project/'
image_dire='fourier_transform_svm/'

test = pd.read_csv('C:/Users/engah/Downloads/GP/Graduation-Project/datasets/test.csv')

# load svm model
model = joblib.load('model.pth')

# 4 test the model
test_img_paths = test["mimic_image_file_path"].tolist()
ytest=[]
Xtest=[]
Xtest_images=[]
yactual=[]

for img_path in test_img_paths:
    img_path = base_path + img_path
    img_path = img_path.replace("\\", "/")
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img = transform(image=img)["image"]
    choice = np.random.choice([1,2,3])
    for i in range(1,4,1):
      choice=i
      if choice == 1:
          noisy_img,label = add_block_pixel_noise(img)
          yactual.append("block-pixel")
      elif choice == 2:
          noisy_img,label = add_salt_and_pepper_noise(img)
          yactual.append("salt-and-pepper")
      elif choice == 3:
          noisy_img,label = add_gaussian_projection_noise(img)
          yactual.append("gaussian")
      else:
          noisy_img,label=img.copy(),img.copy()
          yactual.append("no-noise")
          
      noisy_img = transform2(image=noisy_img)["image"]
      label = transform2(image=label)["image"]
      # Xtest.append(all_image(noisy_img))
      # Xtest.append(Hog(noisy_img))
      # Xtest.append(Hog2(noisy_img))
      Xtest.append(fourier_transform(noisy_img))
      Xtest_images.append(noisy_img)
      ytest.append(label)

print("Testing the model...")
print("Xtest shape: ", np.array(Xtest).shape)
print("ytest shape: ", np.array(ytest).shape)
y_pred = model.predict(Xtest)
# loop over the images and try to denoise them
final_SSIM = 0
final_PSNR = 0

for i in range(len(Xtest_images)):
    print("prediction: ", y_pred[i], " actual: ", yactual[i])
    with open(image_dire+"res.txt", "a") as f:
        f.write("prediction: "+str(y_pred[i])+" actual: "+str(yactual[i])+"\n")
    denoised_image = denoise_image(Xtest_images[i], y_pred[i])
    # calculte SSIM and PSNR
    ssim, psnr = eval_metrics(ytest[i], denoised_image)
    final_SSIM += ssim
    final_PSNR += psnr
    print("Image: ", i, " SSIM: ", ssim, " PSNR: ", psnr)
    with open(image_dire+"res.txt", "a") as f:
        f.write("Image: "+str(i)+" SSIM: "+str(ssim)+" PSNR: "+str(psnr)+"\n")
    cv2.imwrite(image_dire+"denoised_image_{}.png".format(i), denoised_image)
    cv2.imwrite(image_dire+"noised_image_{}.png".format(i), Xtest_images[i])
    cv2.imwrite(image_dire+"image_{}.png".format(i), ytest[i])

print("Average SSIM: ", final_SSIM/len(Xtest_images))
print("Average PSNR: ", final_PSNR/len(Xtest_images))
with open(image_dire+"res.txt", "a") as f:
    f.write("Average SSIM: "+str(final_SSIM/len(Xtest_images))+" Average PSNR: "+str(final_PSNR/len(Xtest_images))+"\n")
