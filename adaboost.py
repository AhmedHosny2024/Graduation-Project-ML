import pandas as pd
import cv2
import numpy as np
import joblib
import sys
import albumentations as A
from generate_noise import *
from feature_extraction import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

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

# 1 read images
train = pd.read_csv('C:/Users/engah/Downloads/GP/Graduation-Project/datasets/train.csv')
test = pd.read_csv('C:/Users/engah/Downloads/GP/Graduation-Project/datasets/test.csv')

tain_img_paths = train["mimic_image_file_path"].tolist()
Xtrain=[]
ytrain=[]

# 2 add noise and save the label as the key
for img_path in tain_img_paths:
    img_path = base_path + img_path
    img_path = img_path.replace("\\", "/")
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    # resize the image to 512x512
    img = transform(image=img)["image"]
    noise_type = np.random.choice([1,2,3])
    for i in range(1,4,1):
      choice=i
      if noise_type == 1:
          noisy_img,_ = add_block_pixel_noise(img)
          label ="block-pixel"
      elif noise_type == 2:
          noisy_img,_ = add_salt_and_pepper_noise(img)
          label ="salt-and-pepper"
      elif noise_type == 3:
          noisy_img,_ = add_gaussian_projection_noise(img)
          label ="gaussian"
      else:
          noisy_img,_=img.copy(),img.copy()
          label ="no-noise"
      
      noisy_img = transform2(image=noisy_img)["image"]
      # Xtrain.append(all_image(noisy_img))
      # Xtrain.append(Hog(noisy_img))
      # Xtrain.append(Hog2(noisy_img))
      # Xtrain.append(fourier_transform(noisy_img))
      Xtrain.append(mix_features(noisy_img))
      ytrain.append(label)

# 3 train the model
model = AdaBoostClassifier(RandomForestClassifier(random_state=0, class_weight='balanced',n_estimators=11,max_features='sqrt',min_samples_leaf=2,min_samples_split=2), n_estimators=11, random_state=0)

model.fit(Xtrain,ytrain)

# 4 save the model
joblib.dump(model, 'model.pth')

# 5 test the model
test_img_paths = test["mimic_image_file_path"].tolist()
Xtest=[]
ytest=[]
for img_path in test_img_paths:
    img_path = base_path + img_path
    img_path = img_path.replace("\\", "/")
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img = transform(image=img)["image"]
    choice = np.random.choice([1,2,3])   
    for i in range(1,4,1):
      choice=i     
      if choice == 1:
          noisy_img,_ = add_block_pixel_noise(img)
          label ="block-pixel"
      elif choice == 2:
          noisy_img,_ = add_salt_and_pepper_noise(img)
          label ="salt-and-pepper"
      elif choice == 3:
          noisy_img,_ = add_gaussian_projection_noise(img)
          label ="gaussian"
      else:
          noisy_img,_=img.copy(),img.copy()
          label ="no-noise"
      noisy_img = transform2(image=noisy_img)["image"]
      # Xtest.append(all_image(noisy_img))
      # Xtest.append(Hog(noisy_img))
      # Xtest.append(Hog2(noisy_img))
      # Xtest.append(fourier_transform(noisy_img))
      Xtest.append(mix_features(noisy_img))
      ytest.append(label)

print("Testing the model...")
print("Xtest shape: ", np.array(Xtest).shape)
print("ytest shape: ", np.array(ytest).shape)
y_pred = model.predict(Xtest)
# calculate the accuracy
accuracy = np.sum(y_pred == ytest) / len(ytest)
print("Accuracy: ", accuracy)
