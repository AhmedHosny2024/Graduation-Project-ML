import cv2
import selectivesearch
import matplotlib.pyplot as plt
import pandas as pd 
import os

# Load the image
data_info = pd.read_csv("datasets/train.csv", header=None)
data_info = data_info.iloc[1:]  # assuming your CSV has headers

for idx in range(len(data_info)):
    # get the image path from the CSV file
    img_path = data_info.iloc[idx, 3]

    # construct the full image path
    img_path = os.path.join(os.getcwd(), img_path.replace("\\", "/"))

    # read the image
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"Failed to read image: {img_path}")
        continue

    # resize the image to 512x512 (if necessary)
    image = cv2.resize(image, (512, 512))

    # convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform selective search
    _, regions = selectivesearch.selective_search(image,min_size=500)
    print("Number of region proposals: ", len(regions))

    # Draw rectangles on the image for each proposal
    output_image = image.copy()
    for region in regions:
        x, y, w, h = region['rect']
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result 
    plt.imshow(output_image)
    plt.axis('off')  # turn off axis labels
    plt.show()
