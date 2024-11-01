import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(image_folder, label):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(label)
    return images, labels

def prepare_data():
    wally_images, wally_labels = load_images('./images/wally_images', label=1)
    non_wally_images, non_wally_labels = load_images('./images/non_wally_images', label=0)

    images = np.array(wally_images + non_wally_images)
    labels = np.array(wally_labels + non_wally_labels)

    return train_test_split(images, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print("Data loaded and split into training and testing sets.")
