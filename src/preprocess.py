import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(image_folder):
    images = []
    labels = []

    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        image = cv2.imread(img_path)

        if image is not None:
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(1 if 'wally' in filename.lower() else 0)

    return np.array(images), np.array(labels)

def split_data(images, labels):
    return train_test_split(images, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    images, labels = load_images('./images')
    X_train, X_test, y_train, y_test = split_data(images, labels)
    print("Data loaded and split into training and testing sets.")
