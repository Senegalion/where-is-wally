import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.resize(image, (128, 128))
            images.append(image)
            labels.append(label)
    return images, labels

def augment_images(images):
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    augmented_images = []
    for image in images:
        image = np.expand_dims(image, 0)
        for batch in datagen.flow(image, batch_size=1):
            augmented_images.append(batch[0])
            if len(augmented_images) >= len(images):
                break
    return augmented_images

def prepare_data():
    wally_images, wally_labels = load_images_from_folder('./images/wally_images', label=1)
    non_wally_images, non_wally_labels = load_images_from_folder('./images/non_wally_images', label=0)

    crowd_wally_images, crowd_wally_labels = load_images_from_folder('./images/crowd_images/with_wally_images', label=1)
    crowd_non_wally_images, crowd_non_wally_labels = load_images_from_folder('./images/crowd_images/without_wally_images', label=0)
    
    augmented_wally_images = augment_images(wally_images)
    images = np.array(augmented_wally_images + non_wally_images + crowd_wally_images + crowd_non_wally_images)
    labels = np.array(
        [1] * len(augmented_wally_images) + 
        [0] * len(non_wally_images) + 
        [1] * len(crowd_wally_images) + 
        [0] * len(crowd_non_wally_images)
    )
    
    return images, labels
