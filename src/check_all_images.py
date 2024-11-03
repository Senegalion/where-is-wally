import os
import cv2
from predict import load_model, predict, preprocess_image

def test_images_in_folder(folder_path, model, label=""):
    print(f"\nTesting images in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load image {filename}. Skipping.")
            continue
        
        if predict(img_path, model):
            print(f"Wally detected in {label} image: {filename}")
        else:
            print(f"No Wally detected in {label} image: {filename}")

if __name__ == "__main__":
    model = load_model()

    test_images_in_folder('./images/crowd_images/without_wally_images', model, label="crowd without Wally")
    test_images_in_folder('./images/crowd_images/with_wally_images', model, label="crowd with Wally")
