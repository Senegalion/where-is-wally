import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path='./model/model.keras'):
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image_path, model):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction[0][0] > 0.5

if __name__ == "__main__":
    model = load_model()
    image_path = './images/crowd_images/with_wally_images/image_8_wally.jpg'
    # image_path = './images/crowd_images/without_wally_images/image14_no_wally.jpg'
    if predict(image_path, model):
        print("Wally detected in the image.")
    else:
        print("No Wally detected in the image.")
