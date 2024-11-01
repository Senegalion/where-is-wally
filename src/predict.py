import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
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
    print("Raw prediction score:", prediction)
    return prediction[0][0] > 0.5

if __name__ == "__main__":
    model = load_model('../models/model.keras')
    # image_path = './images/crowd_images/without_wally_images/image19_no_wally.jpg'
    # image_path = './images/crowd_images/with_wally_images/image6_wally.jpg'
    image_path = './images/wally_images/wally_2.jpg'
    if predict(image_path, model):
        print("Wally is present!")
    else:
        print("Wally is not present.")
