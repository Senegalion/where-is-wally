import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from preprocess import prepare_data

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    images, labels = prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = create_model()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    model_dir = './model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model.save(os.path.join(model_dir, 'model.keras'))
    print("Model trained and saved.")
    
if __name__ == "__main__":
    train_model()