import matplotlib.pyplot as plt
import cv2

def visualize_images(X, y, num=5):
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(X[i])
        plt.title('Label: ' + ('Wally' if y[i] == 1 else 'Not Wally'))
        plt.axis('off')
    plt.show()

# After loading your images and labels
visualize_images(X, y)
