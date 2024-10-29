import matplotlib.pyplot as plt
import cv2

def display_image(image, title=""):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image = cv2.imread('./images/image1_wally.jpg')
    display_image(image, "Test Image")
