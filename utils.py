# io.py
import cv2

def read_images(image_directory, image1_filename, image2_filename):
    image1 = cv2.imread(image_directory + image1_filename)
    image2 = cv2.imread(image_directory + image2_filename)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    return image1, image2
