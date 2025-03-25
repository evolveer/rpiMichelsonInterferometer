import cv2
import numpy as np
import time
from picamera2 import Picamera2
import matplotlib.pyplot as plt
from skimage.transform import warp_polar
from scipy.signal import find_peaks
import os

# Constants
IMAGE_FOLDER = "captured_images2"
N_IMAGES = 400  # Number of images to capture and analyze
WAVELENGTH_NM = 532
PIXEL_SIZE_UM = 1.12
FOCAL_LENGTH_MM = 50
REFERENCE_WIDTH_MM = 80

def capture_and_save_images(n=N_IMAGES):
    """Capture multiple images and save them to disk."""
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1920, 1080)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    
    for i in range(n):
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filename = os.path.join(IMAGE_FOLDER, f"image_{i:03d}.jpg")
        cv2.imwrite(filename, gray)
        print(f"Saved {filename}")
        time.sleep(0.1)
    print("Image capture complete.")

def load_images():
    """Load all images from the disk."""
    images = []
    for filename in sorted(os.listdir(IMAGE_FOLDER)):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(IMAGE_FOLDER, filename), cv2.IMREAD_GRAYSCALE)
            images.append(img)
    print(f"Loaded {len(images)} images.")
    return images

def detect_fringes_from_edges(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.boundingRect(max(contours, key=cv2.contourArea))
    return None

def detect_white_screen(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    return detect_fringes_from_edges(edges), edges

def find_initial_center(edges):
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None
    return (np.min(x_indices) + np.max(x_indices)) // 2, (np.min(y_indices) + np.max(y_indices)) // 2

def transform_to_polar(gray, center_x, center_y):
    polar_image = warp_polar(gray, center=(center_x, center_y), radius=min(gray.shape) // 2)
    return np.mean(polar_image, axis=0)

def detect_circular_fringes(gray):
    roi, edges = detect_white_screen(gray)
    if roi is None:
        return None, None, None, None
    x, y, w, h = roi
    cropped = gray[y:y+h, x:x+w]
    center_x, center_y = find_initial_center(edges)
    if center_x is None or center_y is None:
        center_x, center_y = w // 2, h // 2
    radial_profile = transform_to_polar(cropped, center_x, center_y)
    return detect_maxima(radial_profile), cropped, center_x, center_y

def detect_maxima(intensity_profile):
    peaks, _ = find_peaks(intensity_profile, distance=10, prominence=3)
    return peaks

def analyze_images():
    images = load_images()
    for i, img in enumerate(images):
        peaks, cropped, center_x, center_y = detect_circular_fringes(img)
        if peaks is not None:
            plt.figure(figsize=(8, 4))
            plt.plot(transform_to_polar(cropped, center_x, center_y), label=f"Image {i}")
            plt.scatter(peaks, [transform_to_polar(cropped, center_x, center_y)[p] for p in peaks], color='red')
            plt.xlabel("Radius (pixels)")
            plt.ylabel("Intensity")
            plt.title(f"Radial Intensity Profile of Image {i}")
            plt.legend()
            plt.show()
        else:
            print(f"No fringes detected in image {i}")

# Main Execution
capture_and_save_images()  # Capture and save images
#analyze_images()  # Analyze the saved images
