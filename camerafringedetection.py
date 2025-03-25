import cv2
import numpy as np
import matplotlib.pyplot as plt
from picamera2 import Picamera2
import time
from scipy.signal import find_peaks
from scipy.optimize import differential_evolution
from skimage.transform import warp_polar

# Constants
WAVELENGTH_NM = 680  # Laser wavelength in nm (adjust for your laser)
PIXEL_SIZE_UM = 1.12  # Raspberry Pi Camera v2 pixel size in micrometers
FOCAL_LENGTH_MM = 50  # Lens focal length in mm (adjust to your setup)
REFERENCE_WIDTH_MM = 80  # Physical width of the white screen in mm
N_FRAMES = 5  # Number of frames to average for noise reduction
N_FRINGES = 4  # Number of fringes to analyze

def capture_frames(n=N_FRAMES):
    """Capture multiple images from the camera and compute an average frame."""
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1920, 1080)})  # Higher resolution
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    
    frames = []
    for _ in range(n):
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        time.sleep(0.05)  # Short delay to allow minor variations
    
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)  # Compute average frame
    cv2.imwrite("captured_avg_gray.jpg", avg_frame)  # Save averaged grayscale image
    return avg_frame

def detect_fringes_from_edges(edges):
    """Find the bounding box for the interference fringes in the detected edges."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return x, y, w, h
    return None

def detect_white_screen(gray):
    """Detect the white reference screen and use it to crop the image based on fringe edges."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    cv2.imwrite("detected_edges.jpg", edges)
    
    fringe_bbox = detect_fringes_from_edges(edges)
    if fringe_bbox:
        return fringe_bbox, edges  # Use detected fringes as cropping reference
    
    return None, edges

def find_initial_center(edges):
    """Find initial fringe center by using the outermost detected edges in x and y directions."""
    y_indices, x_indices = np.where(edges > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return center_x, center_y

def transform_to_polar(gray, center_x, center_y):
    """Transform the image to polar coordinates centered at the detected fringe center."""
    polar_image = warp_polar(gray, center=(center_x, center_y), radius=min(gray.shape) // 2)
    radial_intensity = np.mean(polar_image, axis=0)  # Sum along the angular axis
    return radial_intensity

def detect_circular_fringes(gray, edges, roi):
    """Detect interference fringes using a polar transformation of intensity."""
    x, y, w, h = roi
    cropped = gray[y:y+h, x:x+w]
    cv2.imwrite("expanded_fringes.jpg", cropped)
    
    # Find initial center from edges
    initial_center_x, initial_center_y = find_initial_center(edges)
    if initial_center_x is None or initial_center_y is None:
        initial_center_x, initial_center_y = w // 2, h // 2  # Fallback
    
    radial_profile = transform_to_polar(cropped, initial_center_x, initial_center_y)
    
    peaks = detect_maxima(radial_profile)
    
    # Plot the radial intensity profile
    plt.figure(figsize=(8, 4))
    plt.plot(radial_profile, label="Radial Intensity")
    plt.scatter(peaks, radial_profile[peaks], color='red', label="Detected Peaks")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Intensity")
    plt.title("Radial Intensity Profile of Interference Fringes")
    plt.legend()
    plt.savefig("radial_intensity_plot.png")
    plt.show()
    
    if len(peaks) < 3:
        print("Warning: Not enough fringes detected. Adjust laser focus or detection parameters.")
        return None, cropped, initial_center_x, initial_center_y
    
    return peaks, cropped, initial_center_x, initial_center_y

def detect_maxima(intensity_profile):
    """Detect maxima in radial intensity profile using peak detection."""
    peaks, _ = find_peaks(intensity_profile, distance=10, prominence=3)
    return peaks

def calculate_path_difference(peaks, pixel_scale_um):
    """Calculate path difference using first and second order fringe distances."""
    if len(peaks) < 3:
        return None
    
    fringe_1 = peaks[1] * pixel_scale_um
    fringe_2 = peaks[2] * pixel_scale_um
    
    path_difference_um = WAVELENGTH_NM * (fringe_2**2 - fringe_1**2) / (2 * FOCAL_LENGTH_MM * 1e3)
    return path_difference_um, fringe_1, fringe_2

def main():
    """Main function to capture, process, and analyze interference fringes."""
    gray = capture_frames()
    roi, edges = detect_white_screen(gray)
    
    if roi:
        x, y, w, h = roi
        pixel_scale_um = (REFERENCE_WIDTH_MM * 1e3) / w
        
        peaks, cropped, center_x, center_y = detect_circular_fringes(gray, edges, roi)
        
        if peaks is None:
            print("Error: No valid fringes detected. Try adjusting the experiment setup.")
            return
        
        result = calculate_path_difference(peaks, pixel_scale_um)
        
        if result is None:
            print("Error: Not enough detected fringes for calculation.")
            return
        
        path_diff, fringe_1, fringe_2 = result
        
        print(f"Detected {len(peaks)} fringes")
        print(f"1st Order Fringe Distance: {fringe_1:.2f} µm")
        print(f"2nd Order Fringe Distance: {fringe_2:.2f} µm")
        print(f"Calculated Path Difference: {path_diff:.2f} µm")
    else:
        print("No reference frame detected. Ensure the white frame is visible.")

if __name__ == "__main__":
    main()
