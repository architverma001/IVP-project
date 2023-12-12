import cv2
import numpy as np

def gaussian_kernel(size, sigma):
    """Generates a 2D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

#step 1: Apply Gaussian blur
def apply_gaussian_blur(image, kernel_size, sigma):
    """Applies Gaussian blur to the input image."""
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return blurred_image

#step 2: Calculate gradients
def calculate_gradients(image):
    """Calculates gradients (magnitude and direction) using Sobel filters."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.hypot(sobel_x, sobel_y)
    theta = np.arctan2(sobel_y, sobel_x)
    
    return magnitude, theta
#step 3: Non-maximum suppression
def non_maximum_suppression(magnitude, theta):
    """Performs non-maximum suppression to thin edges."""
    rows, cols = magnitude.shape
    result = np.zeros_like(magnitude, dtype=np.uint8)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                result[i, j] = magnitude[i, j]
            else:
                result[i, j] = 0
    
    return result
#step 4: Hysteresis edge tracking
def hysteresis_edge_tracking(image, low_threshold, high_threshold):
    """Performs edge tracking by hysteresis."""
    strong_edges = (image > high_threshold).astype(np.uint8) * 255
    weak_edges = ((image >= low_threshold) & (image <= high_threshold)).astype(np.uint8) * 255
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(weak_edges, connectivity=8)
    
    for i in range(1, stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] >= 50:  # Adjust the area threshold as needed
            strong_edges[labels == i] = 255
    
    return strong_edges
#step 5: Canny edge detection
def canny_edge_detection(image, kernel_size=5, sigma=1.4, low_threshold=20, high_threshold=50):
    """Performs Canny edge detection on the input image."""
    # Step 1: Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image, kernel_size, sigma)
    
    # Step 2: Calculate gradients
    magnitude, theta = calculate_gradients(blurred_image)
    
    # Step 3: Non-maximum suppression
    suppressed_image = non_maximum_suppression(magnitude, theta)
    
    # Step 4: Hysteresis edge tracking
    edges = hysteresis_edge_tracking(suppressed_image, low_threshold, high_threshold)
    
    return edges

# Load the image
img = cv2.imread('girlImage.png', cv2.IMREAD_GRAYSCALE)
blurred_image = apply_gaussian_blur(img, 5, 1.4)
cv2.imshow('Blurred Image', blurred_image)
magnitude, theta = calculate_gradients(blurred_image)
cv2.imshow('Magnitude', magnitude)
cv2.imshow('Theta', theta)
suppressed_image = non_maximum_suppression(magnitude, theta)
cv2.imshow('Suppressed Image', suppressed_image)
# Perform Canny edge detection
edges = canny_edge_detection(img)

# Display the result
cv2.imshow('Canny Edge Detection', edges)
cv2.imwrite('cannyimage.png', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
