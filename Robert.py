import cv2
import numpy as np

def convolve(image, kernel):
    """Performs convolution operation on the image."""
    kernel_size = kernel.shape[0]
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant', constant_values=0)
    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)

    return result

roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=np.float64)
roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=np.float64)

img = cv2.imread("girlImage.png", 0).astype('float64') / 255.0

# Convolve with Roberts Cross kernels
vertical = convolve(img, roberts_cross_v)
horizontal = convolve(img, roberts_cross_h)

# Compute gradient magnitude
edged_img = np.sqrt(np.square(horizontal) + np.square(vertical)) * 255
edged_img = np.clip(edged_img, 0, 255).astype(np.uint8)

# Save the output image
cv2.imwrite("Robert_output.jpg", edged_img)
