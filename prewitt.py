import cv2
import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    height, width = image.shape
    k_size = len(kernel)
    pad = k_size // 2
    output = np.zeros((height - 2 * pad, width - 2 * pad))

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            output[i - pad, j - pad] = np.sum(image[i - pad:i + pad + 1, j - pad:j + pad + 1] * kernel)

    return output

def prewitt_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Prewitt kernels
    prewitt_kernel_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])

    prewitt_kernel_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])

    # Apply convolution to compute gradients in the horizontal and vertical directions
    gradient_x = convolve(gray_image, prewitt_kernel_x)
    gradient_y = convolve(gray_image, prewitt_kernel_y)

    # Combine the gradients to find the magnitude
    edges = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalize the output to be in the range [0, 255]
    edges = ((edges - np.min(edges)) / (np.max(edges) - np.min(edges)) * 255).astype(np.uint8)

    return edges

# Read an image from file
image = cv2.imread('girlImage.png')

# Apply Prewitt edge detection
edges = prewitt_edge_detection(image)
cv2.imwrite('prewitt.png', edges)

# Display the original and the edges images
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Prewitt Edges')
plt.show()
