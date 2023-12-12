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

def laplacian_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Apply convolution
    edges = convolve(gray_image, laplacian_kernel)

    # Normalize the output to be in the range [0, 255]
    edges = ((edges - np.min(edges)) / (np.max(edges) - np.min(edges)) * 255).astype(np.uint8)

    return edges

# Read an image from file
image = cv2.imread('girlImage.png')

# Apply Laplacian edge detection
edges = laplacian_edge_detection(image)

# Display the original and the edges images
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')

cv2.imwrite('laplacian.png', edges)
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Laplacian Edges')
plt.show()
