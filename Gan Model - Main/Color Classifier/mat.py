import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_dominant_color(image_path, k=3):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Apply color quantization using K-means clustering
    # kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)

    # Get the RGB values of the cluster centers
    colors = kmeans.cluster_centers_

    # print("colors : ", colors)

    # Convert the colors to integers
    colors = colors.round().astype(int)

    # Sort the colors by frequency
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    colors = colors[sorted_indices]

    # Create an image with the dominant colors
    dominant_image = np.zeros_like(image)
    labels_reshaped = kmeans.labels_.reshape(image.shape[:2])
    for i, color in enumerate(colors):
        dominant_image[labels_reshaped == i] = color

    return colors[0]

    # Display the image with the dominant colors
    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title("Original Image")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(dominant_image)
    # plt.title("Dominant Colors (k={})".format(k))
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()

# Example usage
# extract_dominant_color("2_2000.jpg", k=5)
