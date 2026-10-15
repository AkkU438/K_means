import numpy as np
import matplotlib.pyplot as plt
import struct

#Need to add the old work from homework 2 here. Then need to make the k-means algo - then start working on the stuff

with open("data/train-images.idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)

images = data.reshape(num_images, rows * cols)
images = images / 255.0

def kmeans(images, k):
    centroids = images[np.random.choice(images.shape[0], k, replace=False)]
    
    for _ in range(100):
        distances = np.linalg.norm(images[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([images[labels == i].mean(axis=0) for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
            
        centroids = new_centroids
    
    return centroids

def display_centroids(centroids):
    k = len(centroids)

    cols = 5
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))

    axes = np.array(axes).reshape(-1)

    for i in range(k):
        axes[i].imshow(centroids[i].reshape(28, 28), cmap="gray")
        axes[i].set_title(f"Centroid {i + 1}")
        axes[i].axis("off")

    # Hide unused plots
    for i in range(k, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

