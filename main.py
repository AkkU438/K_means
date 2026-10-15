import numpy as np
import matplotlib.pyplot as plt
import struct

#Need to add the old work from homework 2 here. Then need to make the k-means algo - then start working on the stuff

with open("data/train-images.idx3-ubyte", "rb") as f:
    magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)

images = data.reshape(num_images, rows * cols)
images = images / 255.0
print(images.shape)