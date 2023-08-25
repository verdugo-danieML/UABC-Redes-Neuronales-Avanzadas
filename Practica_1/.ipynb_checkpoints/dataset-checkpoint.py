import gzip
import numpy as np
import struct

def read_mnist_images(images_file_path):
    with gzip.open(images_file_path, "rb") as images_file:
        images_content = images_file.read()
    
    magic_number, num_images, num_rows, num_cols = struct.unpack_from('>IIII', images_content, 0)
    images_offset = struct.calcsize('>IIII')
    
    images = np.frombuffer(images_content, dtype=np.uint8, offset=images_offset)
    images = images.reshape((num_images, num_rows, num_cols))
    return images

def read_mnist_labels(labels_file_path):
    with gzip.open(labels_file_path, "rb") as labels_file:
        labels_content = labels_file.read()
    
    magic_number, num_labels = struct.unpack_from('>II', labels_content, 0)
    labels_offset = struct.calcsize('>II')
    
    labels = np.frombuffer(labels_content, dtype=np.uint8, offset=labels_offset)
    return labels

