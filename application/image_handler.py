import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

def get_image(url):
    return load_image(tf.keras.utils.get_file(origin=url))

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Function to load an image and limit its maximum dimension to 512 pixels.
def load_image(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to load an image and limit its maximum dimension to 512 pixels.
def load_image_from_file(img):
    max_dim = 512
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Function to display an image.
def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)

def show_image(image, image_name):
    imshow(image, image_name)
    plt.show()

def show_two_images(image1, image1_name: str, image2, image2_name: str):
    plt.subplot(1, 2, 1)
    imshow(image1, image1_name)
    plt.subplot(1, 2, 2)
    imshow(image2, image2_name)
    plt.show()
