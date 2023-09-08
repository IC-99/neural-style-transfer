import tensorflow as tf
from werkzeug import datastructures
from image_handler import tensor_to_image, load_image_from_file
import tensorflow_hub as hub

def transfer(content_image_file: datastructures.file_storage.FileStorage, style_image_file: datastructures.file_storage.FileStorage):

    content_image_file_data = content_image_file.read()
    style_image_file_data = style_image_file.read()

    content_image = load_image_from_file(content_image_file_data)
    style_image = load_image_from_file(style_image_file_data)

    tensor_to_image(content_image).save('./static/content_image.png')
    tensor_to_image(style_image).save('./static/style_image.png')

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    tensor_to_image(stylized_image).save('./static/result_image.png')
    return