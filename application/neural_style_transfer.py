import tensorflow as tf
import tensorflow_hub as hub
from werkzeug import datastructures
import time
from image_handler import tensor_to_image, load_image_from_file
from model import NeuralNetwork

### PARAMETRI ###
output_file_name = 'result_image.png'

#default (original content is more visible)
content_weight = 1e4
style_weight = 1e-2
total_variation_weight = 30

#variation 1 (more style)
#content_weight = 1e-10
#style_weight = 1e10
#total_variation_weight = 30

#variation 2 (idk)
#content_weight = 7.5e0
#style_weight = 1e2
#total_variation_weight = 2e2

#variation 3 (balanced?)
#content_weight = 1e4
#style_weight = 1e4
#total_variation_weight = 30

#epochs = 10
#steps_per_epoch = 100
#################

def transfer(content_image_file: datastructures.file_storage.FileStorage,
             style_image_file: datastructures.file_storage.FileStorage,
             epochs: int, steps_per_epoch: int, mode: int):
    
    content_image_file_data = content_image_file.read()
    style_image_file_data = style_image_file.read()

    content_image = load_image_from_file(content_image_file_data)
    style_image = load_image_from_file(style_image_file_data)

    tensor_to_image(content_image).save('./static/content_image.png')
    tensor_to_image(style_image).save('./static/style_image.png')

    global style_weight
    if mode == 1:
        style_weight = 1e4

    if mode != 2:
        neural_network = NeuralNetwork(content_image, style_image, content_weight, style_weight, total_variation_weight)
        #neural_network.print_stats()

        image = tf.Variable(content_image)

        start = time.time()
        neural_network.train(image, epochs, steps_per_epoch)
        end = time.time()

        print("Total time: {:.1f}".format(end-start))
    else:
        #hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        model = tf.keras.models.load_model("./model")
        image = model(tf.constant(content_image), tf.constant(style_image))[0]

    tensor_to_image(image).save('./static/' + output_file_name)
    return