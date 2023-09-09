import tensorflow as tf
from werkzeug import datastructures
import time
from image_handler import tensor_to_image, load_image_from_file
from model import NeuralNetwork

### PARAMETRI ###
#content_url = 'https://media.gettyimages.com/id/1207663571/it/foto/empty-pavement-with-modern-architecture.jpg?s=612x612&w=gi&k=20&c=zs96UNIqroSsjZLtkS0VUww6lBUh1rZ3zq3QU0qaXAA='
#style_url = 'https://p.turbosquid.com/ts-thumb/5m/fo2FbO/Kq/render/png/1679499783/600x600/fit_q87/d2223da563cc146e6647eb07a3396663a27647f8/render.jpg'
output_file_name = 'result_image.png'

content_weight = 10000.0
style_weight = 0.01
total_variation_weight = 30

#epochs = 10
#steps_per_epoch = 100
#################

def transfer(content_image_file: datastructures.file_storage.FileStorage, style_image_file: datastructures.file_storage.FileStorage, epochs, steps_per_epoch):

    content_image_file_data = content_image_file.read()
    style_image_file_data = style_image_file.read()

    content_image = load_image_from_file(content_image_file_data)
    style_image = load_image_from_file(style_image_file_data)

    tensor_to_image(content_image).save('./static/content_image.png')
    tensor_to_image(style_image).save('./static/style_image.png')

    #show_two_images(content_image, 'Content Image', style_image, 'Style Image')

    neural_network = NeuralNetwork(content_image, style_image, content_weight, style_weight, total_variation_weight)
    #neural_network.print_stats()

    image = tf.Variable(content_image)

    start = time.time()
    neural_network.train(image, epochs, steps_per_epoch)
    end = time.time()

    print("Total time: {:.1f}".format(end-start))

    #show_image(image, 'Result')
    tensor_to_image(image).save('./static/' + output_file_name)
    return