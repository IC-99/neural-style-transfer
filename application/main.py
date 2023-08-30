import tensorflow as tf
import time
from image_handler import get_image, tensor_to_image, show_image, show_two_images
from model import NeuralNetwork

### PARAMETRI ###
content_url = 'https://media.gettyimages.com/id/1207663571/it/foto/empty-pavement-with-modern-architecture.jpg?s=612x612&w=gi&k=20&c=zs96UNIqroSsjZLtkS0VUww6lBUh1rZ3zq3QU0qaXAA='
style_url = 'https://p.turbosquid.com/ts-thumb/5m/fo2FbO/Kq/render/png/1679499783/600x600/fit_q87/d2223da563cc146e6647eb07a3396663a27647f8/render.jpg'
output_file_name = 'result-image.png'

style_weight = 0.01
content_weight = 10000.0
total_variation_weight = 30

epochs = 2
steps_per_epoch = 3
#################

if __name__ == "__main__":
    content_image = get_image(content_url)
    style_image = get_image(style_url)

    show_two_images(content_image, 'Content Image', style_image, 'Style Image')

    neural_network = NeuralNetwork(content_image, style_image, content_weight, style_weight, total_variation_weight)
    #neural_network.print_stats()

    image = tf.Variable(content_image)

    start = time.time()
    neural_network.train(image, epochs, steps_per_epoch)
    end = time.time()

    print("Total time: {:.1f}".format(end-start))

    tensor_to_image(image)
    show_image(image, 'Result')
    tensor_to_image(image).save(output_file_name)