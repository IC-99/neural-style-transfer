import tensorflow as tf

# Creates a VGG model (a pretrained image classification network) that returns a list of intermediate output values.
def vgg_layers(layer_names):
    # Build a VGG19 model loaded with pre-trained ImageNet weights.
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Get the symbolic outputs of each "key" layer.
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # Set up a model that returns the activation values for every layer in VGG19.
    model = tf.keras.Model([vgg.input], outputs)
    return model