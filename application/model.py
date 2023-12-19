import tensorflow as tf
import os
from vgg19 import vgg_layers
from style_content_model import StyleContentModel

class NeuralNetwork:

    def __init__(self, content_image, style_image, content_weight, style_weight, total_variation_weight) -> None:
        # The layer to use for the content loss.
        self.content_layers = ['block5_conv2'] 

        # List of layers to use for the style loss.
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

        self.style_extractor = vgg_layers(self.style_layers)
        self.style_outputs = self.style_extractor(style_image * 255)

        self.extractor = StyleContentModel(self.style_layers, self.content_layers)

        self.style_targets = self.extractor(style_image)['style']
        self.content_targets = self.extractor(content_image)['content']

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = total_variation_weight

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # Prints the statistics of each layer's output
    def print_stats(self):
        for name, output in zip(self.style_layers, self.style_outputs):
            print(name)
            print("  shape: ", output.numpy().shape)
            print("  min: ", output.numpy().min())
            print("  max: ", output.numpy().max())
            print("  mean: ", output.numpy().mean())
            print()

    # The "style loss" is designed to maintain the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of feature maps from the style reference image
    # and from the generated image.
    # The "content loss" is designed to maintain the "content" of the base image in the generated image.
    def style_content_loss(self, outputs, style_weight, content_weight):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2) for name in style_outputs.keys()])
        style_loss *= style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2) for name in content_outputs.keys()])
        content_loss *= content_weight / self.num_content_layers

        loss = style_loss + content_loss
        return loss
    
    # Function to keep the pixel values between 0 and 1
    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    
    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs, self.style_weight, self.content_weight)
            loss += self.total_variation_weight * tf.image.total_variation(image)
            grad = tape.gradient(loss, image)
            self.opt.apply_gradients([(grad, image)])
            image.assign(self.clip_0_1(image))

    def train(self, image, epochs, steps_per_epoch):
        step = 0
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                step += 1
                self.train_step(image)
                print(".", end='', flush=True)
            print("Train step: {}".format(step))

    def save(self):
        self.extractor.compile()
        self.extractor.save("./checkpoints/model")