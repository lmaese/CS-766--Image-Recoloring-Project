import tensorflow as tf
import torchvision.transforms as T
from keras.preprocessing.image import img_to_array
import numpy as np


# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(img):
    # Read and decode an image file to a uint8 tensor
    image = np.array(img)
    image = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.cast(image, tf.float32)

    input_image = tf.image.resize(image, [IMG_HEIGHT // 4, IMG_WIDTH // 4],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = tf.image.rgb_to_grayscale(input_image)
    input_image = tf.cast(input_image, tf.float32)
    input_image = tf.reshape(input_image, [IMG_HEIGHT // 4, IMG_WIDTH // 4])
    input_image = tf.tile(input_image, [1, 4])
    input_image = tf.reshape(input_image, [IMG_HEIGHT, IMG_WIDTH // 4, 1])
    input_image = tf.tile(input_image, [1, 1, 4])
    input_image = tf.reshape(input_image, [IMG_HEIGHT, IMG_WIDTH, 1])
    input_image = tf.tile(input_image, [1, 1, 3])

    return input_image, real_image
    
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def test_image(model, test_input):
    prediction = model(test_input, training=True)
    return prediction[0]

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def pix_run(img):
    # Define model and load model checkpoint
    generator = tf.keras.models.load_model("./model/pixgen")
    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(tf.train.latest_checkpoint("./training_checkpoints/")).expect_partial()
    norm_image, oth =  load_image_test(img)
    norm_image = norm_image[None]
    
    gen_image = test_image(generator,norm_image)
    output_image = img_to_array(gen_image)
    pil_image= tf.keras.utils.array_to_img(output_image)

    return pil_image
