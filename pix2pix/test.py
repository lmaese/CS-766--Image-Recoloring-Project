import tensorflow as tf
import argparse
import os
from keras.preprocessing.image import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_path", type=str, default="./training_checkpoints/", help="Path to checkpoint folder")
parser.add_argument("--model", type=str, default="./model/pixgen", help="Path to model")
opt = parser.parse_args()
print(opt)


# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
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
    #pred_pil = pil_transform(prediction[0])
    return prediction[0]

def load_image_test(image_file):
    input_image, real_image = load(image_file)
    #   input_image, real_image = resize(input_image, real_image,
    #                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


os.makedirs("output", exist_ok=True)

# Define model and load model checkpoint
generator = tf.keras.models.load_model(opt.model)
checkpoint = tf.train.Checkpoint(generator=generator)
checkpoint.restore(tf.train.latest_checkpoint(opt.checkpoint_path))

norm_image, oth =  load_image_test(opt.image_path)
norm_image = norm_image[None]
gen_image = test_image(generator,norm_image)

# Save image
save_img("output/test.png", gen_image)


