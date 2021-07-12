import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tflite_model_file = "HDF5/2021.05.04_mnist.tflite"

# Helper function to display digit images
def show_sample(images, labels, sample_count=25):
  # Create a square with can fit {sample_count} images
  grid_count = math.ceil(math.ceil(math.sqrt(sample_count)))
  grid_count = min(grid_count, len(images), len(labels))
  
  plt.figure(figsize=(2*grid_count, 2*grid_count))
  for i in range(sample_count):
    plt.subplot(grid_count, grid_count, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.gray)
    plt.xlabel(labels[i])
  plt.show()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()


### Get input and output tensors. ###
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# tf.print("input_details:\n", input_details, end="\n-----------------------------\n")
# tf.print("output_details:\n", output_details, end="\n-----------------------------\n")
# out_details = sorted(output_details, key=lambda k: k['index']) 

# tf.print("output_details: ", output_details[0])

### Get inp_index and out_details. ###
print("input_index: ", input_details[0]["index"])
print("output_index: ", output_details[0]["index"], end="\n-----------------------------\n")



### Verify the TensorFlow Lite model. ###

# Download a test image
zero_img_path = tf.keras.utils.get_file(
    'zero.png', 
    'https://storage.googleapis.com/khanhlvg-public.appspot.com/digit-classifier/zero.png'
)

image = tf.keras.preprocessing.image.load_img(
    zero_img_path,
    color_mode = 'grayscale',
    target_size=(28, 28),
    interpolation='bilinear'
)

input_image = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
print("input_image: {}\n {}".format(input_image.shape, input_image))
# Show the pre-processed input image
# show_sample(input_image, ['Input Image: 0'], 1)

# interpreter.set_tensor(input_details[0]["index"], input_image)
# interpreter.invoke()
# output = interpreter.tensor(output_details[0]["index"])()[0]
# print("output:\n{}".format(output))



'''
### Test tflite model on random input data. ###
input_shape = input_details[0]['shape']
# print("input_shape:", input_shape)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(inp_index, input_data)

interpreter.invoke()


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("output_data:\n", output_data, end="\n-----------------------------\n")
'''
