import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

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

print("Tflite information:", end="\n-----------------------------\n")
### Get input and output tensors. ###
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# tf.print("input_details:\n", input_details, end="\n-----------------------------\n")
# tf.print("output_details:\n", output_details, end="\n-----------------------------\n")
# out_details = sorted(output_details, key=lambda k: k['index']) 
# tf.print("output_details: ", output_details[0])

print("input_index: ", input_details[0]["index"])
print("output_index: ", output_details[0]["index"])

print("input_shape: ", input_details[0]["shape"])
print("output_shape: ", output_details[0]["shape"], end="\n-----------------------------\n")

### Verify the TensorFlow Lite model. ###
# Load a test image.
# shape: (716, 716, 3)
img = cv2.imread('./zero.png')

# shape: (28, 28, 3)
image_28 = cv2.resize(img, (28, 28))

# shape: (28, 28)
gray_img = cv2. cvtColor(image_28, cv2.COLOR_BGR2GRAY)

# shape: (1, 28, 28)
image = np.expand_dims(np.array(gray_img, dtype=np.float32) / 255.0, axis=0)

# shape: (1, 28, 28, 1)
input_image = np.expand_dims(image, axis=3)
print("input_image: {}, {}".format(type(input_image), input_image.shape))
# show_sample(input_image, ['Input Image: 0'], 1)

interpreter.set_tensor(input_details[0]["index"], input_image)
interpreter.invoke()
output = interpreter.tensor(output_details[0]["index"])()[0]
print("\noutput_image: {}, {}\n {}".format(type(output), output.shape, output))

# Print the model's classification result
digit = np.argmax(output)
print('\nPredicted Digit: %d\nConfidence: %f' % (digit, output[digit]))
