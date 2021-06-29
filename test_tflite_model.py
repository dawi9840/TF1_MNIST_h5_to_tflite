import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import numpy as np
import tensorflow as tf

tflite_model_file = "HDF5/2021.05.04_mnist.tflite"

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()
print(interpreter.get_input_details(), end="\n-----------------------------\n")


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print("input_details:", input_details)
# print("output_details:", output_details, end="\n-----------------------------\n")


# Get inp_index and out_details.
inp_index = input_details[0]["index"]
out_details = interpreter.get_output_details()
# out_details = sorted(output_details, key=lambda k: k['index']) 
# print("inp_index:", inp_index)
print("output_details:", out_details, end="\n-----------------------------\n")


# Test model on random input data.
input_shape = input_details[0]['shape']
print("input_shape:", input_shape)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(inp_index, input_data)

interpreter.invoke()


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("output_data:", output_data, end="\n-----------------------------\n")