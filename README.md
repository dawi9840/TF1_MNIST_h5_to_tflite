# TF1_MNIST_h5_to_tflite
Use TF1 to train MNIST dataset saving HDF5 parameter and convert to TFLite file.

TensorFlow version is the 1.15, and details with all package name version is in "tf1.15_py37_package_version.txt" file.

Using python gpu_v2_mnist_keras_train_h5.py can get the result.

![123333](https://user-images.githubusercontent.com/19554347/116948851-28e2ae80-acb3-11eb-9151-53efa94ee6f2.PNG)

When we have the HDF5 (.h5) file, use reload_h5_file.py to check the model training parameter with a summary.

![reload](https://user-images.githubusercontent.com/19554347/116949464-dc986e00-acb4-11eb-895e-84972b222175.PNG)
