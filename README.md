# TF1_MNIST_h5_to_tflite
This is example about using TF1 to train MNIST dataset saving HDF5 parameter and convert to TFLite file with CPU and GPU.  

TensorFlow version is the 1.15, and details with all package name version is in "tf1.15_py37_package_version.txt" file.  

Using python gpu_v2_mnist_keras_train_h5.py can get the result.  

![123333](https://user-images.githubusercontent.com/19554347/116948851-28e2ae80-acb3-11eb-9151-53efa94ee6f2.PNG)

When we have the HDF5 (.h5) file, use reload_h5_file.py to check the model training parameter with a summary.  

![reload](https://user-images.githubusercontent.com/19554347/116949464-dc986e00-acb4-11eb-895e-84972b222175.PNG)

Tflite information:  
We can get the input tensor and output tensor format with tf.lite.Interpreter.  
![Screenshot from 2021-07-13 08-58-10](https://user-images.githubusercontent.com/19554347/125374424-93058700-e3b9-11eb-93c0-3efc7922ebed.png)

Then use a image to verify the TensorFlow Lite model. The result as the show:  
![Screenshot from 2021-07-13 09-02-59](https://user-images.githubusercontent.com/19554347/125374721-29d24380-e3ba-11eb-8f61-e1087f3aaeb8.png)





