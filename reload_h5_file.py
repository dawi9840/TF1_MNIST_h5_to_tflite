import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import load_model

backend.clear_session()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,
                                                            allow_soft_placement=True,
                                                            log_device_placement=True)
                                                            )

backend.set_session(sess)


with tf.device('/gpu:0'):
    reload_model = load_model('./HDF5/2021.05.04_mnist.h5')
    reload_model.summary()

