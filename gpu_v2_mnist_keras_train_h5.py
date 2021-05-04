import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend
from keras import layers
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import TensorBoard

EPOCH = 10
BATCH_SIZE = 1200

#清除暫存
backend.clear_session()

#只使用40%的GPU記憶體
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options,     #啟用gpu控制選項
                                                            allow_soft_placement=True,   #如果指定的設備不存在，允許TF自動分配設備
                                                            log_device_placement=True))  #是否檢視設備分配日誌資訊

#設定Keras使用的TensorFlow Session
backend.set_session(sess)

tbCallBack = TensorBoard(log_dir='./logs',             #模型保存目錄
                         histogram_freq=0,             #是否計算直方圖，0表示不計算
                         batch_size=32,                #可以進行直方圖計算的預期神經元網絡輸入批的大小
                         write_graph=True,             #是否儲存計算圖
                         write_grads=False,            #是否可視化梯度梯度直方圖，histogram_freq 必须要大於 0
                         write_images=True,            #將模型權重以圖片可視化參數
                         embeddings_freq=0,            #frequency (in epochs) at which embedding layers will be visualized. If set to 0, embeddings will not be visualized.
                         embeddings_layer_names=None,  #如果是None或空列表，那麼所有的嵌入層都會被監測。
                         embeddings_metadata=None,     #一個dictionary 將圖層名稱映射到文件名，該嵌入層的元數據保存在該文件名中。
                         update_freq='epoch')          #選用epoch次數更新TensorBoard的紀錄

def show_train_history(train_history,train,validation):
    #顯示損失率與精確度
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Training history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

def show_multi_image(images, labels, prediction, idx, num=10):
    #畫出原始圖形、label、預測結果，idx:第0筆開始顯示，預設秀10筆。
    flg=plt.gcf()
    flg.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title="label=" + str(labels[idx])
        if len(prediction)>0:
            title+=",predict=" + str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()

with tf.device('/cpu:0'):
    # (train, val_train),(label, val_test) = mnist.load()
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    ### Dataset preprocess. ###
    # 原始資料的數值型態為uint8，數值分布範圍是0~255[0, 2^8 - 1]，必須先調整到 0~1 之間的浮點數，才能輸入神經網路加以訓練。
    # 資料正規化數值到[0, 1]區間: 將數值28x28x1x60000四維陣列轉成浮點數 (float32)，並除以 255.0
    x_train = (x_train.reshape(60000,28,28,1).astype('float32'))/255.0
    x_test = (x_test.reshape(10000,28,28,1).astype('float32'))/255.0

    #驗證用資料集因為不用經過神經網路訓練，所以我推測直接轉one-hot encoding
    #to_categorical轉換成one-hot encoding，類別向量轉換為二進制（只有0和1）的矩陣類型表示。
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Use Keras to creat model.
    model = Sequential([
      layers.Conv2D(filters=16,kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='relu'),
      layers.MaxPool2D(pool_size=(2,2)),
      layers.Conv2D(filters=36, kernel_size=(5,5), padding='same', activation='relu'),
      layers.MaxPool2D(pool_size=(2,2)),
      layers.Dropout(rate=0.25),
      layers.Flatten(),
      layers.Dense(units=128, kernel_initializer='normal', activation='relu'),
      layers.Dropout(rate=0.5),
      layers.Dense(units=10, kernel_initializer='normal', activation='softmax')
      ])

with tf.device('/gpu:0'):
    model.compile(
      optimizer='adam', 
      loss='categorical_crossentropy', 
      metrics=['accuracy'])

    # Start training.
    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCH,
                        verbose=2,
                        callbacks=[tbCallBack],
                        validation_split=0.2)

    # 存訓練好參數model的HDF5檔案
    model.save('./HDF5/2021.05.04_mnist.h5')
    
    # Use test data to calculate accuracy.
    scores = model.evaluate(x_test, y_test)
    
    # [0]:loss rate, [1]:accuracy.
    print('Accuracy=', scores[1])

    backend.clear_session()

    #suppress:是否壓縮由科學計數法表示的浮點數
    np.set_printoptions(suppress=True)

    ### Convert to TFLite. ###
    input_graph_name = './HDF5/2021.05.04_mnist.h5'

    output_graph_name = input_graph_name[:-3] + '.tflite'

    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file=input_graph_name)

    converter.post_training_quantize = True

    tflite_model = converter.convert()

    open(output_graph_name, "wb").write(tflite_model)

    print ("generate:",output_graph_name)

print("finished!")

### In your Terminal key the command to start TensorBoard. ###
# tensorboard --logdir = [Your project path./logs]





