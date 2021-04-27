import datetime
import keras
import numpy as np
from keras.applications import mobilenet_v2
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense
from keras.models import Model

img_size = 224
# Since this model will just detect the faces, the output size is 4
# Top-left landmark's coordinate and bottom-right's coordinate
output_size = 4

# For log
start = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# Error detected: 'Object arrays cannot be loaded when allow_pickle=False'
# Preserve the original np.load
# and set the 'allow_pickle' as True
tmp_np_load = np.load
np.load = lambda *a,**k: tmp_np_load(*a, allow_pickle=True, **k)

# Load .npy datasets
print('Loading datasets...')
data_00 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_00.npy')
data_01 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_01.npy')
data_02 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_02.npy')
data_03 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_03.npy')
data_04 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_04.npy')
data_05 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_05.npy')
data_06 = np.load('/Users/lim/Desktop/CatHisterizer/dataset/CAT_06.npy')
print('Data loading finished')

# Read the datasets
# In this exercise, dataset_00 to 05 are assigned for training and only 06 for validation
print('Reading datasets...')
x_train = np.concatenate((data_00.item().get('imgs'),
                          data_01.item().get('imgs'),
                          data_02.item().get('imgs'),
                          data_03.item().get('imgs'),
                          data_04.item().get('imgs'),
                          data_05.item().get('imgs')), axis = 0)
y_train = np.concatenate((data_00.item().get('bbs'),
                          data_01.item().get('bbs'),
                          data_02.item().get('bbs'),
                          data_03.item().get('bbs'),
                          data_04.item().get('bbs'),
                          data_05.item().get('bbs')), axis = 0)
print('Data reading finished.')

# Start building the model
print('Building models...')
x_test = np.array(data_06.item().get('imgs'))
y_test = np.array(data_06.item().get('bbs'))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

# Pretrained model
mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3),
                                 alpha=1.0,
                                 include_top=False,
                                 weights='imagenet',
                                 input_tensor=inputs,
                                 pooling='max')

net = Dense(128, activation='relu')(mobilenet_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()
print('Building finished.')

# Start training
print('Training models...')
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')

model.fit(x_train, y_train,
          epochs=50,
          batch_size=32,
          shuffle=True,
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks= [
              TensorBoard(log_dir='/Users/lim/Desktop/CatHisterizer/logs/%s' % (start)),
              ModelCheckpoint('/Users/lim/Desktop/CatHisterizer/models/%s.h5' % (start), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
              ReduceLROnPlateau(monitor='val_lose', factor=0.2, patience=5, verbose=1, mode='auto')
            ]
          )
print('Training finished.')