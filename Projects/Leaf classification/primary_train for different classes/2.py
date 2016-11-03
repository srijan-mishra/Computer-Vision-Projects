import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import accuracy_score

weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 585
nb_validation_samples = 65
nb_epoch = 30


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1.)
    datagen.mean=np.array([103.939,116.779,123.68],dtype=np.float32).reshape(3,1,1)


    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))) #3*152*152

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1')) #64*150*150
    model.add(ZeroPadding2D((1, 1))) #64*152*152
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2')) #64*150*150
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #64*75*75

    model.add(ZeroPadding2D((1, 1))) #64*77*77
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1')) #128*75*75
    model.add(ZeroPadding2D((1, 1))) #64*77*77
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2')) #128*75*75
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #128*37*37

    model.add(ZeroPadding2D((1, 1))) #128*39*39
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1')) #256*37*37
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2')) #256*37*37
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3')) #256*37*37
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #256*18*18

    model.add(ZeroPadding2D((1, 1))) #256*20*20
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1')) #512*18*18
    model.add(ZeroPadding2D((1, 1))) #512*20*20
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2')) #512*18*18
    model.add(ZeroPadding2D((1, 1))) #512*20*20
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3')) #512*18*18
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #512*9*9

    model.add(ZeroPadding2D((1, 1))) #512*11*11
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1')) #512*9*9
    model.add(ZeroPadding2D((1, 1))) #512*11*11
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2')) #512*9*9
    model.add(ZeroPadding2D((1, 1))) #512*11*11
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3')) #512*9*9
    model.add(MaxPooling2D((2, 2), strides=(2, 2))) #512*4*4


    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model(num_neurons):
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (360) + [1] * (225))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (40) + [1] * (25))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    y_pred=model.predict_classes(validation_data)
    acc= accuracy_score(validation_labels,y_pred)
    return acc





#save_bottlebeck_features()
#train_top_model(num_neurons)
