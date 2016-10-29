#second mail
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from collections import defaultdict


weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 13545
nb_validation_samples = 1505
nb_epoch = 30
#num_neurons=64


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1.)
    datagen.mean=np.array([103.939,116.779,123.68],dtype=np.float32).reshape(3,1,1)


    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


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
    train_labels = np.array([0] * (360) + [1] * (225) + [2] * (225) + [3] * (216) + [4] * (198) + [5] * (189) + [6] * (180) + [7] * (180) + [8] * (180) + [9] * (180) + [10] * (171) + [11] * (162) + [12] * (162) + [13] * (162) + [14] * (153) + [15] * (153) + [16] * (153) + [17] * (144) + [18] * (144) + [19] * (144) + [20] * (135) + [21] * (135) + [22] * (126) + [23] * (126) + [24] * (126) + [25] * (126) + [26] * (126) + [27] * (108) + [28] * (108) + [29] * (108) + [30] * (108) + [31] * (108) + [32] * (108) + [33] * (108) + [34] * (108) + [35] * (108) + [36] * (108) + [37] * (108) + [38] * (108) + [39] * (108) + [40] * (108) + [41] * (108) + [42] * (108) + [43] * (108) + [44] * (108) + [45] * (108) + [46] * (108) + [47] * (108) + [48] * (108) + [49] * (108) + [50] * (108) + [51] * (108) + [52] * (108) + [53] * (108) + [54] * (108) + [55] * (108) + [56] * (108) + [57] * (108) + [58] * (108) + [59] * (108) + [60] * (108) + [61] * (108) + [62] * (108) + [63] * (108) + [64] * (108) + [65] * (108) + [66] * (108) + [67] * (108) + [68] * (108) + [69] * (108) + [70] * (108) + [71] * (108) + [72] * (108) + [73] * (108) + [74] * (108) + [75] * (108) + [76] * (108) + [77] * (108) + [78] * (108) + [79] * (108) + [80] * (108) + [81] * (108) + [82] * (108) + [83] * (108) + [84] * (108) + [85] * (108) + [86] * (108) + [87] * (108) + [88] * (108) + [89] * (108) + [90] * (108) + [91] * (108) + [92] * (108) + [93] * (108) + [94] * (108) + [95] * (108) + [96] * (108) + [97] * (108) + [98] * (108) + [99] * (108) + [100] * (108) + [101] * (108) + [102] * (108) + [103] * (108) + [104] * (108) + [105] * (108) + [106] * (108) + [107] * (108) + [108] * (108) + [109] * (108))
    t_labels=to_categorical(train_labels)

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (40) + [1] * (25) + [2] * (25) + [3] * (24) + [4] * (22) + [5] * (21) + [6] * (20) + [7] * (20) + [8] * (20) + [9] * (20) + [10] * (19) + [11] * (18) + [12] * (18) + [13] * (18) + [14] * (17) + [15] * (17) + [16] * (17) + [17] * (16) + [18] * (16) + [19] * (16) + [20] * (15) + [21] * (15) + [22] * (14) + [23] * (14) + [24] * (14) + [25] * (14) + [26] * (14) + [27] * (12) + [28] * (12) + [29] * (12) + [30] * (12) + [31] * (12) + [32] * (12) + [33] * (12) + [34] * (12) + [35] * (12) + [36] * (12) + [37] * (12) + [38] * (12) + [39] * (12) + [40] * (12) + [41] * (12) + [42] * (12) + [43] * (12) + [44] * (12) + [45] * (12) + [46] * (12) + [47] * (12) + [48] * (12) + [49] * (12) + [50] * (12) + [51] * (12) + [52] * (12) + [53] * (12) + [54] * (12) + [55] * (12) + [56] * (12) + [57] * (12) + [58] * (12) + [59] * (12) + [60] * (12) + [61] * (12) + [62] * (12) + [63] * (12) + [64] * (12) + [65] * (12) + [66] * (12) + [67] * (12) + [68] * (12) + [69] * (12) + [70] * (12) + [71] * (12) + [72] * (12) + [73] * (12) + [74] * (12) + [75] * (12) + [76] * (12) + [77] * (12) + [78] * (12) + [79] * (12) + [80] * (12) + [81] * (12) + [82] * (12) + [83] * (12) + [84] * (12) + [85] * (12) + [86] * (12) + [87] * (12) + [88] * (12) + [89] * (12) + [90] * (12) + [91] * (12) + [92] * (12) + [93] * (12) + [94] * (12) + [95] * (12) + [96] * (12) + [97] * (12) + [98] * (12) + [99] * (12) + [100] * (12) + [101] * (12) + [102] * (12) + [103] * (12) + [104] * (12) + [105] * (12) + [106] * (12) + [107] * (12) + [108] * (12) + [109] * (12))
    v_labels=to_categorical(validation_labels)

    #getting dictionary values
    train_dict=defaultdict(int)

    for w in train_labels:
      train_dict[w]+=1

    validation_dict=defaultdict(int)
    for w in validation_labels:
      validation_dict[w]+=1

    #defining the FC model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(110, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    ratio=(train_dict[0]+validation_dict[0])/validation_dict[0]
    accuracy=dict()
    index_dict=dict()
    val_index_dict=dict()
    index_dict[-1]=0
    val_index_dict[-1]=0
    accuracy[num_neurons]=[]

    for j in train_dict:
        index_dict[j]=0
        for k in range(0,j+1):
            index_dict[j]=index_dict[j]+train_dict[k]

    for j in validation_dict:
        val_index_dict[j]=0
        for k in range(0,j+1):
            val_index_dict[j]=val_index_dict[j]+validation_dict[k]


    for i in range(ratio):
        model.fit(train_data, t_labels,nb_epoch=nb_epoch, batch_size=32,validation_data=(validation_data, v_labels))
        y_pred=model.predict_classes(validation_data)
        acc= accuracy_score(validation_labels,y_pred)
        accuracy[num_neurons].append(acc)
        if i<9:
            for j in range(len(index_dict)-1):
                train_data[index_dict[j-1]+i*validation_dict[j]:index_dict[j-1]+(i+1)*validation_dict[j],:,:,:]=train_data[index_dict[j-1]+i*validation_dict[j]:index_dict[j-1]+(i+1)*validation_dict[j],:,:,:]+validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]
                validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]=train_data[index_dict[j-1]+i*validation_dict[j]:index_dict[j-1]+(i+1)*validation_dict[j],:,:,:]-validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]
                train_data[index_dict[j-1]+i*validation_dict[j]:index_dict[j-1]+(i+1)*validation_dict[j],:,:,:]=train_data[index_dict[j-1]+i*validation_dict[j]:index_dict[j-1]+(i+1)*validation_dict[j],:,:,:]-validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]
        model.save_weights('/Users/srijanmishra/Desktop/leafsnap/Mailed code/110 classes/weights/'+str(num_neurons)+'_'+str(i))

    return accuracy





#save_bottlebeck_features()
#train_top_model(num_neurons)
