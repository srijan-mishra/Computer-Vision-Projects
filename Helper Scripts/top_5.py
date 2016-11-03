#top5 for 256 neurons in final layer

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



train_data = np.load(open('/Users/srijanmishra/Desktop/Acads+Gym/Projects/Leafsnap/leafsnap/Mailed code/170 top 5/bottleneck_features_train.npy'))
train_labels = np.array([0] * (360) + [1] * (225) + [2] * (225) + [3] * (216) + [4] * (198) + [5] * (189) + [6] * (180) + [7] * (180) + [8] * (180) + [9] * (180) + [10] * (171) + [11] * (162) + [12] * (162) + [13] * (162) + [14] * (153) + [15] * (153) + [16] * (153) + [17] * (144) + [18] * (144) + [19] * (144) + [20] * (135) + [21] * (135) + [22] * (126) + [23] * (126) + [24] * (126) + [25] * (126) + [26] * (126) + [27] * (108) + [28] * (108) + [29] * (108) + [30] * (108) + [31] * (108) + [32] * (108) + [33] * (108) + [34] * (108) + [35] * (108) + [36] * (108) + [37] * (108) + [38] * (108) + [39] * (108) + [40] * (108) + [41] * (108) + [42] * (108) + [43] * (108) + [44] * (108) + [45] * (108) + [46] * (108) + [47] * (108) + [48] * (108) + [49] * (108) + [50] * (108) + [51] * (108) + [52] * (108) + [53] * (108) + [54] * (108) + [55] * (108) + [56] * (108) + [57] * (108) + [58] * (108) + [59] * (108) + [60] * (108) + [61] * (108) + [62] * (108) + [63] * (108) + [64] * (108) + [65] * (108) + [66] * (108) + [67] * (108) + [68] * (108) + [69] * (108) + [70] * (108) + [71] * (108) + [72] * (108) + [73] * (108) + [74] * (108) + [75] * (108) + [76] * (108) + [77] * (108) + [78] * (108) + [79] * (108) + [80] * (108) + [81] * (108) + [82] * (108) + [83] * (108) + [84] * (108) + [85] * (108) + [86] * (108) + [87] * (108) + [88] * (108) + [89] * (108) + [90] * (108) + [91] * (108) + [92] * (108) + [93] * (108) + [94] * (108) + [95] * (108) + [96] * (108) + [97] * (108) + [98] * (108) + [99] * (108) + [100] * (108) + [101] * (108) + [102] * (108) + [103] * (108) + [104] * (108) + [105] * (108) + [106] * (108) + [107] * (108) + [108] * (108) + [109] * (108) + [110] * (108) + [111] * (108) + [112] * (108) + [113] * (108) + [114] * (108) + [115] * (108) + [116] * (108) + [117] * (108) + [118] * (108) + [119] * (108) + [120] * (108) + [121] * (108) + [122] * (108) + [123] * (108) + [124] * (108) + [125] * (108) + [126] * (108) + [127] * (108) + [128] * (108) + [129] * (108) + [130] * (108) + [131] * (108) + [132] * (108) + [133] * (108) + [134] * (108) + [135] * (108) + [136] * (108) + [137] * (108) + [138] * (108) + [139] * (108) + [140] * (108) + [141] * (108) + [142] * (108) + [143] * (108) + [144] * (108) + [145] * (108) + [146] * (108) + [147] * (108) + [148] * (108) + [149] * (108) + [150] * (108) + [151] * (99) + [152] * (99) + [153] * (99) + [154] * (99) + [155] * (99) + [156] * (99) + [157] * (99) + [158] * (99) + [159] * (99) + [160] * (99) + [161] * (99) + [162] * (99) + [163] * (99) + [164] * (90) + [165] * (90) + [166] * (90) + [167] * (90) + [168] * (90) + [169] * (72))
t_labels=to_categorical(train_labels)

validation_data = np.load(open('/Users/srijanmishra/Desktop/Acads+Gym/Projects/Leafsnap/leafsnap/Mailed code/170 top 5/bottleneck_features_validation.npy'))
validation_labels = np.array([0] * (40) + [1] * (25) + [2] * (25) + [3] * (24) + [4] * (22) + [5] * (21) + [6] * (20) + [7] * (20) + [8] * (20) + [9] * (20) + [10] * (19) + [11] * (18) + [12] * (18) + [13] * (18) + [14] * (17) + [15] * (17) + [16] * (17) + [17] * (16) + [18] * (16) + [19] * (16) + [20] * (15) + [21] * (15) + [22] * (14) + [23] * (14) + [24] * (14) + [25] * (14) + [26] * (14) + [27] * (12) + [28] * (12) + [29] * (12) + [30] * (12) + [31] * (12) + [32] * (12) + [33] * (12) + [34] * (12) + [35] * (12) + [36] * (12) + [37] * (12) + [38] * (12) + [39] * (12) + [40] * (12) + [41] * (12) + [42] * (12) + [43] * (12) + [44] * (12) + [45] * (12) + [46] * (12) + [47] * (12) + [48] * (12) + [49] * (12) + [50] * (12) + [51] * (12) + [52] * (12) + [53] * (12) + [54] * (12) + [55] * (12) + [56] * (12) + [57] * (12) + [58] * (12) + [59] * (12) + [60] * (12) + [61] * (12) + [62] * (12) + [63] * (12) + [64] * (12) + [65] * (12) + [66] * (12) + [67] * (12) + [68] * (12) + [69] * (12) + [70] * (12) + [71] * (12) + [72] * (12) + [73] * (12) + [74] * (12) + [75] * (12) + [76] * (12) + [77] * (12) + [78] * (12) + [79] * (12) + [80] * (12) + [81] * (12) + [82] * (12) + [83] * (12) + [84] * (12) + [85] * (12) + [86] * (12) + [87] * (12) + [88] * (12) + [89] * (12) + [90] * (12) + [91] * (12) + [92] * (12) + [93] * (12) + [94] * (12) + [95] * (12) + [96] * (12) + [97] * (12) + [98] * (12) + [99] * (12) + [100] * (12) + [101] * (12) + [102] * (12) + [103] * (12) + [104] * (12) + [105] * (12) + [106] * (12) + [107] * (12) + [108] * (12) + [109] * (12) + [110] * (12) + [111] * (12) + [112] * (12) + [113] * (12) + [114] * (12) + [115] * (12) + [116] * (12) + [117] * (12) + [118] * (12) + [119] * (12) + [120] * (12) + [121] * (12) + [122] * (12) + [123] * (12) + [124] * (12) + [125] * (12) + [126] * (12) + [127] * (12) + [128] * (12) + [129] * (12) + [130] * (12) + [131] * (12) + [132] * (12) + [133] * (12) + [134] * (12) + [135] * (12) + [136] * (12) + [137] * (12) + [138] * (12) + [139] * (12) + [140] * (12) + [141] * (12) + [142] * (12) + [143] * (12) + [144] * (12) + [145] * (12) + [146] * (12) + [147] * (12) + [148] * (12) + [149] * (12) + [150] * (12) + [151] * (11) + [152] * (11) + [153] * (11) + [154] * (11) + [155] * (11) + [156] * (11) + [157] * (11) + [158] * (11) + [159] * (11) + [160] * (11) + [161] * (11) + [162] * (11) + [163] * (11) + [164] * (10) + [165] * (10) + [166] * (10) + [167] * (10) + [168] * (10) + [169] * (8))
v_labels=to_categorical(validation_labels)

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(170, activation='softmax'))


accuracies=[]

train_dict=defaultdict(int)

for w in train_labels:
  train_dict[w]+=1

validation_dict=defaultdict(int)
for w in validation_labels:
  validation_dict[w]+=1

ratio=(train_dict[0]+validation_dict[0])/validation_dict[0]
index_dict=dict()
val_index_dict=dict()
index_dict[-1]=0
val_index_dict[-1]=0

for j in train_dict:
    index_dict[j]=0
    for k in range(0,j+1):
        index_dict[j]=index_dict[j]+train_dict[k]

for j in validation_dict:
    val_index_dict[j]=0
    for k in range(0,j+1):
        val_index_dict[j]=val_index_dict[j]+validation_dict[k]

for k in range(10):
    weights_path='/Users/srijanmishra/Desktop/Acads+Gym/Projects/Leafsnap/leafsnap/Mailed code/170 top 5/weights/256_'+str(k)+'.h5'
    model.load_weights(weights_path)
    y_pred=model.predict_proba(validation_data)
    a=np.zeros(2198*5).reshape(2198,5)
    for i in range(y_pred.shape[0]):
        a[i]=y_pred[i].argsort()[-5:][::-1]

    count=0
    for i in range(a.shape[0]):
        if validation_labels[i] in a[i]:
            count=count+1
    accuracy=(count*1.0)/a.shape[0]

    accuracies.append(accuracy)
    if k<9:
        for j in range(len(index_dict)-1):
            train_data[index_dict[j-1]+k*validation_dict[j]:index_dict[j-1]+(k+1)*validation_dict[j],:,:,:]=train_data[index_dict[j-1]+k*validation_dict[j]:index_dict[j-1]+(k+1)*validation_dict[j],:,:,:]+validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]
            validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]=train_data[index_dict[j-1]+k*validation_dict[j]:index_dict[j-1]+(k+1)*validation_dict[j],:,:,:]-validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]
            train_data[index_dict[j-1]+k*validation_dict[j]:index_dict[j-1]+(k+1)*validation_dict[j],:,:,:]=train_data[index_dict[j-1]+k*validation_dict[j]:index_dict[j-1]+(k+1)*validation_dict[j],:,:,:]-validation_data[val_index_dict[j-1]:val_index_dict[j],:,:,:]
