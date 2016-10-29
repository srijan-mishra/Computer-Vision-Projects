# Leafsnap Dataset

1) VGG weights download link(https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

2) Here, VGG has been used as a fixed feature extractor. In the first part(save_bottlenecl_features) of primary_train.py, vgg scores are calculated for the train and test images in the dataset. In the second part, the fully connected layer is trained for different number of neurons-[32,64,128,256,512] for 30 epochs and accuracies are calculated.

3) Models were trained for incereasing number of classes- [30,50,70,90,110,130,150,170]. Following is the graph for the maximum accuracy:

