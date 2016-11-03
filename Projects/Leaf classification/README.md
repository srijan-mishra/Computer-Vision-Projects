# Leafsnap Dataset

###Introduction:

The leaf classification project is an attempt at solving a fine-grained classification problem. A fine-grained classification problem is aimed at successfully differentiating between sub-classes of a particular species(leaves in this case). Another example of a fine-grained classification problem could be differentiating between various species of birds.

###Explanation:

1) VGG weights download link(https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

2) Here, VGG has been used as a fixed feature extractor. In the first part(save_bottleneck_features) of primary_train.py, vgg scores are calculated for the train and test images in the dataset. In the second part, the fully connected layer is trained for different number of neurons-[32,64,128,256,512] for 30 epochs and accuracies are calculated.

3) Models were trained for increasing number of classes- [30,50,70,90,110,130,150,170]. Following is the graph for the maximum accuracy:

![alt tag](https://github.com/srijan-mishra/Computer-Vision-Projects/blob/master/Projects/Leafsnap%20Dataset/graphs/Graph1.png)

###Files and folders:

1) main.py: Self-explanatory.

2) primary_train.py: Contains the main functions that are run by main.py.

3) primary_train for different classes: Contains variations of primary_train.py for different number of classes

4) species_used.csv: Contains details of species and numbers used for anaylsis.
