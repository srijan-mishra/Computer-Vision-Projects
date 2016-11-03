# Computer-Vision-Projects

Contains the following:

1) Computer Vision Projects:

       1.1 Leaf Classification: A fine grained classification problem on the Leafsnap datset. The dataset consists of images
       of 23147 Lab images belonging to 185 tree species.
       Link: http://leafsnap.com/
       Reference: Leafsnap: A Computer Vision System for Automatic Plant Species Identification.


2) Helper functions:

       2.1 copy_paste script: For directory level manipulations to get images from the main dataset and putting them
       into test, validation and specimen folders. The script reads from 'test.csv' which has the same format as
       https://github.com/srijan-mishra/Computer-Vision-Projects/blob/master/Projects/Leafsnap%20Dataset/species%20used.csv .

       2.2 generate_string.py: To generate the two strings to put in primary_train.py for different number of train/test
       cases.

       2.3 top_5.py: Script to use saved weights to get top 5 accuracy on a test dataset.
