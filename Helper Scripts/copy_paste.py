import csv
import os
import shutil
import random

file=open('test.csv','rU')
reader=csv.reader(file)
j=dict()


for row in reader:
    j[row[2]]=row[3]

d=[]
file=open('test.csv','rU')
reader=csv.reader(file)
for row in reader:
    d.append(row[2])


source='/Users/srijanmishra/Desktop/leafsnap/Mailed code/170 classes/data/lab images'
destination='/Users/srijanmishra/Desktop/leafsnap/Mailed code/170 classes/data/train'
val_dest='/Users/srijanmishra/Desktop/leafsnap/Mailed code/170 classes/data/validation'
specimen_dest='/Users/srijanmishra/Desktop/leafsnap/Mailed code/170 classes/data/specimen'

for filename in os.listdir(source):
            if filename in d[151:171]:
                shutil.copytree(source+'/'+filename, destination+'/'+filename)
                os.chdir(destination+'/'+filename)
                for var in range(int(j[filename])%10):
                    f=random.choice(os.listdir(destination+'/'+filename))
                    os.remove(f)
                os.makedirs(val_dest+'/'+filename)
                os.makedirs(specimen_dest+'/'+filename)
                for var in range(int(j[filename])/10):
                    f=random.choice(os.listdir(destination+'/'+filename))
                    shutil.move(destination+'/'+filename+'/'+f,val_dest+'/'+filename+'/'+f)
                f=random.choice(os.listdir(destination+'/'+filename))
                shutil.copy(destination+'/'+filename+'/'+f,specimen_dest+'/'+filename+'/'+f)
