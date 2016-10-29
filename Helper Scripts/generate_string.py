#query to generate strings for train_labels and test_labels in primary_train.py

import csv
file=open('test.csv','rU')
reader=csv.reader(file)
j=[]

for row in reader:
  j.append(row)


q='[0] * (360) + [1] * (225)'

for i in range(3,171):
  q=q+' + ['+str(int(j[i][6])-1)+'] * ('+str(j[i][4])+')'

p='[0] * (40) + [1] * (25)'
for i in range(3,171):
  p=p+' + ['+str(int(j[i][6])-1)+'] * ('+str(j[i][5])+')'
