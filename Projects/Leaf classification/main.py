#second script
from primary_train import save_bottlebeck_features
from primary_train import train_top_model


d=dict()
num_neurons=[32,64,128,256,512]

save_bottlebeck_features()


for i in range(len(num_neurons)):
    accuracy=train_top_model(num_neurons[i])
    d[num_neurons[i]]=accuracy[num_neurons[i]]
    print '\nDone for %d'%(i)
