from __future__ import division, print_function, absolute_import
import os, os.path
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import sys
import numpy as np 


def load_files(base_dir, split_name):
    fibers_name = base_dir + split_name + '_fibers.npy'
    labels_name = base_dir + split_name + '_labels.npy'

    fibers = np.load(fibers_name)
    labels = np.load(labels_name)
    return fibers, labels
    
batch_size = 500
num_points = 50
num_fiber_bundles = 17
num_epochs = 1 
base_dir = sys.argv[1]

out_dir = base_dir + 'CNN_model/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir) 

print('Reading data')
train_fibers, train_labels = load_files (base_dir, 'train')
test_fibers, test_labels = load_files (base_dir, 'test')
valid_fibers, valid_labels = load_files(base_dir, 'valid')
print('Data reading complete')

## Network architecture
network = input_data(shape = [None, num_points, 3, 1], name= 'input')

network = conv_2d(network, 32, 3, activation = 'relu', regularizer ='L2')
network = local_response_normalization(network)

network = conv_2d(network, 64, 3, activation = 'relu', regularizer ='L2')
network = local_response_normalization(network)

network = fully_connected(network, 128, activation = 'tanh')
network = dropout(network, 0.8)
 
network = fully_connected(network, 256, activation = 'tanh')
network = dropout(network, 0.8)

network = fully_connected(network, num_fiber_bundles, activation = 'softmax')
network = regression(network, optimizer = 'adam', learning_rate = 0.0001, loss='categorical_crossentropy', name='target')
print('Architecutr definition complete')
## Network architecture completed


## Generate Batches. Actually no need to generate batches just give the whole thing 
train_data = train_fibers.reshape([-1, num_points, 3, 1])
valid_data = valid_fibers.reshape([-1, num_points, 3, 1])

print(train_data.shape)
print(valid_data.shape)

print(train_data[1,:])
print(valid_data[1,:])

## Start training
model = tflearn.DNN(network, tensorboard_verbose = 0)
mode_name = out_dir + 'CNN_conv2d_32_conv2d_64_fc_128_fc_256_epoch_' + str(num_epochs) + '.tflearn'

model.fit({'input': train_data}, {'target': train_labels}, n_epoch= num_epochs,validation_set=({'input': valid_data}, {'target': valid_labels}), snapshot_step=100, show_metric=True, batch_size=3000, run_id='convnet_track')
model.save(model_name)
print('done')
