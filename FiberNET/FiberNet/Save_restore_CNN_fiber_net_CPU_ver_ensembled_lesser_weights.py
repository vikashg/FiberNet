from __future__ import division, print_function, absolute_import
import os, os.path
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import sys
import numpy as np 
import Select_Random_dataset as srd

def load_files(base_dir, fiber_grp, split_name):
    fibers_name = base_dir + fiber_grp + '_'  + split_name + '_fibers.npy'
    labels_name = base_dir + fiber_grp + '_' + split_name + '_labels.npy'

    fibers = np.load(fibers_name)
    labels = np.load(labels_name)
    return fibers, labels
    
batch_size = 2000
num_points = 50
num_epochs = 80
base_dir = sys.argv[1]
fiber_grp = sys.argv[2]
model_num = int(sys.argv[3])
split_ratio = 0.8
resample_ratio =0.7

out_dir = base_dir + 'CNN_model_lesser_w/CPU/' + fiber_grp + '/Model_' + str(model_num) + '/'

TensorBoard_dir = base_dir + 'CNN_model_lesser_w/CPU/TensorBoard_dir/'+ fiber_grp  +'/Model_' + str(model_num) +'/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir) 

if not os.path.exists(TensorBoard_dir):
    os.makedirs(TensorBoard_dir)

print('Reading data')
global_fibers, global_labels = load_files (base_dir, fiber_grp, 'global_train')
print('Data reading complete')
num_fiber_bundles = global_labels.shape[1]
num_fibers = global_labels.shape[0] 
#Extract random sampled
train_fibers, train_labels, valid_fibers, valid_labels = srd.SelectRandomData(global_fibers, global_labels, int(resample_ratio*num_fibers), split_ratio)

'''
train_fibers = train_fibers[0:batch_size,:]
train_labels = train_labels[0:batch_size,:]
valid_fibers = valid_fibers[0:batch_size,:]
valid_labels = valid_labels[0:batch_size,:]
'''

print(train_fibers.shape)

## Network architecture
network = input_data(shape = [None, num_points, 3, 1], name= 'input')

network = conv_2d(network, 8, 3, activation = 'relu', regularizer ='L2')
network = local_response_normalization(network)

network = conv_2d(network, 16, 3, activation = 'relu', regularizer ='L2')
network = local_response_normalization(network)

network = fully_connected(network, 32, activation = 'tanh')
network = dropout(network, 0.8)
 
network = fully_connected(network, 64, activation = 'tanh')
network = dropout(network, 0.8)

network = fully_connected(network, num_fiber_bundles, activation = 'softmax')
network = regression(network, optimizer = 'adam', learning_rate = 0.0001, loss='categorical_crossentropy', name='target')
print('Architecutr definition complete')
## Network architecture completed

## Generate Batches. Actually no need to generate batches just give the whole thing 
train_data = train_fibers.reshape([-1, num_points, 3, 1])
valid_data = valid_fibers.reshape([-1, num_points, 3, 1])

## Start training
model = tflearn.DNN(network, tensorboard_verbose = 3, tensorboard_dir = TensorBoard_dir)
model_name = out_dir + 'CNN_conv2d_16_conv2d_32_fc_64_fc_128_epoch_' + str(num_epochs) + '.tflearn'

model.fit({'input': train_data}, {'target': train_labels}, n_epoch= num_epochs,validation_set=({'input': valid_data}, {'target': valid_labels}), snapshot_step=100, show_metric=True, batch_size=batch_size, run_id='convnet_track')
model.save(model_name)
print('done')
