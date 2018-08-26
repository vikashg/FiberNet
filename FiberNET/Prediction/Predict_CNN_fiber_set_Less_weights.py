from __future__ import division, print_function, absolute_import
import os, os.path 
import tflearn 
from sklearn.metrics import confusion_matrix
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization 
from tflearn.layers.estimator import regression 
import sys 
import numpy as np 

def load_files_evaluate(data_dir, fiber_grp):
    fiber_name = data_dir + fiber_grp + '_global_test_fibers.npy'
    label_name = data_dir + fiber_grp + '_global_test_labels.npy'
    
    fibers= np.load(fiber_name)
    labels = np.load(label_name)

    return fibers, labels

def load_files_predict():
    placeholder = 0 

data_dir = sys.argv[1]
model_dir = sys.argv[2]
fiber_grp = sys.argv[3]
out_dir = sys.argv[4]
flag = sys.argv[5]
num_epoch = sys.argv[6]

num_points = 50 
num_fiber_bundles = 2

if (flag == 'eval'):
    test_fibers, true_labels = load_files_evaluate(data_dir, fiber_grp)
    test_fibers = test_fibers.reshape([-1, num_points, 3, 1])


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

network = input_data(shape = [None, num_points, 3, 1], name= 'input')

network = conv_2d(network, 16, 3, activation = 'relu', regularizer ='L2')
network = local_response_normalization(network)

network = conv_2d(network, 32, 3, activation = 'relu', regularizer ='L2')
network = local_response_normalization(network)

network = fully_connected(network, 64, activation = 'tanh')
network = dropout(network, 0.8)

network = fully_connected(network, 128, activation = 'tanh')
network = dropout(network, 0.8)

network = fully_connected(network, num_fiber_bundles, activation = 'softmax')
network = regression(network, optimizer = 'adam', learning_rate = 0.0001, loss='categorical_crossentropy', name='target')
print('Architecutr definition complete')
## Network architecture completed

model = tflearn.DNN(network)
num_fibers_test_case = test_fibers.shape[0]

sum_a = np.zeros([num_fibers_test_case, 2])
for i in range(20):
    model_name = model_dir + fiber_grp + '/Model_' + str(i+1) + '/CNN_conv2d_16_conv2d_32_fc_64_fc_128_epoch_' + str(num_epoch) + '.tflearn'
    model.load(model_name)
    a = model.predict(test_fibers)
    sum_a += a
    print(i+1)

avg_prediction=sum_a/20
predict_val = np.argmax(avg_prediction, axis = 1)
true_val = np.argmax(true_labels, axis = 1)

cf_mat = confusion_matrix(true_val, predict_val)
cm = cf_mat.astype('float') / cf_mat.sum(axis=1)[:, np.newaxis]
print(cm)

output_file_name = model_dir + fiber_grp + '_cnf_matrix.txt'
print(output_file_name)
np.savetxt(output_file_name, cm) 


