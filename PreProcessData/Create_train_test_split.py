import numpy as np 
import os 
import sys
from sklearn.utils import shuffle 

def save_fiber_labels(base_dir, split_name, fibers, labels):
    fiber_name = base_dir+ split_name + '_fibers.npy'
    label_name = base_dir + split_name + '_labels.npy'
    
    np.save(fiber_name, fibers)
    np.save(label_name, labels)    


#Read files 
base_dir=sys.argv[1]
split_ratio=float(sys.argv[2]) #enter the fraction of training data rest is divided equally into test and validation sets 


all_fibers_file = base_dir + 'All_training_fibers.npy'
all_labels_file = base_dir + 'All_training_labels.npy'

all_fibers = np.load(all_fibers_file)
all_labels = np.load(all_labels_file)

num_fibers = all_fibers.shape[0]
num_labels = all_labels.shape[0]

num_train = int(split_ratio*num_fibers)
remain_items = num_fibers - num_train

num_test = int(0.5*remain_items)
num_valid = num_fibers - num_train - num_test

#Shuffling the data
all_fibers_shuffled, all_labels_shuffled = shuffle(all_fibers, all_labels, random_state=0)

train_fibers = all_fibers_shuffled[0:num_train,:]
train_labels = all_labels_shuffled[0:num_train,:]

test_fibers = all_fibers_shuffled[num_train:num_train+num_test, :]
test_labels = all_labels_shuffled[num_train: num_train+num_test, :]

valid_fibers = all_fibers_shuffled[num_train+num_test:-1, :]
valid_labels = all_labels_shuffled[num_train+num_test:-1, :]

save_fiber_labels(base_dir, 'train', train_fibers, train_labels)
save_fiber_labels(base_dir, 'test', test_fibers, test_labels)
save_fiber_labels(base_dir, 'valid', valid_fibers, valid_labels)

