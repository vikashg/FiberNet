import numpy as np 
import os 
import sys
from sklearn.utils import shuffle 

def save_fiber_labels(base_dir, fiber_grp, split_name, fibers, labels):
    fiber_name = base_dir + fiber_grp + '_' + split_name + '_fibers.npy'
    label_name = base_dir + fiber_grp + '_' + split_name + '_labels.npy'
    print(fiber_name)
    print(label_name)    
    np.save(fiber_name, fibers)
    np.save(label_name, labels)    


#Read files 
base_dir=sys.argv[1]
split_ratio=float(sys.argv[2]) #enter the fraction of training data rest is divided equally into test and validation sets 
fiber_grp = sys.argv[3]
out_dir = sys.argv[4]

all_fibers_file = base_dir + fiber_grp + '_global_train_fibers.npy'
all_labels_file = base_dir + fiber_grp + '_global_train_labels.npy'

all_fibers = np.load(all_fibers_file)
all_labels = np.load(all_labels_file)

num_fibers = all_fibers.shape[0]
num_labels = all_labels.shape[0]

num_train = int(split_ratio*num_fibers)

#Shuffling the data
for i in range(100):
    all_fibers, all_labels = shuffle(all_fibers, all_labels, random_state=0)


all_fibers_shuffled = all_fibers
all_labels_shuffled = all_labels

train_fibers = all_fibers_shuffled[0:num_train,:]
train_labels = all_labels_shuffled[0:num_train,:]

test_fibers = all_fibers_shuffled[num_train:-1, :]
test_labels = all_labels_shuffled[num_train:-1, :]

save_fiber_labels(out_dir, fiber_grp, 'global_train', train_fibers, train_labels)
save_fiber_labels(out_dir, fiber_grp, 'global_test', test_fibers, test_labels)

