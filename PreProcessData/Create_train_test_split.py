import numpy as np 
import os 
import sys

#Read files 
base_dir=sys.argv[1]
split_ratio=float(sys.argv[2]) #enter the fraction of training data rest is divided equally into test and validation sets 


all_fibers_file = base_dir + 'All_training_fibers.npy'
all_labels_file = base_dir + 'All_training_labels.npy'

all_fibers = np.load(all_fibers_file)
all_labels = np.load(all_labels_file)

num_fibers = all_fibers.shape[0]
print(num_fibers)
