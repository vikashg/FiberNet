import numpy as np 
import os
import sys
from sklearn.utils import shuffle
import time

def SelectRandomData( fiber, labels, num_to_be_selected, split_ratio):
    # num =  number of fibers to be selected
    total_num_fibers = fiber.shape[0]
    random_indices = np.random.choice(total_num_fibers, size = num_to_be_selected, replace = 'TRUE')
    
    selected_fibers = fiber[random_indices,:]
    selected_labels = labels[random_indices,:]
    print(total_num_fibers)
    print(num_to_be_selected)

    fibers_shuffled, labels_shuffled = shuffle(selected_fibers, selected_labels, random_state =  int(time.time()))
    
    num_train  = int(split_ratio*num_to_be_selected)
    remain_items = num_to_be_selected - num_train
    num_test = int(0.5*(remain_items))
    num_valid = num_to_be_selected - num_train - num_test 

    fibers_train = fibers_shuffled[0:num_train, :]
    labels_train = labels_shuffled[0:num_train, :]

    fibers_valid = fibers_shuffled[num_train:-1,:]
    labels_valid = labels_shuffled[num_train:-1,:]
    
    return fibers_train,  labels_train, fibers_valid, labels_valid
