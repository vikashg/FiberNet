import numpy as np 
import sys 
import os


##Make changes here for directories
training_subj_file = sys.argv[1]
out_dir=sys.argv[2]

base_dir = sys.argv[3]
fiber_grp_file=sys.argv[4]
sub_dir=sys.argv[5]

training_subj_list = open(training_subj_file, 'r').readlines()
fiber_grps = open(fiber_grp_file, 'r').readlines()

def readSingleFile(file_name):
    '''
    This function may need modification based on better understanding of the reshape function.
    '''
    mapped_fiber = np.loadtxt(file_name)
    fiber_bundle_shape = mapped_fiber.shape
    num_fibers = fiber_bundle_shape[0]
    num_points = int(fiber_bundle_shape[1]/3)

    X_FIBER = np.zeros([num_fibers, num_points*3])
    for i in range(num_fibers):
        a = mapped_fiber[i,]
        b = np.zeros([num_points, 3])
        b[:,0] = a[0:50]
        b[:,1] = a[50:100]
        b[:,2] = a[100:150]
        x_fiber = b.flatten()
        X_FIBER[i,:] = x_fiber

    return X_FIBER, num_fibers

for fib in fiber_grps:
    
    ALL_FIBERS_GRP = np.zeros([1,150])
    ALL_TRAINING_LABELS = np.zeros([1, 2])

    file_name_kept = base_dir + sub_dir + 'Mapped_fiber_' + fib.strip('\n') + '_cleaned.txt'
    file_name_removed = base_dir + sub_dir + 'Mapped_fiber_' + fib.strip('\n') + '_removed.txt'

    X_FIBER, num_fiber = readSingleFile(file_name_kept)
    count = 0
    label_matrix = np.zeros([num_fiber, 2])
    label_matrix[:,count] = 1

    ALL_FIBERS_GRP = np.row_stack((ALL_FIBERS_GRP, X_FIBER))
    ALL_TRAINING_LABELS = np.row_stack((ALL_TRAINING_LABELS, label_matrix))

    count = 1 
    X_FIBER_noisy, num_noisy_fibers = readSingleFile(file_name_removed)
    label_matrix = np.zeros([num_noisy_fibers, 2])
    label_matrix[:,count] = 1
    ALL_FIBERS_GRP = np.row_stack((ALL_FIBERS_GRP, X_FIBER_noisy))
    ALL_TRAINING_LABELS = np.row_stack((ALL_TRAINING_LABELS, label_matrix))

    ALL_FIBERS_GRP = np.delete(ALL_FIBERS_GRP, 0 , axis = 0)
    ALL_TRAINING_LABELS = np.delete(ALL_TRAINING_LABELS, 0, axis = 0)

    output_fiber_file_name = out_dir + fib.strip('\n') + '_global_train_fibers.npy'
    output_labels_file_name = out_dir + fib.strip('\n') + '_global_train_labels.npy'
    print(output_fiber_file_name)
    print(output_labels_file_name)
    
    np.save(output_fiber_file_name, ALL_FIBERS_GRP)
    np.save(output_labels_file_name, ALL_TRAINING_LABELS)

        
