import numpy as np 
import sys
import os 


## Make changes here primarily directories 
training_subj_list_file = sys.argv[1] # Give the name for the list of training subjects
out_dir = sys.argv[2] # stores output data 

base_dir = sys.argv[3]
sub_dir = sys.argv[4]
fiber_grps_file = sys.argv[5]# I store the file containing the fiber groups in the same directory but you can change it 


def readSingleFile(fileName):
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

fid_fiber_grps = open(fiber_grps_file, 'r')
fiber_grps = fid_fiber_grps.readlines()
num_groups = len(fiber_grps)

##Read list of training subjs 
fid_training_subj = open(training_subj_list_file, 'r')
training_subj_list = fid_training_subj.readlines()


ALL_TRAINING_FIBERS=np.zeros([1,150])
ALL_TRAINING_LABELS = np.zeros([1, num_groups])

for subj in training_subj_list:
    data_dir = base_dir + subj.strip('\n') + sub_dir
    ## Counting the number of fibers 
    num_groups=len(fiber_grps) 

    ALL_FIBERS = np.zeros([1, 150])
    ALL_LABELS = np.zeros([1, num_groups])
    count = 0
    for fib in fiber_grps:
        file_name = data_dir + 'Mapped_fiber_' + fib.strip('\n') + '.txt'
        X_FIBER, num_FIBER = readSingleFile(file_name)
        label_matrix = np.zeros([num_FIBER, num_groups])
        label_matrix[:,count] = 1
        ALL_FIBERS = np.row_stack((ALL_FIBERS, X_FIBER))
        ALL_LABELS = np.row_stack((ALL_LABELS, label_matrix))
        count = count + 1
        print(fib.strip('\n'))

    ALL_FIBERS = np.delete(ALL_FIBERS, 0, axis =0)
    ALL_LABELS = np.delete(ALL_LABELS, 0, axis =0)

    file_name_fibers = data_dir+'All_fibers_mapped.npy'
    np.save(file_name_fibers, ALL_FIBERS)

    file_name_labels = data_dir + 'All_labels.npy'
    np.save(file_name_labels, ALL_LABELS)    
    
    ALL_TRAINING_FIBERS = np.row_stack((ALL_TRAINING_FIBERS, ALL_FIBERS))
    ALL_TRAINING_LABELS = np.row_stack((ALL_TRAINING_LABELS, ALL_LABELS))


ALL_TRAINING_FIBERS = np.delete(ALL_TRAINING_FIBERS, 0, axis = 0)
ALL_TRAINING_LABELS = np.delete(ALL_TRAINING_LABELS, 0, axis = 0)

file_name_labels = out_dir + 'All_training_labels.npy'
file_name_fibers = out_dir + 'All_training_fibers.npy'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
np.save(file_name_fibers, ALL_TRAINING_FIBERS)
np.save(file_name_labels, ALL_TRAINING_LABELS)


