import numpy as np 
import sys

training_subj_list_file = sys.argv[1]

fiber_grps_file = './fiber_groups_reduced.txt'
base_dir = '/ifs/loni/faculty/thompson/four_d/Faisal/Projects/NeuralNet_TBI/AD_DOD_vol_param/'  
sub_dir = '/NewResults/Mapped_tracks/Tracks_text_files/'


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

##Read list of training subjs 
fid_training_subj = open(training_subj_list_file, 'r')
training_subj_list = fid_training_subj.readlines()

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


