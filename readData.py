from __future__ import division, print_function, absolute_import
import os, os.path
import sys


def readData_files(file_list_name):
    _file_list = open(file_list_name, 'r')
    file_list = _file_list.readlines()
    with open(file_list) as f:
        num_files = sum(1 for _ in f)
    
    return num_files, file_list

base_dir = sys.argv[1]

train_file_list_name = base_dir + 'train_list.txt'
num_train, train_list = readData_files(train_file_list_name)

test_file_list_name = base_dir + 'test_list.txt'
num_test, test_list = readData_files(test_file_list_name)

valid_file_list_name = base_dir + 'valid_list.txt'
num_valid, valid_list = readData_files(valid_file_list_name)


