#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -o /home/rcf-proj2/vg/Softwares/FiberNet/Job_Submission_scripts/logs
#PBS -j oe
#PBS -N "DataPrep"  
apps_dir=/home/rcf-proj2/vg/Softwares/FiberNet/tflearn_based
base_dir=/home/rcf-proj2/vg/Data/AD_DOD/TrainingData/
out_dir=${base_dir}

split_ratio=0.8
#sub-directory within the subject directory 
sub_dir=/NewResults/Mapped_tracks/Tracks_text_files/

#training subjects
train_subj_list=${base_dir}/training_subj_list.txt
fiber_list=${apps_dir}/fiber_groups_reduced.txt
python_CPU_dir=/home/rcf-proj2/vg/Softwares/CPU_version/miniconda3/bin
python_GPU_dir=/home/rcf-proj2/vg/Softwares/miniconda3/bin

${python_CPU_dir}/python3 ${apps_dir}/Save_restore_CNN_fiber_net.py  ${base_dir}


