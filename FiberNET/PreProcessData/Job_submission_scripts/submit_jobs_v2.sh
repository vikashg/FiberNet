#$ -S /bin/bash
#$ -o /ifs/loni/faculty/thompson/four_d/vgupta/Tools/Codes_modded_by_Sophie/FiberNet/BinaryClassifier/PreProcessData/Job_submission_scripts_ppmi/logs
#$ -j y

py_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Softwares/miniconda3/bin
apps_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Codes_modded_by_Sophie/FiberNet/BinaryClassifier/PreProcessData

vol_param_dir=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Codes_modded_by_Sophie/Volumetric_Parameterization/JobSubmission_scripts_ppmi

base_dir=/ifs/loni/faculty/thompson/four_d/sthomopo/Tractography/PPMI/Volumetric_Parameterization/
training_subj_list=${vol_param_dir}/training_subj_list.txt

sub_dir=Training_data_bin_classification_imbalance_resolved_final/Tracks_text_files/
out_dir=${base_dir}${sub_dir}numpy_files/
mkdir -p ${out_dir}
fiber_list=/ifs/loni/faculty/thompson/four_d/vgupta/Tools/Codes_modded_by_Sophie/Volumetric_Parameterization/JobSubmission_scripts_ppmi/fiber_groups_new_ppmi.txt

cat ${training_subj_list}
${py_dir}/python3.5 ${apps_dir}/scratch_read_file_combined_tracks.py ${training_subj_list} ${out_dir} ${base_dir} ${fiber_list} ${sub_dir}

while read roi; do 

out_dir_split=${out_dir}/split/
mkdir -p ${out_dir_split}
${py_dir}/python3.5 ${apps_dir}/Create_train_test_split_ver2.py ${out_dir} 0.8 ${roi} ${out_dir_split} 
done < ${fiber_list}
