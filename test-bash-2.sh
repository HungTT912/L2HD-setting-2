python3 test_for_ab1_tfbind8.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_128.yaml \
&python3 test_for_ab2_ant.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-ant-type_points_highest.yaml \
&python3 test_for_ab2_tfbind8.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-tfbind8-type_points_highest.yaml \
&python3 test_for_ab3.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-ant-grads_25.yaml 

python3 test_for_ab3.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_25.yaml \
&python3 test_for_ab5.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-ant-no_GP_last_bins.yaml \
&python3 test_for_ab5.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-no_GP_last_bins.yaml 




