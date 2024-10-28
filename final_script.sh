python3 test_for_all.py --config configs/final/Template-L2HD-ant.yaml \
&python3 test_for_all.py --config configs/final/Template-L2HD-dkitty.yaml 

python3 test_for_all.py --config configs/final/Template-L2HD-tfbind8.yaml \
&python3 test_for_all.py --config configs/final/Template-L2HD-tfbind10.yaml 


python3 train_and_test_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_128.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_256.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_512.yaml 



python3 train_and_test_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-tfbind8-type_points_lowest.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-tfbind8-type_points_random.yaml 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_25.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_50.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_75.yaml 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-GP-delta0.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-no_GP_last_bins.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-no_GP_two_big_bins.yaml 
