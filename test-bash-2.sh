python3 train_and_test_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_128.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_256.yaml --save_top \

python3 train_and_test_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_512.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_25.yaml --save_top 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_50.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind8-grads_75.yaml --save_top \
