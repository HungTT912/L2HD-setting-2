python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-dkitty-grads_25.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-dkitty-grads_50.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-dkitty-grads_75.yaml --save_top 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind10-grads_25.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind10-grads_50.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind10-grads_75.yaml --save_top 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind10-grads_25.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind10-grads_50.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab3_GP_num_gradient_steps/Template-BBDM-tfbind10-grads_75.yaml --save_top 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-dkitty-GP-delta0.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-dkitty-no_GP_last_bins.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-dkitty-no_GP_two_big_bins.yaml --save_top 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind10-GP-delta0.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind10-no_GP_last_bins.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind10-no_GP_two_big_bins.yaml --save_top 

