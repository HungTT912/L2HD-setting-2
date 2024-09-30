python3 test_for_ab5.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-ant-GP-delta0.yaml \
&python3 test_for_ab5.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-GP-delta0.yaml \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-ant-type_points_random.yaml 

python3 test_for_ab4.py --config configs/ablation_studies/ab4_type_of_conditioning_points/Template-BBDM-ant-highest.yaml \
&python3 test_for_ab4.py --config configs/ablation_studies/ab4_type_of_conditioning_points/Template-BBDM-tfbind8-highest.yaml 
