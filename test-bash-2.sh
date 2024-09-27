python3 train_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-ant-no_GP.yaml --save_top \
&python3 train_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-no_GP.yaml --save_top 

python3 train_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_128.yaml --save_top \
&python3 train_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_256.yaml --save_top 

python3 train_for_all.py --config configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-tfbind8-num_points_512.yaml --save_top

