python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-ant-GP-delta0.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-GP-delta0.yaml --save_top \
&python3 test_for_ab5.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-no_GP.yaml 

python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-ant-no_GP_two_big_bins.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/ablation_studies/ab5_no_GP/Template-BBDM-tfbind8-no_GP_two_big_bins.yaml --save_top 