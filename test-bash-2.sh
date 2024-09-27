python3 test_ant_tune_22_100steps.py --config configs/tune_20/Template-BBDM-ant-l1.0-lr0.001-d0.25.yaml \
&python3 test_dkitty_tune_22_100steps.py --config configs/tune_20/Template-BBDM-dkitty-l1.0-lr0.001-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_1.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_2.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_3.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 train_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-ant-num_points_128.yaml --save_top \
&python3 train_for_all.py --config configs/ablation_studies/ab2_GP_type_of_initial_points/Template-BBDM-ant-num_points_256.yaml --save_top 

python3 train_for_all.py --config configs/ablation_studies/ab2_GP_num_of_initial_points/Template-BBDM-ant-num_points_512.yaml --save_top \
&python3 test_tfbind10_tune_22_100steps_4.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_5.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_6.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_7.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_8.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_22_100steps_9.py --config configs/tune_20/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25.yaml \
