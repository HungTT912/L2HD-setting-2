python3 train_and_test_for_all.py --config configs/final/Template-L2HD-ant.yaml --save_top \
&python3 train_and_test_for_all.py --config configs/final/Template-L2HD-dkitty.yaml --save_top 

python3 test_tfbind10_tune_23.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l5.3-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_23_1.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l5.5-lr0.05-d0.25.yaml 

python3 test_tfbind10_tune_23.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l5.7-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_23_1.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l5.75-lr0.05-d0.25.yaml 

python3 test_tfbind10_tune_23.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l5.8-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_23_1.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l6.0-lr0.05-d0.25.yaml 

python3 test_tfbind10_tune_23_1.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l6.25-lr0.05-d0.25.yaml 