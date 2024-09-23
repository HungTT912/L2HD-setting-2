python3 train_for_all.py --config configs/tune_21/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25_n2.yaml --save_top \
&python3 train_for_all.py --config configs/tune_21/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25_n3.yaml --save_top \
&python3 test_tfbind10_tune_20-l5.5-lr0.05-d0.5.py --config configs/tune_20/Template-BBDM-tfbind10-l5.5-lr0.05-d0.5.yaml \
&python3 test_tfbind10_tune_20-l5.5-lr0.05-d0.75.py --config configs/tune_20/Template-BBDM-tfbind10-l5.5-lr0.05-d0.75.yaml \
&python3 test_tfbind10_tune_20-l5.5-lr0.05-d1.0.py --config configs/tune_20/Template-BBDM-tfbind10-l5.5-lr0.05-d1.0.yaml \


python3 test_tfbind10_tune_21-l6.0-lr0.05-d0.25.py --config configs/tune_21/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25_n2.yaml \
&python3 test_tfbind10_tune_21-l6.0-lr0.05-d0.25.py --config configs/tune_21/Template-BBDM-tfbind10-l6.0-lr0.05-d0.25_n2.yaml 
