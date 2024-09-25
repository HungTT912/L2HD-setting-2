python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s13000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s14000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind10-s10000-l5.0-lr0.05-d0.25.yaml --save_top 


python3 test_tfbind8_tune_22_100steps.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s13000-l5.0-lr0.05-d0.25.yaml \
&python3 test_tfbind8_tune_22_100steps.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s14000-l5.0-lr0.05-d0.25.yaml \
&python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s14500-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s15500-l5.0-lr0.05-d0.25.yaml --save_top 

python3 test_tfbind8_tune_22_100steps.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s14500-l5.0-lr0.05-d0.25.yaml \
&python3 test_tfbind8_tune_22_100steps.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s15500-l5.0-lr0.05-d0.25.yaml \
&python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s16000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s17000-l5.0-lr0.05-d0.25.yaml --save_top 

python3 test_tfbind8_tune_22_100steps.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s16000-l5.0-lr0.05-d0.25.yaml \
&python3 test_tfbind8_tune_22_100steps.py --config configs/tune_22_100steps/Template-BBDM-tfbind8-s17000-l5.0-lr0.05-d0.25.yaml \

