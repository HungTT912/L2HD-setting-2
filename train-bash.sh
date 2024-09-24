python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8--s1000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8-s2500-l5.0-lr0.05-d0.25.yaml --save_top \

&python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8-s10000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 train_for_all.py --config configs/tune_22/Template-BBDM-tfbind8-s15000-l5.0-lr0.05-d0.25.yaml --save_top \
&python3 test_tfbind8_tune_22.py --config configs/tune_22/Template-BBDM-tfbind8-s1000-l5.0-lr0.05-d0.25.yaml \
&python3 test_tfbind8_tune_22.py --config configs/tune_22/Template-BBDM-tfbind8-s2500-l5.0-lr0.05-d0.25.yaml \

python3 test_tfbind8_tune_22.py --config configs/tune_22/Template-BBDM-tfbind8-s10000-l5.0-lr0.05-d0.25.yaml \
&python3 test_tfbind8_tune_22.py --config configs/tune_22/Template-BBDM-tfbind8-s15000-l5.0-lr0.05-d0.25.yaml 

