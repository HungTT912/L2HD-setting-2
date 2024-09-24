# python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n2.yaml --save_top \
# &python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n3.yaml --save_top 

python3 test_tfbind8_tune_21-l5-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n2.yaml \
&python3 test_tfbind8_tune_21-l5-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n3.yaml \
&python3 test_tfbind8_tune_21-l5-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n5.yaml \
&python3 test_tfbind8_tune_21-l5-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n4.yaml \
&python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n2.yaml --save_top \
&python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n3.yaml --save_top 

# &python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n4.yaml --save_top
# &python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n5.yaml --save_top 

# python3 test_tfbind8_tune_21-l5-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n4.yaml \
# &python3 test_tfbind8_tune_21-l5-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l5.0-lr0.05-d0.25_n5.yaml \
# &python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n2.yaml --save_top \
# &python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n3.yaml --save_top 

python3 test_tfbind8_tune_21-l6-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n2.yaml \
&python3 test_tfbind8_tune_21-l6-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n3.yaml \
&python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n4.yaml --save_top \
&python3 train_for_all.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n5.yaml --save_top 

python3 test_tfbind8_tune_21-l6-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n4.yaml \
&python3 test_tfbind8_tune_21-l6-lr0.05-d0.25.py --config configs/tune_21_down_samples_for_tf10/Template-BBDM-tfbind8-l6.0-lr0.05-d0.25_n5.yaml 