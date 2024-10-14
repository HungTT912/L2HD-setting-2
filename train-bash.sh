
# python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s10000-l5.5-lr0.05-d0.25.yaml --save_top \
# &python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s18000-l5.5-lr0.05-d0.25.yaml --save_top 

# python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s12000-l5.5-lr0.05-d0.25.yaml --save_top \
# &python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s13000-l5.5-lr0.05-d0.25.yaml --save_top 

# python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s14000-l5.5-lr0.05-d0.25.yaml --save_top \
# &python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s15000-l5.5-lr0.05-d0.25.yaml --save_top 

# python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s16000-l5.5-lr0.05-d0.25.yaml --save_top \
# &python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind8-s17000-l5.5-lr0.05-d0.25.yaml --save_top 


# python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s12000-l5.5-lr0.05-d0.25.yaml \
# &python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s13000-l5.5-lr0.05-d0.25.yaml \
# &python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s14000-l5.5-lr0.05-d0.25.yaml \
# &python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s16000-l5.5-lr0.05-d0.25.yaml  


# python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s10000-l5.5-lr0.05-d0.25.yaml \
# &python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s15000-l5.5-lr0.05-d0.25.yaml \
# &python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s17000-l5.5-lr0.05-d0.25.yaml \
# &python3 test_tfbind8_tune_23.py --config configs/tune_23/Template-BBDM-tfbind8-s18000-l5.5-lr0.05-d0.25.yaml 

# python3 train_for_all_1.py --config configs/tune_23/Template-BBDM-tfbind10-s10000-l5.5-lr0.05-d0.25.yaml --save_top 

python3 test_tfbind10_tune_23.py --config configs/tune_23/Template-BBDM-tfbind10-l5.5-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_23_1.py --config configs/tune_23/Template-BBDM-tfbind10-l5.5-lr0.05-d0.25.yaml 

python3 test_tfbind10_tune_23_2.py --config configs/tune_23/Template-BBDM-tfbind10-l5.5-lr0.05-d0.25.yaml \
&python3 test_tfbind10_tune_23_3.py --config configs/tune_23/Template-BBDM-tfbind10-l5.5-lr0.05-d0.25.yaml 

python3 test_tfbind10_tune_23_4.py --config configs/tune_23/Template-BBDM-tfbind10-l5.5-lr0.05-d0.25.yaml 