# python3 train_and_test_for_all.py --config configs/final/Template-L2HD-ant.yaml --save_top \
# &python3 train_and_test_for_all.py --config configs/final/Template-L2HD-dkitty.yaml --save_top 

python3 test_ant_tune_22_100steps.py --config configs\tune_20\Template-BBDM-ant-l1.0-lr0.001-d0.25.yaml --save_top \
&python3 test_dkitty_tune_22_100steps.py --config configs\tune_20\Template-BBDM-dkitty-l1.0-lr0.001-d0.25.yaml --save_top 