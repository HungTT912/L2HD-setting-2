# python3 main.py --config configs/Template-BBDM-ant.yaml --train --save_top --gpu_ids 0 --seed 1
# python3 main.py --config configs/Template-BBDM-ant.yaml --gpu_ids 1

#final 
python3 main.py --config configs/Template-BBDM-ant.yaml --save_top --gpu_ids 0
python3 main2.py --config configs/Template-BBDM-ant.yaml --save_top --gpu_ids 0
python3 main3.py --config configs/Template-BBDM-ant.yaml --save_top --gpu_ids 0
python3 main4.py --config configs/Template-BBDM-ant.yaml --save_top --gpu_ids 0




python3 test_ant_1.py --config configs/Template-BBDM-ant.yaml 
python3 test_ant_2.py --config configs/Template-BBDM-ant.yaml 

python3 test_dkitty.py --config configs/Template-BBDM-dkitty.yaml 


python3 training_dkitty.py --config configs/Template-BBDM-dkitty.yaml --save_top --gpu_ids 0
python3 training_dkitty_2.py --config configs/Template-BBDM-dkitty.yaml --save_top --gpu_ids 0

python3 training_tfbind_8.py --config configs/Template-BBDM-tfbind8.yaml --save_top --gpu_ids 0
python3 training_ant.py --config configs/Template-BBDM-ant.yaml --save_top --gpu_ids 0
python3 training_tfbind_10.py --config configs/Template-BBDM-tfbind10.yaml --save_top --gpu_ids 0

python3 train_ant_tune_2.py --config configs/tune_2/Template-BBDM-ant.yaml --save_top --gpu_ids 0
python3 test_ant_tune_2.py --config configs/tune_2/Template-BBDM-ant.yaml 

python3 train_dkitty_tune_2.py --config configs/tune_2/Template-BBDM-dkitty.yaml --save_top --gpu_ids 0
python3 train_ant_tune_3.py --config configs/tune_3/Template-BBDM-ant.yaml --save_top --gpu_ids 0

python3 test_ant_tune_7.py --config configs/tune_11/Template-BBDM-ant-150.yaml --gpu_ids 0

python3 train_dkitty_tune_7.py --config configs/tune_12/Template-BBDM-dkitty.yaml --save_top --gpu_ids 0
python3 train_tfbind10_tune_6.py --config configs/tune_6/Template-BBDM-tfbind10.yaml --save_top --gpu_ids 0
python3 train_tfbind8_tune_6.py --config configs/tune_6/Template-BBDM-tfbind8.yaml --save_top --gpu_ids 0

python3 test_dkitty_tune_7.py --config configs/tune_11/Template-BBDM-dkitty.yaml --gpu_ids 0
python3 test_tfbind8_tune_14.py --config configs/tune_15/Template-BBDM-tfbind8.yaml --gpu_ids 0

python3 train_tfbind8_tune_13.py --config configs/tune_16/Template-BBDM-tfbind8.yaml --save_top --gpu_ids 0

python3 train_tfbind8_tune_17.py --config configs/tune_17/Template-BBDM-ant.yaml --save_top --gpu_ids 0