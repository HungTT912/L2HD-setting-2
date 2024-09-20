# python3 main.py --config configs/Template-BBDM-ant.yaml --train --save_top --gpu_ids 0 --seed 1
# python3 main.py --config configs/Template-BBDM-ant.yaml --gpu_ids 1

#final 
python3 train_for_all.py --config configs/tune_20/Template-BBDM-tfbind8-d0.05.yaml --save_top & python3 train_for_all.py --config configs/tune_20/Template-BBDM-tfbind8-d0.1.yaml --save_top & python3 train_for_all.py --config configs/tune_20/Template-BBDM-tfbind8-d0.25.yaml --save_top & python3 train_for_all.py --config configs/tune_20/Template-BBDM-tfbind8-d0.5.yaml --save_top & python3 train_for_all.py --config configs/tune_20/Template-BBDM-tfbind8-d1.yaml --save_top
# python test.py
