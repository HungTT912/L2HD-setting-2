python3 train_ant.py --config configs/tune_wandb/Template-BBDM-ant.yaml --save_top --gpu_ids 1
python3 test_ant_wandb.py --config configs/tune_wandb/Template-BBDM-ant.yaml 

python3 test_ant_tune_13.py --config configs/tune_14/Template-BBDM-ant.yaml --gpu_ids 0