import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
# import pandas as pd 
import wandb 
import design_bench
wandb.login(key="1cfab558732ccb32d573a7276a337d22b7d8b371")

from utils import dict2namespace, get_runner, namespace2dict


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('-c', '--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('-s', '--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('-r', '--result_path', type=str, default='results', help="The directory to save results")

    parser.add_argument('-t', '--train', action='store_true', default=False, help='train the model')
    parser.add_argument('--save_top', action='store_true', default=False, help="save top loss checkpoint")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')

    parser.add_argument('--resume_model', type=str, default=None, help='model checkpoint')
    parser.add_argument('--resume_optim', type=str, default=None, help='optimizer checkpoint')

    parser.add_argument('--max_epoch', type=int, default=None, help='optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='optimizer checkpoint')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        dict_config = yaml.load(f, Loader=yaml.FullLoader)

    namespace_config = dict2namespace(dict_config)
    namespace_config.args = args

    if args.resume_model is not None:
        namespace_config.model.model_load_path = args.resume_model
    if args.resume_optim is not None:
        namespace_config.model.optim_sche_load_path = args.resume_optim
    if args.max_epoch is not None:
        namespace_config.training.n_epochs = args.max_epoch
    if args.max_steps is not None:
        namespace_config.training.n_steps = args.max_steps

    dict_config = namespace2dict(namespace_config)

    return namespace_config, dict_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def trainer(config): 
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    return runner.train()
def tester(config,task):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    return runner.test(task) 

def main():
    nconfig, dconfig = parse_args_and_config()
    wandb.init(project='L2HD-x_high=x_t->x_0',
            name=nconfig.wandb_name,
            config = dconfig) 
    
    args = nconfig.args
    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.device = [torch.device("cpu")]
    else:
        nconfig.training.device = [torch.device(f"cuda:{gpu_ids}")]
    
    seed_list = range(8) ### number of independent runs with randomly seed 
    if nconfig.task.name != 'TFBind10-Exact-v0':
        task = design_bench.make(nconfig.task.name)
    else:
        task = design_bench.make(nconfig.task.name,
                                dataset_kwargs={"max_samples": 10000})
    if task.is_discrete: 
        task.map_to_logits()
    
    results_100th, results_80th, results_50th = [], [], [] 
    for seed in seed_list:
        
        nconfig.args.train = True 
        nconfig.args.seed = seed 
        nconfig.model.model_load_path = f"results/tune_1_setting_2_200/AntMorphology-Exact-v0/sampling_lr0.001/initial_lengthscale1.0/delta0.25/seed0/BrownianBridge/checkpoint/top_model_epoch_100.pth"
        nconfig.model.optim_sche_load_path = f"results/tune_1_setting_2_200/AntMorphology-Exact-v0/sampling_lr0.001/initial_lengthscale1.0/delta0.25/seed0/BrownianBridge/checkpoint/top_optim_sche_epoch_100.pth"

        
        nconfig.args.train = False 
        result = tester(nconfig,task)
        
        results_100th.append(result[0])
        results_80th.append(result[1])
        results_50th.append(result[2]) 
        
    results_100th, results_80th, results_50th = np.array(results_100th), np.array(results_80th), np.array(results_50th)
    print("Normalized 100th percentile score: ")
    print("Mean: ", np.mean(results_100th))
    print("Std: ", np.std(results_100th)) 
    print("Normalized 80th percentile score: ")
    print("Mean: ", np.mean(results_80th))
    print("Std: ", np.std(results_80th))
    print("Normalized 50th percentile score: ")
    print("Mean: ", np.mean(results_50th))
    print("Std: ", np.std(results_50th))
    # optional, print normalized 80th percentile or 50th percentile scores
    
if __name__ == "__main__":
    main()