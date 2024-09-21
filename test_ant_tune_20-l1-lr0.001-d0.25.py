import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np
import pandas as pd 
import csv
import subprocess 
from utils import dict2namespace, get_runner, namespace2dict
import design_bench
import wandb 

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


def CPU_singleGPU_launcher(config):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    if config.args.train:
        return runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return

def trainer(config): 
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    return runner.train()
def tester(config, task):
    set_random_seed(config.args.seed)
    runner = get_runner(config.runner, config)
    return runner.test(task) 

def main():
    nconfig, dconfig = parse_args_and_config()
    args = nconfig.args
    gpu_ids = args.gpu_ids
    if gpu_ids == "-1": # Use CPU
        nconfig.training.device = [torch.device("cpu")]
    else:
        nconfig.training.device = [torch.device(f"cuda:{gpu_ids}")]
    
    wandb.init(project='BBDM',
               name='test'+nconfig.wandb_name,
               config = dconfig) 
    
    best_tf8_hyper = pd.read_csv('./tuning_results/tune_20/result/tuning_result_tfbind8_lengthscale6.0_sampling_lr0.05_delta0.25.csv')
    best_tf8_hyper = best_tf8_hyper[best_tf8_hyper['mean (100th)']>0.9795]
    best_tf8_hyper = best_tf8_hyper[best_tf8_hyper['eta', 'alpha', 'classifier_free_guidance_weight']].to_numpy()
    seed_list = range(8)
    num_fit_samples = 10000
    
    
    if nconfig.task.name != 'TFBind10-Exact-v0':
        task = design_bench.make(nconfig.task.name)
    else:
        task = design_bench.make(nconfig.task.name,
                                dataset_kwargs={"max_samples": 10000})
    classifier_free_guidance_prob = 0.15 
    plot_time = 20 
    sampling_lr = 0.05
    for lengthscale in [1.0]:
        for delta in [0.25]: 

            folder_path = './tuning_results/tune_20/result' 
            if not os.path.exists(folder_path): 
                os.makedirs(folder_path)
            file_path = f'./tuning_results/tune_20/result/tuning_result_ant_lengthscale{lengthscale}_sampling_lr{sampling_lr}_delta{delta}.csv'
    
            if not os.path.isfile(file_path):
                with open(file_path, 'a') as file:
                    header = ['sampling_lr','lengthscale','delta', 'eta','alpha','classifier_free_guidance_weight', 'mean (100th)', 'std (100th)', 'mean (80th)', 'std (80th)', 'mean (50th)', 'std (50th)']
                    writer = csv.writer(file)
                    writer.writerow(header)

            if not os.path.isfile(file_path):
                with open(file_path, 'a') as file:
                    header = ['sampling_lr','lengthscale','delta', 'eta','alpha','classifier_free_guidance_weight', 'mean (100th)', 'std (100th)', 'mean (80th)', 'std (80th)', 'mean (50th)', 'std (50th)']
                    writer = csv.writer(file)
                    writer.writerow(header)
            df = pd.read_csv(file_path) 
            tested_parameters = df[['lengthscale','delta','eta','alpha','classifier_free_guidance_weight']].values.tolist()
    
            for eta, alpha, classifier_free_guidance_weight in best_tf8_hyper:  
                if [lengthscale, delta, eta, alpha, classifier_free_guidance_weight] in tested_parameters: 
                    continue 
                print([lengthscale,delta, eta, alpha, classifier_free_guidance_weight])
                results_100th = []
                results_80th = []
                results_50th = []
                nconfig.model.BB.params.eta = eta 
                for seed in seed_list:      
                    nconfig.training.classifier_free_guidance_prob = classifier_free_guidance_prob 
                    cmd = f"grep -Rlw './results/tune_20/AntMorphology-Exact-v0/sampling_lr{sampling_lr}/initial_lengthscale{lengthscale}/delta{delta}/seed{seed}' -e 'train: true'"
                    result_path = subprocess.check_output(cmd, shell=True, text=True)
                    result_path = result_path.strip()
                    #print(result_path)
                    cmd = 'find ' + result_path[:-12]+ " -name 'top_model*'"
                    model_load_path = subprocess.check_output(cmd, shell = True, text= True) 
                    model_load_path = model_load_path.strip() 
                    cmd = 'find ' + result_path[:-12]+ " -name 'top_optim*'"
                    optim_sche_load_path = subprocess.check_output(cmd, shell = True, text= True) 
                    optim_sche_load_path = optim_sche_load_path.strip()
                    print(model_load_path)
                    nconfig.args.train = False 
                
                    nconfig.testing.classifier_free_guidance_weight = classifier_free_guidance_weight
                    nconfig.testing.alpha = alpha
                    nconfig.model.model_load_path = model_load_path
                    nconfig.model.optim_sche_load_path = optim_sche_load_path
                    nconfig.args.seed = seed
                    result = tester(nconfig, task)
                    print("Score : ",result[0]) 
                    results_100th.append(result[0])
                    results_80th.append(result[1]) 
                    results_50th.append(result[2]) 
                
                assert len(results_100th) == 8 
                assert nconfig.GP.sampling_from_GP_lr == sampling_lr 
                assert nconfig.GP.delta_lengthscale == delta 
                assert nconfig.GP.initial_lengthscale == lengthscale 
                
                np_result_100th = np.array(results_100th)
                mean_score_100th = np_result_100th.mean() 
                std_score_100th = np_result_100th.std()
                np_result_80th = np.array(results_80th)
                mean_score_80th = np_result_80th.mean() 
                std_score_80th = np_result_80th.std()
                np_result_50th = np.array(results_50th)
                mean_score_50th = np_result_50th.mean() 
                std_score_50th = np_result_50th.std()
                print(mean_score_100th)
                
                with open(file_path, 'a') as file:
                    new_row = [sampling_lr,lengthscale,delta, eta, alpha, classifier_free_guidance_weight, mean_score_100th, std_score_100th, mean_score_80th, std_score_80th, mean_score_50th, std_score_50th]
                    writer = csv.writer(file)
                    writer.writerow(new_row)
                    plot_time -= 1 
                    if plot_time == 0 : 
                        df = pd.read_csv(file_path)
                        table = wandb.Table(dataframe=df)
                        wandb.log({"data_table": table})
                        plot_time = 20 
                            

    wandb.finish()

if __name__ == "__main__":
    main()