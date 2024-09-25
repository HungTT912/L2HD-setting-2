import yaml

# Template config
config_template = {
    "runner": "BBDMRunner",
    "tune": "ablation_studies/ab1",
    "wandb_name": "",
    "training": {
        "n_epochs": 100,
        "save_interval": 20,
        "validation_interval": 20,
        "accumulate_grad_batches": 2,
        "batch_size": 64,
        "val_frac": 0.1,
        "classifier_free_guidance_prob": 0.15,
        "no_GP" : False 
    },
    "testing": {
        "clip_denoised": False,
        "type_sampling": "highest",
        "percentile_sampling": 0.2,
        "num_candidates": 128
    },
    "task": {
        "name": 'AntMorphology-Exact-v0',
        "normalize_y": True,
        "normalize_x": True
    },
    "GP": {
        "initial_lengthscale": 1.0,
        "initial_outputscale": 1.0,  # Will be the same as lengthscale
        "noise": 1.e-2,
        "num_functions": 8,
        "num_gradient_steps": 100,
        "num_points": 1024,
        "sampling_from_GP_lr": 0.001,
        "delta_lengthscale": 0.25,
        "delta_variance": 0.25,
        "threshold_diff": 0.001,
        "type_of_initial_points": 'highest'
        # "num_fit_samples": 0
    },
    "model": {
        "model_name": "BrownianBridge",
        "model_type": "BBDM",
        "latent_before_quant_conv": False,
        "normalize_latent": False,
        "only_load_latent_mean_std": False,
        "EMA": {
            "use_ema": True,
            "ema_decay": 0.995,
            "update_ema_interval": 8,
            "start_ema_step": 4000
        },
        "CondStageParams": {
            "n_stages": 2,
            "in_channels": 3,
            "out_channels": 3
        },
        "BB": {
            "optimizer": {
                "weight_decay": 0.0,
                "optimizer": "Adam",
                "lr": 1.e-3,
                "beta1": 0.9
            },
            "lr_scheduler": {
                "factor": 0.5,
                "patience": 200,
                "threshold": 0.0001,
                "cooldown": 200,
                "min_lr": 5.e-7
            },
            "params": {
                "mt_type": "linear",
                "objective": "grad",
                "loss_type": "l1",
                "skip_sample": True,
                "sample_type": "linear",
                "sample_step": 200,
                "num_timesteps": 1000,
                "max_var": 1.0,
                "MLPParams": {
                    "image_size": 60,
                    "hidden_size": 1024,
                    "condition_key": "SpatialRescaler"
                }
            }
        }
    }
}

task = 'ant'
num_points_list = [128, 256, 512, 1024]

# Function to create file names and adjust wandb_name
def create_filename_and_wandb_name(num_points):
    return f"./configs/ablation_studies/ab1_GP_num_of_initial_points/Template-BBDM-{task}-num_points_{num_points}",f"abs1-{task}-num_points_{num_points}"

# Generate config files
for num_points in num_points_list: 
    filename, wandb_name = create_filename_and_wandb_name(num_points)

    # Update the config template with specific values
    config = config_template.copy()
    config["wandb_name"] = wandb_name
    config['GP']['num_points'] = num_points
    # Save to a YAML file
    with open(f"{filename}.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Config file '{filename}.yaml' created.")
