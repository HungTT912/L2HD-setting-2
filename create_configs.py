import yaml

# Template config
config_template = {
    "runner": "BBDMRunner",
    "tune": "tune_22_100steps",
    "wandb_name": "",
    "training": {
        "n_epochs": 100,
        "save_interval": 20,
        "validation_interval": 20,
        "accumulate_grad_batches": 2,
        "batch_size": 64,
        "val_frac": 0.1,
        "classifier_free_guidance_prob": 0.15
    },
    "testing": {
        "clip_denoised": False,
        "type_sampling": "highest",
        "percentile_sampling": 0.2,
        "num_candidates": 128
    },
    "task": {
        "name": 'TFBind8-Exact-v0',
        "normalize_y": True,
        "normalize_x": True
    },
    "GP": {
        "initial_lengthscale": 0.0,
        "initial_outputscale": 0.0,  # Will be the same as lengthscale
        "noise": 1.e-2,
        "num_functions": 8,
        "num_gradient_steps": 100,
        "num_points": 1024,
        "sampling_from_GP_lr": 0.0,
        "delta_lengthscale": 0.25,
        "delta_variance": 0.25,
        "threshold_diff": 0.001,
        "num_fit_samples": 0
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
                    "image_size": 24,
                    "hidden_size": 1024,
                    "condition_key": "SpatialRescaler"
                }
            }
        }
    }
}

# Hyperparameter lists
lengthscale = 5.0
learning_rates = [0.05]
delta_lengthscales = [0.25]
num_fit_samples_list = [2500, 5000, 7500, 8500, 9000, 9500, 10000, 15000]
task = 'tfbind8'

# Function to create file names and adjust wandb_name
def create_filename_and_wandb_name(lengthscale, lr, num_fit_samples):
    return f"./configs/tune_22_100steps/Template-BBDM-{task}-s{num_fit_samples}-l{lengthscale}-lr{lr}-d{delta_lengthscale}", f"tune_22_100steps-{task}-s{num_fit_samples}-l{lengthscale}-lr{lr}-d{delta_lengthscale}"

# Generate config files
for num_fit_samples in num_fit_samples_list: 
    for delta_lengthscale in delta_lengthscales:
        for lr in learning_rates:
            # Create filename and wandb_name
            filename, wandb_name = create_filename_and_wandb_name(lengthscale, lr, num_fit_samples)

            # Update the config template with specific values
            config = config_template.copy()
            config["wandb_name"] = wandb_name
            config["GP"]["initial_lengthscale"] = lengthscale
            config["GP"]["initial_outputscale"] = lengthscale  # Set outputscale equal to lengthscale
            config["GP"]["sampling_from_GP_lr"] = lr
            config["GP"]["delta_lengthscale"] = delta_lengthscale
            config["GP"]["delta_variance"] = delta_lengthscale 
            config["GP"]["num_fit_samples"] = num_fit_samples

            # Save to a YAML file
            with open(f"{filename}.yaml", "w") as f:
                yaml.dump(config, f)

        print(f"Config file '{filename}.yaml' created.")
