# List of configuration file names based on the hyperparameters
lengthscales = [3.0, 4.0, 5.0, 6.0, 7.0]
learning_rates = [0.005, 0.01, 0.05, 0.1]
delta_lengthscale = 0.25

# Base path for configs and script
config_base_path = "configs/tune_20/"
script_name = "train_for_all.py"

# File to store the run commands
output_filename = "run_all_train.sh"

# Open file for writing
with open(output_filename, "w") as file:
    for lengthscale in lengthscales:
        for lr in learning_rates:
            config_name = f"Template-BBDM-tf8-l{lengthscale}-lr{lr}-d{delta_lengthscale}.yaml"
            command = f"python3 {script_name} --config {config_base_path}{config_name} --save_top \\"
            file.write(command + "\n& ")

    # Remove the last '& ' for proper shell script syntax
    file.seek(file.tell() - 2, 0)
    file.truncate()

print(f"Shell script '{output_filename}' created.")
