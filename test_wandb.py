import wandb
import pandas as pd

# Read our CSV into a new DataFrame
file_path = './tuning_results/tune_11/result/tuning_result_ant.csv'
new_iris_dataframe = pd.read_csv(file_path)

# Convert the DataFrame into a W&B Table
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# Add the table to an Artifact to increase the row
# limit to 200000 and make it easier to reuse
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# log the raw csv file within an artifact to preserve our data
iris_table_artifact.add_file(file_path)

# Start a W&B run to log data
run = wandb.init(project="tables-walkthrough")

# Log the table to visualize with a run...
run.log({"iris": iris_table})

# and Log as an Artifact to increase the available row limit!
run.log_artifact(iris_table_artifact)
print("Done")
# Finish the run (useful in notebooks)
run.finish()