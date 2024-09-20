import wandb 
import csv
import pandas as pd 
import time 

wandb.init(project='test_csv')
file_path = 'test_csv.csv'
header = ['A','B']
with open(file_path, 'a') as file: 
    writer = csv.writer(file)
    writer.writerow(header)
for i,j in zip(range(100),range(100)): 
    with open(file_path, 'a') as file:
        new_row = [i,j] 
        writer = csv.writer(file)
        writer.writerow(new_row)
        df = pd.read_csv(file_path)
        # Log the DataFrame as a table artifact
        table = wandb.Table(dataframe=df)
        wandb.log({"data_table": table})
        time.sleep(20)

wandb.finish()
