import pandas as pd
import glob
import numpy as np

# Specify the directory where your CSV files are located
csv_dir = './Data/raw/'

# List all CSV files in the directory
csv_files = glob.glob(csv_dir + '*.csv')

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through each CSV file, read it, and extract the first 4 columns
for csv_file in csv_files:
    df = pd.read_csv(csv_file, header=None)
    df = df.iloc[:, :4]
    df.columns = ['Datetime', 'mfv_gse_x', 'mfv_gse_y', 'mfv_gse_z']
    dataframes.append(df)
# Concatenate the DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Convert the 'Datetime' column to datetime format
combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'], format='%Y-%m-%d %H:%M:%S')

# Sort the DataFrame by the 'Datetime' column
combined_df = combined_df.sort_values(by='Datetime')
combined_df = combined_df.fillna(-np.inf)

combined_df.to_csv('./Data/dscovr_raw_clean.csv')
