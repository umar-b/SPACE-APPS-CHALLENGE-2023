import wget
import os
import gzip
import netCDF4 as nc
import csv
import glob
from datetime import datetime
import pandas as pd
import numpy as np

#Download the archive
urls = ['https://www.ngdc.noaa.gov/dscovr/data/2023/10/oe_f1m_dscovr_s20231003000000_e20231003235959_p20231004022544_pub.nc.gz']
download_folder = './Data/nc.gz/'
output_folder = './Data/nc/'

for url in urls:
    file_name = url.split('/')[-1]
    wget.download(url, download_folder)
    
    #Decoompress the archive data
    with gzip.open(os.path.join(download_folder, file_name), 'rb') as gz_file:
        nc_data = gz_file.read()
        with open(os.path.join(output_folder, file_name[:-3]), 'wb') as nc_file:
            nc_file.write(nc_data)

# Specify the output CSV file name
csv_file_name = './Data/dscovr_archive.csv'
# Open the CSV file in write mode
with open(csv_file_name, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Create a list to store the variable names
    variable_names = ['time', 'proton_density', 'proton_speed', 'proton_temperature']

    # Loop through the .nc files
    nc_files = glob.glob(f"{output_folder}/*.nc")

    # Write the header row with variable names
    csv_writer.writerow(variable_names)

    # Reopen the CSV file for appending
    csv_file.close()
    csv_file = open(csv_file_name, 'a', newline='')
    csv_writer = csv.writer(csv_file)

    # Loop through the .nc files again and write data to the CSV file
    for nc_file_path in nc_files:
        # Open the .nc file for reading
        dataset = nc.Dataset(nc_file_path, 'r')

        # Create a dictionary to store data for each variable
        variable_data = {var_name: [] for var_name in variable_names}

        # Loop through the variables and extract data
        for var_name, variable in dataset.variables.items():
            if var_name == 'time':
                # Convert each timestamp to a datetime object
                time_data = variable[:]
                time_data = [datetime.fromtimestamp(ts/1000) for ts in time_data]
                variable_data[var_name] = time_data
            else:
                variable_data[var_name] = variable[:]

        # Write data to the CSV file
        num_rows = len(variable_data[variable_names[0]])
        for row_idx in range(num_rows):
            csv_writer.writerow([variable_data[var_name][row_idx] for var_name in variable_names])

        # Close the .nc file
        dataset.close()

#Filter the csv file for nan values
df = pd.read_csv('./Data/dscovr_archive.csv', sep=',')
for (columnName, columnData) in df.iteritems():
    df = df[df[columnName] != '--']
    df = df[df[columnName] != -9.9999000e+04]
df.to_csv('./Data/dscovr_archivce_clean.csv')
df_np = df.to_numpy()
np.save('./Data/dscovr_archive.npy', df_np)