import requests
import csv

#DSCOVR Live Data at https://www.swpc.noaa.gov/products/real-time-solar-wind
url = 'https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'

try:
    # Send an HTTP GET request to the URL.
    response = requests.get(url)

    # Check if the request was successful (status code 200).
    if response.status_code == 200:
        # Parse the JSON data from the response.
        json_data = response.json()
        # Specify the CSV file name where you want to save the data.
        csv_file_name = './Data/dscovr_live_7_days.csv'

        # Open the CSV file in write mode.
        with open(csv_file_name, 'w', newline='') as csv_file:
            # Create a CSV writer object.
            csv_writer = csv.writer(csv_file)

            # Write each row of the 2D array to the CSV file.
            for row in json_data:
                csv_writer.writerow(row)

    else:
        print(f"Failed to download JSON. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")