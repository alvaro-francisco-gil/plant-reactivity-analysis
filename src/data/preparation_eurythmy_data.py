import pandas as pd
import matplotlib.pyplot as plt
import os

"""
This class gathers the eurythmy data related to eurythmy from
text files and combines it with measurements_info.csv

Understand the data related to Eurythmy:
 - We know when the eurythmy was performed thanks of the video data (/data/raw/mp4_files)
 - The data was hand labelled in text files (/data/raw/txt_files)
 - The wav, text and video of a specific measurement share the 'id_measurement'(number at the start of the file #ID)
"""

def read_eurythmy_text_files_to_dict(folder_path):
    """
    Reads all text files in a given folder and creates a dictionary 
    with keys based on the ID extracted from the file names.

    Parameters:
    folder_path (str): The path to the folder containing the text files.

    Returns:
    dict: A dictionary where each key is an ID extracted from the file name 
          and the corresponding value is the content of that file.
    """

    # Initialize an empty dictionary to store file contents
    files_dict = {}

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the current file is a text file
        if filename.endswith('.txt'):
            # Extract the ID from the filename
            # Assuming the ID is the part of the filename after '#'
            id = filename.split('_')[0][1:]

            # Open and read the file
            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()
                
                # Store the file content in the dictionary with the ID as key
                files_dict[id] = content

    # Return the dictionary containing file contents
    return files_dict

def time_to_seconds(time_str):
    """
    Converts a time string in 'minutes:seconds' format to a total number of seconds.

    Parameters:
    time_str (str): Time string in 'minutes:seconds' format.

    Returns:
    int: Total number of seconds.
    """
    # Check if the time string is not empty
    if time_str:
        # Split the time string into minutes and seconds and convert them to integers
        minutes, seconds = map(int, time_str.split(':'))
        # Return the total time in seconds
        return minutes * 60 + seconds
    # Return 0 if the time string is empty
    return 0

def transform_text_labels_to_dictionary(original_dict):
    """
    Transforms a dictionary of text labels into a structured dictionary
    where each key corresponds to a list of dictionaries indicating start and end times
    for each label, converted to seconds.

    Parameters:
    original_dict (dict): Dictionary where each key is an ID and value is a multiline string 
                          with label information.

    Returns:
    dict: Transformed dictionary where each key maps to a list of dictionaries 
          with start and end times for each label.
    """
    transformed_dict = {}
    
    # Iterate through the original dictionary
    for key, value in original_dict.items():
        # Split the value into lines, skipping the first line (header)
        lines = value.strip().split('\n')[1:]
        
        # Initialize a list to hold dictionaries for each label
        letter_dicts = []
        
        # Process each line (label entry)
        for line in lines:
            # Split the line into its constituent parts
            parts = line.split(',')
            letter_id = parts[0]
            # Remove outliers
            if int(letter_id[1])<5:
                # Convert start and end times to seconds
                start_time = time_to_seconds(parts[1])
                end_time = time_to_seconds(parts[2])

                # Create dictionaries for start and end times and add them to the list
                letter_dicts.append({f'{letter_id}_start': start_time})
                letter_dicts.append({f'{letter_id}_end': end_time})
        
        # Map the list of dictionaries to the corresponding key in the transformed dictionary
        transformed_dict[key] = letter_dicts

    return transformed_dict

def create_dataframe_from_dict(transformed_dict):
    """
    Creates a pandas DataFrame from a transformed dictionary where each key-value pair
    in the dictionary corresponds to a row in the DataFrame.

    Parameters:
    transformed_dict (dict): A dictionary where each key is an ID and each value is a 
                             list of dictionaries with time label information.

    Returns:
    DataFrame: A pandas DataFrame where each row represents the data associated with 
               a unique ID from the transformed dictionary.
    """
    # Initialize a list to store the data for the new DataFrame
    new_data = []
    
    # Iterate through each key-value pair in the transformed dictionary
    for key, value_list in transformed_dict.items():
        # Initialize a dictionary for the current row with the ID
        row_data = {'ID': key}

        # Update the row dictionary with each item in the value list
        # This adds each time label's start and end data to the row
        for item in value_list:
            row_data.update(item)

        # Add the completed row data to the new_data list
        new_data.append(row_data)

    # Create a DataFrame from the list of row data
    new_df = pd.DataFrame(new_data)

    # Set the 'ID' column as the index of the new DataFrame
    new_df = new_df.set_index('ID')

    # Return the newly created DataFrame
    return new_df


def group_eurythmy_text_data_with_measurements(measurements_csv_file, txt_folder):
    """
    Integrates eurythmy text data with measurement data from a CSV file and 
    outputs the combined data to a new CSV file.

    Parameters:
    measurements_csv_file (str): Path to the measurements CSV file.
    txt_folder (str): Path to the folder containing eurythmy text files.

    The function performs the following steps:
    1. Reads measurement data from a CSV file into a DataFrame.
    2. Reads and transforms eurythmy text data from a specified folder into a DataFrame.
    3. Merges the two DataFrames on a common column and index.
    4. Finds and adds columns for the start and end of eurythmy sequences.
    5. Outputs the combined data to a new CSV file.
    """
    # Load the measurements data from the CSV file into a DataFrame
    meas_df = pd.read_csv(measurements_csv_file, index_col='id_measurement')

    # Read and transform eurythmy text files from the specified folder
    eurythmy_labels = read_eurythmy_text_files_to_dict(txt_folder)
    eurythmy_labels_clean = transform_text_labels_to_dictionary(eurythmy_labels)
    eurythmy_df = create_dataframe_from_dict(eurythmy_labels_clean)

    # Ensure the index of eurythmy_df is of integer type for merging
    eurythmy_df.index = eurythmy_df.index.astype(int)

    # Merge the measurements DataFrame with the eurythmy DataFrame
    df = pd.merge(meas_df, eurythmy_df, left_on='id_performance', right_index=True, how='outer')

    # Get a list of all column names in the merged DataFrame
    all_columns = df.columns.tolist()
    # Find the index of the column 'A1_start'
    start_index = all_columns.index('A1_start')
    # Create a list of column names starting from 'A1_start'
    letters = all_columns[start_index:]

    # Calculate the minimum and maximum values across the specified columns
    df['eurythmy_start'] = df[letters].min(axis=1)
    df['eurythmy_end'] = df[letters].max(axis=1)

    # Define the file path
    output_file_path = r"../../data/interim/measurements_with_eurythmy.csv"

    # Output the DataFrame to the CSV file
    df.to_csv(output_file_path)

    # Print the file path that has been used
    print(f"The DataFrame has been saved to: {output_file_path}")

"""
os.chdir(os.path.dirname(os.path.abspath(__file__)))
measurements_csv_file= r"..\..\data\raw\measurements_info.csv"
txt_folder= r"..\..\data\raw\txt_files"
group_eurythmy_text_data_with_measurements(measurements_csv_file, txt_folder)
"""
