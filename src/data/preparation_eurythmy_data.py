import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

"""
The following functions gather the eurythmy data related to eurythmy from
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

def return_meas_labels_by_keys(keys):
    """
    Returns a DataFrame with specified measurements based on given keys, 
    including 'id_measurement' as a regular column.

    :param keys: A list of keys to filter the measurements.
    :return: A DataFrame with selected columns for the specified keys.
    """

    meas_file = r"..\data\interim\measurements_with_eurythmy.csv"
    df_meas = pd.read_csv(meas_file)

    columns_to_include = ['id_measurement', 'id_performance', 'datetime', 'plant', 'generation', 'num_eurythmy']
    
    # Filter the DataFrame by keys and columns
    filtered_df = df_meas[df_meas['id_measurement'].isin(keys)][columns_to_include]

    return filtered_df

def extract_data_by_index_and_columns(df, indexes, columns):
    """
    Extracts a dictionary from a DataFrame based on given indexes and columns.

    :param df: The pandas DataFrame to extract data from.
    :param indexes: A list of indexes to extract data for.
    :param columns: A list of columns to extract data for.
    :return: A dictionary where each key is an index and each value is another dictionary with columns as keys.
    """
    extracted_data = {}
    for index in indexes:
        if index in df.index:
            # Extract data for the given index and specified columns
            extracted_data[index] = {col: df.at[index, col] for col in columns if col in df.columns}
        else:
            # Handle case where index is not in the DataFrame
            extracted_data[index] = {col: None for col in columns}
    return extracted_data

def format_letter_dict(original_dict):
    """
    Formats the letter dictionary to make it more readeable.

    :param original_dict: The original letter dictionary.
    :return: A dictionary where each letter is an index and each value is the range of time.

    Example: 
    original_dict= {'A1_start': 12, 'A1_end': 24, 'B1_start': 27, 'B1_end': 38}
    transformed_dict= {'A1': '[12-24]', 'B1': '[27-38]'}
    """
    transformed_dict = {}
    for key, value in original_dict.items():
        transformed_entry = {}
        for label, time in value.items():
            base_label = label[:2]  # Extract base label (like 'A1', 'G1')

            # Initialize with [None, None] only if it hasn't been initialized yet
            if base_label not in transformed_entry:
                transformed_entry[base_label] = [None, None]

            # Set start or end time
            if '_start' in label:
                transformed_entry[base_label][0] = time
            elif '_end' in label:
                transformed_entry[base_label][1] = time

        # Remove entries with both values as None or np.nan
        transformed_entry = {k: v for k, v in transformed_entry.items() if not all(np.isnan(val) if isinstance(val, float) else val is None for val in v)}

        transformed_dict[key] = transformed_entry
    return transformed_dict

def match_measurements_with_letters(df, time_dict):
    """
    Adds a new column 'eurythmy_letter' to the DataFrame indicating the eurythmy letter performed at each moment.

    :param df: DataFrame containing the columns 'id_measurement' and 'initial_second'.
    :param time_dict: Dictionary with keys and time ranges.
    :return: DataFrame with the added column 'eurythmy_letter'.
    """
    # Function to determine the eurythmy letter for a row
    def determine_letter(row):
        key = row['id_measurement']
        position = row['initial_second']
        time_ranges = time_dict.get(key, {})

        for label, (start, end) in time_ranges.items():
            if start <= position <= end:
                return label
        return None

    # Apply the function to each row
    df['eurythmy_letter'] = df.apply(determine_letter, axis=1)

    return df


def add_meas_letters(feat_df):
    """
    Adds a new column 'eurythmy_letter' to the DataFrame.

    :param feat_df: DataFrame containing the measurement features.
    :return: DataFrame with the added column 'eurythmy_letter'.
    """

    letter_columns= ['A1_start', 'A1_end', 'G1_start', 'G1_end', 'D1_start',
       'D1_end', 'A2_start', 'A2_end', 'G2_start', 'G2_end', 'D2_start',
       'D2_end', 'A3_start', 'A3_end', 'G3_start', 'G3_end', 'D3_start',
       'D3_end', 'A4_start', 'A4_end', 'G4_start', 'G4_end', 'D4_start',
       'D4_end', 'O1_start', 'O1_end', 'O2_start', 'O2_end', 'O3_start',
       'O3_end', 'O4_start', 'O4_end', 'L1_start', 'L1_end', 'L2_start',
       'L2_end', 'L3_start', 'L3_end', 'L4_start', 'L4_end']
    
    # Read eurythmy letters information
    meas_file = r"..\data\interim\measurements_with_eurythmy.csv"
    df_meas = pd.read_csv(meas_file, index_col='id_measurement')

    # Extract and format letters data
    indexes= feat_df['id_measurement'].tolist()
    letter_dictionary= extract_data_by_index_and_columns(df_meas, indexes, letter_columns)
    letter_dictionary = format_letter_dict(letter_dictionary)

    # Include letter data in the df
    new_df= match_measurements_with_letters(feat_df, letter_dictionary)
    
    return new_df

@staticmethod
def get_targets_rq1_is_eurythmy(df):
    """
    RQ1: Is there any difference in the signals when someone is performing eurythmy?
    Filters the DataFrame based on 'eurythmy_letter', then processes the 'num_eurythmy' column.

    :param df: DataFrame containing the columns 'num_eurythmy' and 'eurythmy_letter'.
    :return: A tuple containing the indexes list and the targets list.
    """
    # Filter the DataFrame where 'eurythmy_letter' is not None
    filtered_df = df[df['eurythmy_letter'].notnull()]

    # Store the indexes of the filtered rows
    indexes = filtered_df.index.tolist()

    # Apply the operation to generate targets
    # This will create a target of 1 if 'num_eurythmy' is greater than 0, else 0
    targets = [1 if x > 0 else 0 for x in filtered_df['num_eurythmy']]

    return indexes, targets

@staticmethod
def get_targets_rq2_what_letter(df):
    """
    RQ2: Is there any difference in the signals between different eurythmy letters?
    Process a DataFrame to filter based on 'eurythmy_letter' and create a list of targets.

    :param df: DataFrame containing the 'eurythmy_letter' column.
    :return: A tuple containing the indexes list and the 'target' list.
    """
    # Filter the DataFrame and explicitly create a copy
    filtered_df = df[(df['num_eurythmy'] != 0) & df['eurythmy_letter'].str.startswith(('A', 'G', 'D'))].copy()

    # Determine the target for each row based on 'eurythmy_letter'
    target_list = []
    for letter in filtered_df['eurythmy_letter']:
        if letter.startswith('A'):
            target_list.append(0)
        elif letter.startswith('G'):
            target_list.append(1)
        elif letter.startswith('D'):
            target_list.append(2)
        else:
            target_list.append(None)

    # Store the indexes
    indexes = filtered_df.index.tolist()

    return indexes, target_list

@staticmethod
def get_targets_rq3_eurythmy_habituation(df):
    """
    Assigns classes based on 'num_eurythmy' and returns indexes and classes.

    :param df: DataFrame containing the 'num_eurythmy' column.
    :return: A tuple containing the indexes list and the classes list.
    """
    # Filter out rows where 'num_eurythmy' is 0 and 'eurythmy_letter' is not None
    filtered_df = df[(df['num_eurythmy'] != 0) & df['eurythmy_letter'].notnull()]

    # Store the indexes of the filtered rows
    indexes = filtered_df.index.tolist()

    # Assign classes based on 'num_eurythmy'
    classes = []
    for num in filtered_df['num_eurythmy']:
        if num in [1, 2]:
            classes.append(0)
        elif num in [3, 4]:
            classes.append(1)
        elif num in [5, 6]:
            classes.append(2)
        elif num in [7, 8]:
            classes.append(3)
        else:
            classes.append(None)

    return indexes, classes

@staticmethod
def get_targets_rq4_eurythmy_performance_habituation(df):
    """
    Extracts the second character of 'eurythmy_letter' and returns indexes and the characters list.
    Filters out rows where 'num_eurythmy' is 0 or 'eurythmy_letter' is None.

    :param df: DataFrame containing the 'eurythmy_letter' and 'num_eurythmy' columns.
    :return: A tuple containing the indexes list and the list of second characters.
    """
    # Filter out rows where 'num_eurythmy' is 0 or 'eurythmy_letter' is None
    filtered_df = df[(df['num_eurythmy'] != 0) & df['eurythmy_letter'].notnull()]

    # Store the indexes of the filtered rows
    indexes = filtered_df.index.tolist()

    # Extract the second character of 'eurythmy_letter'
    classes = [letter[1] if len(letter) > 1 else None for letter in filtered_df['eurythmy_letter']]

    return indexes, classes

def get_indexes_and_targets_by_rq(rq_number, df):
    """
    Delegates to one of the four functions based on the RQ number.

    :param rq_number: The research question number (1, 2, 3, or 4).
    :param df: DataFrame to process.
    :return: A tuple containing the indexes list and the targets/classes/characters list.
    """
    df= df.reset_index(drop=True)

    if rq_number == 1:
        return get_targets_rq1_is_eurythmy(df)
    elif rq_number == 2:
        return get_targets_rq2_what_letter(df)
    elif rq_number == 3:
        return get_targets_rq3_eurythmy_habituation(df)
    elif rq_number == 4:
        return get_targets_rq4_eurythmy_performance_habituation(df)
    else:
        raise ValueError("Invalid RQ number. Please provide a number between 1 and 4.")
    
def read_indexes_file():

    split_indexes_path= r"..\data\interim\split_indexes.txt"
    train_indexes = []
    val_indexes = []
    test_indexes = []
    current_set = None

    with open(split_indexes_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            if line == "Training Set:":
                current_set = train_indexes
            elif line == "Validation Set:":
                current_set = val_indexes
            elif line == "Test Set:":
                current_set = test_indexes
            else:
                if current_set is not None:
                    current_set.append(int(line))

    return train_indexes, val_indexes, test_indexes

def find_matching_indexes(train_values, val_values, test_values, df, column):
    """
    Find indexes in a DataFrame where the column value matches values in given lists.

    :param train_values: List of values to match against the training set
    :param val_values: List of values to match against the validation set
    :param test_values: List of values to match against the test set
    :param df: DataFrame to search in
    :param column: Column name in DataFrame to match values against
    :return: Three lists containing indexes in the DataFrame for train, validation, and test values
    """
    train_indexes = df[df[column].isin(train_values)].index.tolist()
    val_indexes = df[df[column].isin(val_values)].index.tolist()
    test_indexes = df[df[column].isin(test_values)].index.tolist()

    return train_indexes, val_indexes, test_indexes

def get_train_val_test_indexes(df):
    """
    Get the train, validation and test indexes given the Dataframe.

    :param df: Dataframe representing the features
    """
    # Read 'id_measurement' idxs from text file
    train_values, val_values, test_values= read_indexes_file()

    # Find the indexes equal to 'id_measurement' for each group
    train_indexes, val_indexes, test_indexes= find_matching_indexes(train_values, val_values, test_values, df, column='id_measurement')

    return train_indexes, val_indexes, test_indexes

    




