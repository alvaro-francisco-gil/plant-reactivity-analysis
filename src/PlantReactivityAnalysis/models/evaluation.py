import pandas as pd


def update_results_csv(file_path, df_results):
    """
    Updates the results CSV file by appending new results from a DataFrame.

    Parameters:
    - file_path: str, the path to the existing CSV file with the results.
    - df_results: DataFrame, the new results to append to the file.

    Returns:
    - None, but the CSV file at file_path is updated.
    """
    try:
        # Load the existing results from the CSV file
        df_existing = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"No existing file found at {file_path}. A new file will be created.")
        df_existing = pd.DataFrame()

    # Concatenate the new results to the existing DataFrame
    df_updated = pd.concat([df_existing, df_results], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df_updated.to_csv(file_path, index=False)
    print(f"Results updated and saved to {file_path}.")
