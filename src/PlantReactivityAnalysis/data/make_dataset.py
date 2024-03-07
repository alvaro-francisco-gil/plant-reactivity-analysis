# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    # Example: Read a CSV file, process the data, and save to a new file
    try:
        # Read the data
        df = pd.read_csv(input_filepath)
        # Save the processed data
        df.to_csv(output_filepath, index=False)
        logger.info(f"Processed data saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Error processing data: {e}")


"""
create folders in processed!
os.chdir(os.path.dirname(os.path.abspath(__file__)))
measurements_csv_file= r"../../data/raw/measurements_info.csv"
txt_folder= r"../../data/raw/txt_files"
group_eurythmy_text_data_with_measurements(measurements_csv_file, txt_folder)
"""

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
