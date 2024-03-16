# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

# Import your configuration module
from PlantReactivityAnalysis.config import MEASUREMENTS_INFO, TEXT_EURYTHMY_FILES

import pandas as pd


def group_eurythmy_text_data_with_measurements(measurements_csv_file, txt_folder):
    """
    Placeholder for your actual function implementation.
    Adjust this function to process the eurythmy text data.
    """
    pass


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Use paths from the config module
    measurements_csv_file = MEASUREMENTS_INFO
    txt_folder = TEXT_EURYTHMY_FILES

    try:
        # Custom processing function with paths from config
        group_eurythmy_text_data_with_measurements(measurements_csv_file, txt_folder)

        # Read the data
        df = pd.read_csv(input_filepath)
        # Process and save the data
        df.to_csv(output_filepath, index=False)
        logger.info(f"Processed data saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Error processing data: {e}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
