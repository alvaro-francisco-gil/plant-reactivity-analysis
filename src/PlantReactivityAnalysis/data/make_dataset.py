# -*- coding: utf-8 -*-
import os
import logging

# Import configuration paths
from PlantReactivityAnalysis.config import (DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR,
                                            PROCESSED_DATA_DIR, MEASUREMENTS_INFO, TEXT_EURYTHMY_FILES)
from PlantReactivityAnalysis.data.preparation_eurythmy_data import group_eurythmy_text_data_with_measurements


def ensure_data_directories_exist():
    """Ensures that data directories exist."""
    for path in [DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR]:
        os.makedirs(path, exist_ok=True)


def main():
    """Runs data processing scripts to turn raw data into cleaned data ready to be analyzed."""
    logger = logging.getLogger(__name__)
    logger.info("Ensuring data directories exist...")
    ensure_data_directories_exist()

    logger.info("Making final data set from raw data")
    try:
        group_eurythmy_text_data_with_measurements(MEASUREMENTS_INFO, TEXT_EURYTHMY_FILES)
    except Exception as e:
        logger.error(f"Error processing data: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
