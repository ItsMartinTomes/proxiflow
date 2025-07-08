import click
import polars as pl
from typing import Optional
from .config import Config
from .utils import get_logger, load_data, write_data
from .core import Cleaner, Normalizer, Engineer


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.option(
    "--config-file",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--input-file",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to input data file",
)
@click.option(
    "--output-file",
    "-o",
    required=True,
    type=click.Path(exists=False),
    help="Path to output data file",
)
@click.pass_context
@click.version_option()
def main(ctx: click.Context, config_file: str, input_file: str, output_file: str) -> None:
    # Set up logger
    logger = get_logger(__name__)

    # Load configuration
    config = Config(config_file)

    # Load data
    data: Optional[pl.DataFrame] = None  # or Any if type unknown

    try:
        data = load_data(input_file, input_file_format=config.input_format)
    except FileNotFoundError as e:
        logger.error("Input file not found: %s", str(e))
        return
    except ValueError as e:
        logger.error("Error parsing input file: %s", str(e))
        return

    if data is None:
        logger.error("No data loaded, exiting.")
        return

    # Perform data cleaning
    cleaner = Cleaner(config)
    try:
        cleaned_data = cleaner.execute(data)
    except ValueError as e:
        logger.error("Error cleaning data: %s", str(e))
        return

    # Perform data normalization
    normalizer = Normalizer(config)
    # normalized_data = normalizer.normalize(cleaned_data)
    try:
        normalized_data = normalizer.execute(cleaned_data)
    except Exception as e:
        logger.error("Normalizing data: %s", str(e))
        return

    # Perform feature engineering
    engineer = Engineer(config)
    try:
        engineered_data = engineer.execute(normalized_data)
    except Exception as e:
        logger.error("Engineering data: %s", str(e))
        return
    try:
        write_data(engineered_data, output_file, output_file_format=config.output_format)
    except Exception as e:
        logger.error(f"Error writing data to file {output_file}: {str(e)}")

    # Log completion message
    logger.info("Data preprocessing complete.")


if __name__ == "__main__":
    main()
