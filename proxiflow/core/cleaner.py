import polars as pl
import numpy as np
from sklearn.impute import KNNImputer # type: ignore[import-untyped]
from proxiflow.config import Config
from proxiflow.utils import generate_trace
from typing import Mapping


class Cleaner:
    """
    A class for performing data preprocessing tasks such as cleaning, normalization, and feature engineering.
    """

    def __init__(self, config: Config):
        """
        Initialize a new Cleaner object with the specified configuration.

        :param config: A Config object containing the cleaning configuration values.
        :type config: Config
        """
        self.config = config.cleaning_config

    def execute(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clean a polars DataFrame by removing duplicates and filling in missing values.

        :param df: The DataFrame to clean.
        :type df: polars.DataFrame

        :returns df: The cleaned DataFrame.
        :rtype: polars.DataFrame

        :raises ValueError: If the DataFrame is empty.
        """
        if df.shape[0] == 0:
            raise ValueError("Empty DataFrame, no missing values to fill.")

        cleaned_df = df
        # Handle missing values. drop|mean|mode are mutually exclusive
        missing_values = self.config["handle_missing_values"]

        # Drop missing values
        if missing_values["drop"]:
            try:
                cleaned_df = self._drop_missing(cleaned_df)
            except Exception as e:
                trace = generate_trace(e, self._drop_missing)
                raise Exception(f"Trying to drop missing values: {trace}")
            return cleaned_df

        # Fill missing values with the mean of the column
        if missing_values["mean"]:
            try:
                cleaned_df = self._mean_missing(cleaned_df)
            except Exception as e:
                trace = generate_trace(e, self._mean_missing)
                raise Exception(f"Trying to fill missing values with the mean: {trace}")
            return cleaned_df

        # Fill missing values with the median of the column
        if missing_values["median"]:
            try:
                cleaned_df = self._median_missing(cleaned_df)
            except Exception as e:
                trace = generate_trace(e, self._median_missing)
                raise Exception(f"Trying to fill missing values with the median: {trace}")
            return cleaned_df

        # Fill missing values with KNN Imputer
        if missing_values["knn"]:
            try:
                cleaned_df = self._knn_impute_missing(cleaned_df)
            except Exception as e:
                trace = generate_trace(e, self._knn_impute_missing)
                raise Exception(f"Trying to fill missing values with KNN Imputer: {trace}")

        # NOTE: This is currently disabled because it randomly fails with:
        # Fill missing values with the mode of the column.
        # if missing_values["mode"]:
        #     cleaned_df = self.mode_missing(cleaned_df)
        #     return cleaned_df

        # Fill outliers with the median of the column
        if self.config["handle_outliers"]:
            try:
                cleaned_df = self._handle_outliers(cleaned_df)
            except Exception as e:
                trace = generate_trace(e, self._handle_outliers)
                raise Exception(f"Trying to fill outliers with the median: {trace}")

        # Handle duplicate rows
        if self.config["remove_duplicates"]:
            try:
                cleaned_df = self._remove_duplicates(cleaned_df)
            except Exception as e:
                trace = generate_trace(e, self._remove_duplicates)
                raise Exception(f"Trying to remove duplicate rows: {trace}")

        return cleaned_df

    def _remove_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove duplicate rows from a polars DataFrame.

        :param df: The DataFrame to remove duplicates from.
        :type df: polars.DataFrame

        :returns: The DataFrame with duplicates removed.
        :rtype: polars.DataFrame
        """
        return df.unique(keep="first")

    def _drop_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Drop rows with missing values from a polars DataFrame.

        :param df: The DataFrame to drop rows from.
        :type df: polars.DataFrame

        :returns: The DataFrame with rows with missing values dropped.
        :rtype: polars.DataFrame
        """
        return df.drop_nulls()

    def _mean_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill missing values with the mean of the column.

        :param df: The DataFrame to fill missing values in.
        :type df: polars.DataFrame

        :returns: The DataFrame with missing values filled.
        :rtype: polars.DataFrame
        """
        # Only update numeric columns
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Int64, pl.Float64)]
        # Use with_columns to fill nulls with the mean for each numeric column
        return df.with_columns([pl.col(col).fill_null(strategy="mean").alias(col) for col in numeric_cols])

    def _median_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill missing values with the median of the column.

        :param df: The DataFrame to fill missing values in.
        :type df: polars.DataFrame

        :returns: The DataFrame with missing values filled.
        :rtype: polars.DataFrame
        """

        numeric_cols = [col for col in df.columns if df[col].dtype in (pl.Int64, pl.Float64)]
        return df.with_columns([pl.col(col).fill_null(df[col].median()).alias(col) for col in numeric_cols])

    # TODO: Investigate why this randomly fails with:
    #  Error cleaning data: must specify either a fill 'value' or 'strategy'
    def _mode_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill missing values with the mode of the column. Only Int64 and Str data types are supported.

        :param df: The DataFrame to fill missing values in.
        :type df: polars.DataFrame

        :returns: The DataFrame with missing values filled with mode or original null (in case of unsupported data type)
        :rtype: polars.DataFrame
        """
        # Select columns of type Int64 or Utf8 (string)
        target_cols = [col for col in df.columns if df[col].dtype in (pl.Int64, pl.Utf8)]
        # Build a list of expressions to fill nulls with mode
        fill_exprs = []
        for col in target_cols:
            # .mode() returns a Series, take the first value (most common)
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                fill_exprs.append(pl.col(col).fill_null(mode_val[0]).alias(col))
            # If the column is all null, skip filling (no mode)
        # Return new DataFrame with filled columns
        return df.with_columns(fill_exprs)

    def _knn_impute_missing(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill missing values using KNN imputation.
        :param df: The DataFrame to fill missing values in.
        :type df: polars.DataFrame
        :returns: The DataFrame with missing values filled.
        :rtype: polars.DataFrame
        """
        # Get original schema and column order
        schema = df.schema
        col_order = df.columns
        # Convert to numpy array (float64 for KNNImputer)
        np_df = df.to_numpy().astype(np.float64)
        # Impute missing values
        knn_imputer = KNNImputer(n_neighbors=5, weights="uniform")
        imputed_np_df = knn_imputer.fit_transform(np_df)
        # Rebuild DataFrame with proper null handling for Int64 columns
        data = {}
        for idx, col in enumerate(col_order):
            col_data = imputed_np_df[:, idx]
            if schema[col] == pl.Int64:
                # Convert NaN to None for Int64 columns
                data[col] = [int(x) if not np.isnan(x) else None for x in col_data]
            else:
                data[col] = col_data
        # Create DataFrame and cast to original schema
        schema_dict: Mapping[str, pl.DataType] = dict(df.schema)
        return pl.DataFrame(data).cast(schema_dict) # type: ignore[arg-type]

    # Handle outliers with IQR method
    def _handle_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle outliers in a polars DataFrame by replacing them with median of the

        :param df: The DataFrame to handle outliers in.
        :type df: polars.DataFrame

        :returns: The DataFrame with outliers removed.
        :rtype: polars.DataFrame
        """
        # Select Float64 columns (or adjust for other numeric types)
        float_cols = [col for col in df.columns if df.schema[col] == pl.Float64]
        # Build expressions for outlier replacement
        exprs = []
        for col in float_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            median = df[col].median()

            # TODO: Skip or handle columns where quantile or median is None?
            if q1 is None or q3 is None or median is None:
                continue  # or handle as appropriate

            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            expr = (
                pl.when((pl.col(col) < lower_bound) | (pl.col(col) > upper_bound))
                .then(median)
                .otherwise(pl.col(col))
                .alias(col)
            )
            exprs.append(expr)

        return df.with_columns(exprs)
