import polars as pl
from proxiflow.config import Config
from proxiflow.utils import generate_trace
from .core_utils import check_columns

from typing import Dict, Any, List


class Normalizer:
    """
    A class for performing data normalizing tasks.
    """

    def __init__(self, config: Config):
        """
        Initialize a new Normalizer object with the specified configuration.

        :param config: A Config object containing the normalization configuration values.
        :type config: Config
        """
        self.config: Dict[str, Any] = config.normalization_config

    def execute(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize the specified DataFrame using the specified configuration.

        :param df: The DataFrame to normalize.
        :type df: polars.DataFrame
        :return: The normalized DataFrame.
        :rtype: polars.DataFrame
        """
        normalized_df = df

        # Apply min-max normalization
        min_max_cols: list[str] = self.config.get("min_max", [])
        if min_max_cols:
            try:
                normalized_df = self._min_max_normalize(normalized_df, min_max_cols)
            except Exception as e:
                trace = generate_trace(e, self._min_max_normalize)
                raise Exception(f"Trying min-max normalization: {trace}")

        # Apply z-score normalization
        z_score_cols: list[str] = self.config.get("z_score", [])
        if z_score_cols:
            try:
                normalized_df = self._z_score_normalize(normalized_df, z_score_cols)
            except Exception as e:
                trace = generate_trace(e, self._z_score_normalize)
                raise Exception(f"Trying z-score normalization: {trace}")

        # Apply log normalization
        log_cols: list[str] = self.config.get("log", [])
        if log_cols:
            try:
                normalized_df = self._log_normalize(normalized_df, log_cols)
            except Exception as e:
                trace = generate_trace(e, self._log_normalize)
                raise Exception(f"Trying log normalization: {trace}")

        return normalized_df

    def _min_max_normalize(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Applies min-max normalization to the specified columns of the given DataFrame.

        :param df: The DataFrame to normalize.
        :type df: polars.DataFrame
        :param columns: The columns to normalize.
        :type columns: List[str]
        :return: The normalized DataFrame.
        :rtype: polars.DataFrame
        """
        columns = check_columns(df, columns)
        # If no columns exist, return the original DataFrame
        if len(columns) == 0:
            return df

        exprs = []
        for col in columns:
            # We need to check types because df[col].min()/max() are not guaranteed to be numeric types
            dtype = df.schema[col]
            if dtype in (pl.Int64, pl.Float64): 
                min_val = df[col].min()
                max_val = df[col].max()
                if not (isinstance(min_val, (int, float)) and isinstance(max_val, (int, float))):
                    continue  
                if max_val - min_val == 0:
                    raise ValueError(f"Error normalizing min-max column {col}: division by zero")
                exprs.append(((pl.col(col) - min_val) / (max_val - min_val)).alias(col))
            else:
                exprs.append(pl.col(col))

        # For columns not in the normalization list, keep them unchanged
        for col in df.columns:
            if col not in columns:
                exprs.append(pl.col(col))

        # Return DataFrame with normalized columns
        return df.with_columns(exprs)

    def _z_score_normalize(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Applies z-score normalization to the specified columns of the given DataFrame.

        :param df: The DataFrame to normalize.
        :type df: polars.DataFrame
        :param columns: The columns to normalize.
        :type columns: List[str]
        :return: The normalized DataFrame.
        :rtype: polars.DataFrame
        """
        columns = check_columns(df, columns)
        if len(columns) == 0:
            return df

        exprs = []
        for col in df.columns:
            dtype = df.schema[col]
            if col in columns and dtype in (pl.Int64, pl.Float64):
                exprs.append(((pl.col(col) - pl.col(col).mean()) / pl.col(col).std(ddof=0)).alias(col))
            else:
                exprs.append(pl.col(col))
        return df.with_columns(exprs)

    def _log_normalize(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """
        Applies log normalization to the specified columns of the given DataFrame.

        :param df: The DataFrame to normalize.
        :type df: polars.DataFrame
        :param columns: The columns to normalize.
        :type columns: List[str]
        :return: The normalized DataFrame.
        :rtype: polars.DataFrame
        """
        columns = check_columns(df, columns)
        if len(columns) == 0:
            return df

        exprs = []
        for col in df.columns:
            dtype = df.schema[col]
            if col in columns and dtype in (pl.Int64, pl.Float64):
                exprs.append(((1 + pl.col(col)) / 2).log().alias(col))
            else:
                exprs.append(pl.col(col))
        return df.with_columns(exprs)
