import polars as pl
from proxiflow.config import Config
from .core_utils import check_columns
from proxiflow.utils import generate_trace


class Engineer:
    """
    A class for performing feature engineering tasks.
    """

    def __init__(self, config: Config):
        """
        Initialize a new Engineer object with the specified configuration.

        :param config: A Config object containing the feature engineering configuration values.
        :type config: Config
        """
        self.config = config.feature_engineering_config
        print(self.config)

    def execute(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Perform feature engineering on the specified DataFrame using the specified configuration.

        :param df: The DataFrame to perform feature engineering on.
        :type df: polars.DataFrame
        :return: The DataFrame with the new features.
        :rtype: polars.DataFrame
        """
        engineered_df = df.clone()
        # Apply feature engineering

        if self.config["one_hot_encoding"]:
            # Perform feature engineering on the specified columns
            try:
                engineered_df = self.one_hot_encode(engineered_df, self.config["one_hot_encoding"])
            except Exception as e:
                trace = generate_trace(e, self.one_hot_encode)
                raise Exception(f"Trying one-hot encoding: {trace}")

        feature_scaling = self.config["feature_scaling"]
        if feature_scaling:
            # Perform feature scaling on the specified columns
            try:
                engineered_df = self.feature_scaling(
                    engineered_df, feature_scaling["columns"], feature_scaling["degree"]
                )
            except Exception as e:
                trace = generate_trace(e, self.feature_scaling)
                raise Exception(f"Trying polynomial feature scaling: {trace}")

        return engineered_df

    def _one_hot_encode(self, df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """
        One-hot encode the specified columns of the given DataFrame.

        :param df: The DataFrame to one-hot encode.
        :type df: polars.DataFrame
        :param columns: The columns to one-hot encode.
        :type columns: List[str]
        :return: The one-hot encoded DataFrame.
        :rtype: polars.DataFrame
        """

        columns = check_columns(df, columns)
        if len(columns) == 0:
            return df
        return df.to_dummies(columns=columns)

    def _feature_scaling(self, df: pl.DataFrame, columns: list[str], degree: int) -> pl.DataFrame:
        """
        Creates polynomial features of the given degree for the specified columns of the given DataFrame.

        :param df: The DataFrame to create polynomial features for.
        :type df: polars.DataFrame
        :param columns: The columns to make features from.
        :type columns: List[str]
        :param degree: The degree of the polynomial features to create.
        :type degree: int
        :return: The DataFrame with polynomial features.
        :rtype: polars.DataFrame
        """
        if not columns:
            return df

        columns = check_columns(df, columns)
        exprs = []

        for col in columns:
            dtype = df.schema[col]
            if dtype in (pl.Int64, pl.Float64):
                # Generate polynomial features for degrees 2 to degree
                exprs.extend([
                    (pl.col(col) ** i).alias(f"{col}_{i}") for i in range(2, degree + 1)
                ])
        # Add new columns to DataFrame
        return df.with_columns(exprs) if exprs else df
