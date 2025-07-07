import pytest
import polars as pl
import numpy as np
from proxiflow.config import Config
from proxiflow.core import Engineer

CONFIG_FILE_PATH = "tests/data/config.yaml"


@pytest.fixture(scope="module")
def engineer():
    return Engineer(Config(CONFIG_FILE_PATH))


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [6, 7, 8, 9, 10],
        }
    )


class TestOneHotEncoding:
    """
    A test class for the one hot encoding in the proxiflow library.
    """

    def test_one_hot_encode(self, engineer):
        # Create a sample DataFrame with categorical columns
        df = pl.DataFrame(
            {
                "category1": ["a", "b", "a", "c", "c", "a"],
                "category2": ["x", "y", "y", "z", "x", "y"],
                "num1": [1, 2, 3, 4, 5, 6],
                "num2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        expected = pl.DataFrame(
            {
                "category1": ["a", "b", "a", "c", "c", "a"],
                "category2_x": [1, 0, 0, 0, 1, 0],
                "category2_y": [0, 1, 1, 0, 0, 1],
                "category2_z": [0, 0, 0, 1, 0, 0],
                "num1": [1, 2, 3, 4, 5, 6],
                "num2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        engineered_result = engineer._one_hot_encode(df, ["category2"])
        assert expected.equals(engineered_result)


class TestFeatureScaling:
    """
    A test class for the polynominal feature engineering in the proxiflow library.
    """

    def test_no_columns(self, engineer, df):
        """
        Test feature_scaling with an empty list of columns.

        Given a DataFrame and an empty list of columns,
        feature_scaling should return the original DataFrame unchanged.
        """
        assert df.equals(engineer._feature_scaling(df, [], 2))

    def test_single_degree(self, engineer, df):
        """
        Test feature_scaling with a single degree.

        Given a DataFrame and a single degree for polynomial features,
        feature_scaling should return the DataFrame with the correct polynomial features.
        """
        expected = df.with_columns((pl.col("A") ** 2).alias("A_2"))
        result = engineer._feature_scaling(df, ["A"], 2)
        assert expected.equals(result)

    def test_multiple_degrees(self, engineer, df):
        """
        Test feature_scaling with multiple degrees.

        Given a DataFrame and multiple degrees for polynomial features,
        feature_scaling should return the DataFrame with the correct polynomial features.
        """
        expected = df.with_columns(
            (pl.col("A") ** 2).alias("A_2"),
            (pl.col("A") ** 3).alias("A_3"),
            (pl.col("B") ** 2).alias("B_2"),
            (pl.col("B") ** 3).alias("B_3"),
        )
        result = engineer._feature_scaling(df, ["A", "B"], 3)
        assert expected.equals(result)
