[![PyPi version](https://badgen.net/pypi/v/proxiflow/)](https://pypi.org/project/proxiflow)
[![Documentation Status](https://readthedocs.org/projects/proxiflow/badge/?version=v0.1.10)](https://proxiflow.readthedocs.io/en/latest/?badge=v0.1.10)
[![PyPI download month](https://img.shields.io/pypi/dm/proxiflow.svg)](https://pypi.python.org/pypi/proxiflow/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/itsmartintomes/proxiflow/graphs/commit-activity)
[![PyPI license](https://img.shields.io/pypi/l/proxiflow.svg)](https://pypi.python.org/pypi/proxiflow/)
[![tests](https://github.com/itsmartintomes/proxiflow/actions/workflows/tests.yml/badge.svg?branch=v0.1.10)](https://github.com/itsmartintomes/proxiflow/actions/workflows/tests.yml)


# ProxiFlow

ProxiFlow is a data preprocessig tool for any kind of data processing (eg. machine learning) that performs
data cleaning, normalization, and feature engineering.

The biggest advantage of this library (which is basically a wrapper over [polars](https://github.com/pola-rs/polars) data frame) is that it is configurable via YAML configuration file which makes it suitable for MLOps pipelines or for building API requests over it.

## Documentation
Read the full documentation [here](http://proxiflow.readthedocs.io/).

## Usage

To use ProxiFlow, install it via pip:

```bash
pip install proxiflow
```

You can then call it from the command line:

```bash
proxiflow --config-file myconfig.yaml --input-file mydata.csv --output-file cleaned_data.csv
```

Here\'s an example of a YAML configuration file:

```yaml
input_format: csv
output_format: csv

data_cleaning: #mandatory
  # NOTE: Not handling missing values can cause errors during data normalization
  handle_missing_values:
    drop: false
    mean: true # Only Int and Float columns are handled 
    # mode: true # Turned off for now. 
    knn: true

  handle_outliers: true # Only Float columns are handled
  remove_duplicates: true

data_normalization: # mandatory
  min_max: #mandatory but values are not mandatory. It can be left empty
    # Specify columns:
    - Age # not mandatory
  z_score: 
    - Price 
  log:
    - Floors

feature_engineering:
  one_hot_encoding: # mandatory
    - Bedrooms      # not mandatory

  feature_scaling:  # mandatory
    degree: 2       # not mandatory. It specifies the polynominal degree
    columns:        # not mandatory
      - Floors      # not mandatory
```

The above configuration specifies that duplicate rows should be removed
and missing values should be dropped.

## API

ProxiFlow can also be used as a Python library. Here\'s an example:

```python
import polars as pl
from proxiflow.config import Config
from proxiflow.core import Cleaner

# Load the data
df = pl.read_csv("mydata.csv")

# Load the configuration
config = Config("myconfig.yaml")

# Clean the data
cleaner = Cleaner(config)
cleaned_data = cleaner.clean_data(data)

# Perform data normalization
normalizer = Normalizer(config)
normalized_data = normalizer.normalize(cleaned_data)

# Perform feature engineering
engineer = Engineer(config)
engineered_data = engineer.execute(normalized_data)

# Write the output data
engineered_data.write_csv("cleaned_data.csv")
```

## Log

-   \[x\] Data cleaning
    - \[x\] Missing values handling
        - \[x\] Mean
        - \[x\] Drop
        - \[x\] KNN Imputer 
        - \[x\] Median
-   \[x\] Data normalization
    - \[x\] Min Max normalization
    - \[x\] Z-Score normalization
    - \[x\] Logarithmic normalization
-   \[ ] Feature engineering
    - \[x\] One Hot Encoding
    - \[x\] Feature Scaling
    - \[ ] Recursive Feature Elimination
    - \[ ] SelectKBest
    - \[ ] LASSO regularization
-   \[ ] Text Preprocessing
    - \[ ] Tokenization
    - \[ ] Stemming
    - \[ ] Stopword removal
    - \[ ] Text Vectorization
        - \[ ] Bag of Words
        - \[ ] TF-IDF
    - \[ ] Word embeddings
        -  \[ ] Word2Vec
        -  \[ ] GloVe
        -  \[ ] BERT
- \[ ] Categorical Encoding
    - \[ ] Target encoding
    - \[ ] Count encoding
    - \[ ] Binary encoding
- \[ ] Dimensionality reduction
    - \[ ] Principal Component Analysis (PCA)
    - \[ ] t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - \[ ] Uniform Manifold Approximation and Projection (UMAP) 
