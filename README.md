# FLUX-PREDICT

Implementation of the paper

> "Combining multi-target regression deep neural networks and kinetic modeling to predict relative fluxes in reaction systems"
> 
> by Lucrezia Patruno, Francesco Craighero, Davide Maspero, Alex Graudenzi, Chiara Damiani

## Requirements

All the results in the paper were obtained on `Python 3.8.5` with the following requirements:

```
tensorflow==2.4.1
scikit_learn==0.24.2
pandas==1.2.3
numpy==1.19.5
```

## Usage

To execute the script:

```
python grid_search.py
```

The outputs will be saved in the `results` folder.

## Output

The script will save:
- the train/test partition as `X_train.zip`, `y_train.zip`, `X_test.zip`, `y_test.zip`.
- the `sklearn.model_selection.GridSearchCV` output in `CVcomplete_results.pkl`.
- the `tensorflow` best model in `CVcomplete_best_estimator.h5`.

To load the model perform the following steps (remember to change `path_to_model`):

```
import tensorflow as tf
from model import r2_metric

model = tf.keras.models.load_model(
  "path_to_model/CVcomplete_best_estimator.h5", 
  custom_objects={"r2_metric": r2_metric})

```
## Data

To explore the data in the `data` folder, use the `pandas` library as follows:

```
import pandas as pd
pd.read_csv("data/X_full.zip")
```

