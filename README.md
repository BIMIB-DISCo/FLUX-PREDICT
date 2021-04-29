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

To execute the default script execute:

```
python grid_search.py
```

The outputs will be saved in the `results` folder.



## Data

To explore the data in the `data` folder, use the `pandas` library as follows:

```
import pandas as pd
pd.read_csv("data/X_full.zip")
```

