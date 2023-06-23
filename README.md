# Attention Based Spatial-Temporal Graph Convolutional Recurrent Networks for Traffic Forecasting

## Structure

- config: a configuration file for each of the five datasets.
- data: contains the datasets used in our experiments: PEMS03, PEMS04, PEMS07, PEMS08, PEMS07(M), and DeathsCount_N.
- lib: contains the custom modules we worked with, such as data loading, data pre-processing, evaluation metrics and normalisation.
- logs: stores log files.
- models: our model code.
- BasicTrainer.py: training, validating and predicting the model
- Run.py: running the program

## Requirements

- python=3.8.12
- pytorch
- numpy
- configparser
- argparse
- logging

## Training models

To replicate the results of running the dataset, you can directly run the "Run.py" file. As an example of training our T-ASTGCRN on the PEMS04 dataset (if there are deviations, see the run logs we provide):

```
python Run.py --weight_decay=0.001 --embed_dim=10
```

If you want to run other datasets or our other two attention-based models, please modify the code in "Run.py" and "models/ASTGCRN.py" for them. If you want to use the model for your own dataset, please load your dataset by modifying "load_dataset" in the "lib" folder and creating a configuration file in the 'config' file.