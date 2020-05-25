# AP-Net: An atomic-pairwise neural network for smooth and transferable interaction potentials

This repository contains code and data associated with the [AP-Net paper](https://aip.scitation.org/journal/jcp).

## Dependencies

This project uses Python 3 and requires up-to-date versions of the following Python packages:
* numpy
* scipy
* pandas
* tensorflow (2.1.0 or newer)

## Data 

Datasets are stored as serialized pandas dataframes named `dimers.pkl` in separate subdirectories of `datasets/`.
Each row of a dataframe is an individual dimer.
The columns `RA` and `RB` are atomic coordinates (in Angstrom) and the columns `ZA` and `ZB` are atomic numbers.
For all datasets, the SAPT0/jun-cc-pVDZ interaction energy is stored in the 'Total' column, and the four components that sum to this energy are stored in additional columns: `elst`, `exch`, `ind`, and `disp`.

## Pretrained model

The trained AP-Net model described in the paper can be used to perform inferences on any included or added datasets.
This is demonstrated in the script `infer.py`

## Training a new model

A new AP-Net model can be trained on an arbitrary dataset of dimer interaction energies.
This is demonstrated in the script `train.py`
