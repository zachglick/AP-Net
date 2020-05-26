# AP-Net: An atomic-pairwise neural network for smooth and transferable interaction potentials

This repository contains code and data associated with the [AP-Net paper](https://aip.scitation.org/journal/jcp).

## Dependencies

This project uses Python 3 and requires up-to-date versions of the following Python packages:
* numpy
* scipy
* pandas
* tensorflow (2.1.0 or newer)

## Data 

Datasets of dimer interaction energies are stored as serialized pandas dataframes named `dimers.pkl` in separate subdirectories of `datasets/`.
Each row of a dataframe is an individual dimer.
The columns `RA` and `RB` are monomer atomic coordinates (in Angstrom) and the columns `ZA` and `ZB` are monomer atomic numbers.
For all datasets, the SAPT0/jun-cc-pVDZ interaction energy is stored in the `Total` column, and the four components that sum to this energy are stored in additional columns: `elst`, `exch`, `ind`, and `disp`.

To use your own dataset with AP-Net, add an additional subdirectory to `datasets` with a `dimers.pkl` file. `RA`, `RB`, `ZA`, and `ZB` are required columns.

AP-Net requires atom-pair descriptors for both inference and training. Instead of repeatedly recalculating these descriptors (also called features) on the fly, they are cached in the `datasets/` subdirectory after creation for future use.

## Using a pre-trained model

The trained AP-Net model described in the paper can be used to perform inferences on any included or added datasets.
This is demonstrated in the script `infer_sapt.py`, which predicts the interaction energy (total SAPT0 and components) for all dimers in a dataset.
Run as: ```>>> python infer_sapy.py --help``` for more information.


## Training a new model

A new AP-Net model can be trained on an arbitrary dataset of SAPT0 dimer interaction energies.
This is demonstrated in the script `train_sapt_component.py` which trains a single model to predict a single SAPT0 component.
Run as: ```>>> python train_sapt_component.py --help``` for more information.


