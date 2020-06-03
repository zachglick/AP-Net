import os, sys, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import util
from util import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict SAPT0 interaction energy with a trained AP-Net model')
    parser.add_argument('dataset',
                        help='Dataset to test trained model on (nma-testing, s66x8, etc.)')

    # optional arguments: feature hyperparameters
    parser.add_argument('--acsf_nmu',
                        help='ACSF hyperparameter (number of radial gaussians).',
                        type=int,
                        default=43)
    parser.add_argument('--apsf_nmu',
                        help='APSF hyperparameter (number of angular gaussians).',
                        type=int,
                        default=21)
    parser.add_argument('--acsf_eta',
                        help='ACSF hyperparameter (radial gaussian width).',
                        type=int,
                        default=100)
    parser.add_argument('--apsf_eta',
                        help='APSF hyperparameter (angular gaussian width).',
                        type=int,
                        default=25)

    args = parser.parse_args(sys.argv[1:])
    modelnames = [f'paper{i}' for i in range(1,9)]
    dataset = args.dataset

    # feature hyperparameters
    ACSF_nmu = args.acsf_nmu
    APSF_nmu = args.apsf_nmu
    ACSF_eta = args.acsf_eta
    APSF_eta = args.apsf_eta

    # store predictions, labels and errors for each SAPT component in a dictionary
    prds = {}
    labs = {}
    errs = {}

    # iterate over the four SAPT0 components
    components = ['Elst', 'Exch', 'Ind', 'Disp']

    for c in components:
                    
        # load dataset (features and labels) for this component
        ZA, ZB, GA, GB, IA, IB, RAB, y = util.get_dataset(dataset, c, ACSF_nmu, APSF_nmu, ACSF_eta, APSF_eta)
        lab = np.array(y) # make sure it's not a list
        #lab = np.array([None for _ in y])

        print(f'\nEvaluating {c} with AP-Net')

        # load and infer with all eight models to predict this ensemble
        for modelind, modelname in enumerate(modelnames):

            # load the model
            model_dir = f'./models/{c}_{modelname}'
            model = tf.keras.models.load_model(f'{model_dir}/model.h5', compile=False)
            #print(f'Evaluating {c} with AP-Net ensemble member {i+1} out of 8')

            prd = []
            for i in range(len(ZA)):
                GAi, GBi = inflate(GA[i], GB[i])
                prd.append(np.sum(model([ZA[i], ZB[i], RAB[i], GAi, GBi, IA[i], IB[i]]).numpy()))

            prd = np.array(prd)
            prds[(c, modelname)] = prd
            labs[(c, modelname)] = lab

            # print errors if label exists
            if np.all(lab != None):
                err = prd - lab
                aerr = np.absolute(err)
                print(f' Ensemble member {modelind+1} MAE : {np.average(aerr):8.3f}')
            else:
                print('None')

        prd_avg = np.average([prds[(c, modelname)] for modelname in modelnames], axis=0)

        # print errors if label exists
        if np.all(lab != None):
            err_avg = prd_avg - lab
            aerr_avg = np.absolute(err_avg)
            print(f' Ensemble total    MAE : {np.average(aerr_avg):8.3f}')
        else:
            print('None')

    for modelname in modelnames:
        prds[('Total', modelname)] = np.sum([prds[(c, modelname)] for c in components], axis=0)
        try:
            labs[('Total', modelname)] = np.sum([labs[(c, modelname)] for c in components], axis=0)
        except:
            labs[('Total', modelname)] = labs[components[0], modelname]

    components.insert(0, 'Total')

    for c in components:
        prds[(c,)] = np.stack([prds[(c, modelname)] for modelname in modelnames])
        labs[(c,)] = labs[(c, modelnames[0])].reshape(1,1,-1)

    prds_all = np.stack([prds[(c,)] for c in components])
    labs_all = np.stack([labs[(c,)] for c in components])
    np.save(f'{dataset}-predictions', prds_all)
    np.save(f'{dataset}-labels', labs_all)

    if np.all(lab != None):
        errs_all = prds_all - labs_all
        np.save(f'{dataset}-errors', errs_all)
