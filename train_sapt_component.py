import os, sys, argparse
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import util
from util import  *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a single model to predict a SAPT0 component')

    # required argument : SAPT component
    parser.add_argument('component',
                        choices=['Total', 'Elst', 'Exch', 'Ind', 'Disp'],
                        help='SAPT0 Component to model')

    # optional argument : model name
    parser.add_argument('-n', '--name',
                        help='Save trained model with this name.')

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

    # optional arguments: training hyperparameters
    parser.add_argument('--epochs',
                        help='Number of epochs',
                        type=int,
                        default=200)
    parser.add_argument('--adam_lr',
                        help='Initial learning rate for the Adam optimizer',
                        type=float,
                        default=1e-4)

    args = parser.parse_args(sys.argv[1:])

    component = args.component

    # avoid overwriting an already trained model
    if args.name is not None:
        model_dir = f'./models/{component}_{args.name}'
        if os.path.isdir(model_dir):
            raise Exception(f'A model already exists at {model_dir}')
        os.makedirs(model_dir)
        print(f'Training {component} Model at {model_dir}')
    else:
        model_dir = None 
        print(f'Training {component} AP-Net Model')

    # load datasets
    ACSF_nmu = args.acsf_nmu
    APSF_nmu = args.apsf_nmu
    ACSF_eta = args.acsf_eta
    APSF_eta = args.apsf_eta
    ZAt, ZBt, GAt, GBt, IAt, IBt, RABt, yt = util.get_dataset('nma-training', component, ACSF_nmu, APSF_nmu, ACSF_eta, APSF_eta)
    ZAv, ZBv, GAv, GBv, IAv, IBv, RABv, yv = util.get_dataset('nma-validation', component, ACSF_nmu, APSF_nmu, ACSF_eta, APSF_eta)
    ZAe, ZBe, GAe, GBe, IAe, IBe, RABe, ye = util.get_dataset('nma-testing', component, ACSF_nmu, APSF_nmu, ACSF_eta, APSF_eta)

    # custom training loop
    model = util.make_model(component)

    adam_lr = args.adam_lr
    epochs = args.epochs
    print(adam_lr, epochs)
    optimizer = tf.keras.optimizers.Adam(adam_lr)
    best_mae = np.inf
    print('\nEpoch ||  Validation Errors                                             Best ')

    for epoch in range(epochs):
        inds = np.random.permutation(len(ZAt))
        yt_errs = [] # track training error each epoch
        for mi in inds:
            GAtb, GBtb = inflate(GAt[mi], GBt[mi])
            yt_err = util.train_single(model, optimizer, ZAt[mi], ZBt[mi], RABt[mi], GAtb, GBtb, IAt[mi], IBt[mi], np.array(yt[mi])).numpy()
            yt_errs.append(yt_err)
        yt_errs = np.array(yt_errs)

        yv_preds, yv_errs = [], []
        for i in range(len(ZAv)):
            GAvi, GBvi = inflate(GAv[i], GBv[i])
            yv_pred = model([ZAv[i], ZBv[i], RABv[i], GAvi, GBvi, IAv[i], IBv[i]]).numpy()
            yv_pred = np.sum(yv_pred)
            yv_preds.append(yv_pred)
            yv_errs.append(yv_pred - yv[i])

        yv_preds, yv_errs = np.array(yv_preds), np.array(yv_errs)
        epoch_mae = np.average(np.absolute(yv_errs))

        if epoch_mae < best_mae:
            best_mae = epoch_mae 
            if model_dir is not None:
                model.save(f'{model_dir}/model.h5')

            print_out_errors(epoch+1, yv_errs, ' *')
        else:
            print_out_errors(epoch+1, yv_errs)
