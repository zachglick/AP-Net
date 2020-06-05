import os, sys, argparse, math
from multiprocessing import Pool
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import util
from util import  *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a single model to predict all four SAPT0 components')

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
    parser.add_argument('--batch_size',
                        help='Batch size for training (number of dimers)',
                        type=int,
                        default=8)
    parser.add_argument('--decay_rate',
                        help='learning rate decays by this factor every epoch',
                        type=float,
                        default=(10 ** (-1 / 30.0)))
                        

    args = parser.parse_args(sys.argv[1:])

    adam_lr = args.adam_lr
    epochs = args.epochs
    batch_size = args.batch_size
    decay_rate = args.decay_rate
    ACSF_nmu = args.acsf_nmu
    APSF_nmu = args.apsf_nmu
    ACSF_eta = args.acsf_eta
    APSF_eta = args.apsf_eta

    # avoid overwriting an already trained model
    if args.name is not None:
        model_dir = f'./models/{args.name}'
        if os.path.isdir(model_dir):
            raise Exception(f'A model already exists at {model_dir}')
        os.makedirs(model_dir)
        print(f'Training AP-Net Model at {model_dir}')
    else:
        model_dir = None 
        print(f'Training AP-Net Model')

    dimert, labelt = util.get_dimers('nma-training')
    dimerv, labelv = util.get_dimers('nma-validation')
    #dimere, labele = util.get_dimers('nma-testing')

    Nt = len(dimert)
    Nv = len(dimerv)
    #Ne = len(dimere)
    Nb = math.ceil(Nt / batch_size)

    print(f'  Epochs:     {epochs}')
    print(f'  Adam LR:    {adam_lr}')
    print(f'  LR decay:   {decay_rate}')
    print(f'  Batch size: {batch_size}')
    print(f'  ACSF count: {ACSF_nmu}')
    print(f'  ACSF eta:   {ACSF_eta}')
    print(f'  APSF count: {APSF_nmu}')
    print(f'  APSF eta:   {APSF_eta}')
    print(f'  T/V split:  {Nt}/{Nv}')

    # custom training loop
    model = util.make_model()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = adam_lr,
        decay_steps = Nb,
        decay_rate = decay_rate,
        staircase=True, 
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    best_mae = np.inf
    print('\nEpoch ||  Validation Errors                                             Best ')

    for epoch in range(epochs):

        feature_time = 0.0
        start = time.time()

        inds = np.random.permutation(Nt).astype(int)
        yt_errs = [] # track training error each epoch

        dimer_batches, label_batches = util.make_batches(dimert, labelt, batch_size, order=inds)
        for dimer_batch, label_batch in zip(dimer_batches, label_batches):
            feature_start = time.time()
            feature_batch = [util.make_features(*d) for d in dimer_batch]
            feature_time += (time.time() - feature_start)
            yt_err = util.train_batch(model, optimizer, feature_batch, label_batch)
            yt_errs.append(yt_err)

        yt_errs = np.concatenate(yt_errs)
        print()
        print_out_errors_comp(-1, yt_errs, ' T')

        yv_preds, yv_errs = [], []

        start2 = time.time()
        dimer_batches, label_batches = util.make_batches(dimerv, labelv, batch_size)
        for dimer_batch, label_batch in zip(dimer_batches, label_batches):
            feature_start = time.time()
            feature_batch = [util.make_features(*d) for d in dimer_batch]
            feature_time += (time.time() - feature_start)
            for fv, lv in zip(feature_batch, label_batch):
                yv_pred = util.predict_single(model, fv)
                yv_pred = np.sum(yv_pred, axis=0)
                yv_preds.append(yv_pred)
                yv_errs.append(yv_pred - lv)

        yv_preds, yv_errs = np.array(yv_preds), np.array(yv_errs)
        #epoch_mae = np.average(np.absolute(yv_errs))
        epoch_mae = np.average(np.absolute(np.sum(yv_errs, axis=1)))

        if epoch_mae < best_mae:
            best_mae = epoch_mae 
            if model_dir is not None:
                model.save(f'{model_dir}/model.h5')

            print_out_errors_comp(epoch+1, yv_errs, ' *')
        else:
            print_out_errors_comp(epoch+1, yv_errs)
        print(f'          Epoch Time (s): Total={int(time.time() - start)}, Features={int(feature_time)}, Val={int(time.time() - start2)}')