import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
import math
import os
import glob
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

import pickle

from ResNet50_1D import *
from ConvNet_1D import *



if __name__ == "__main__":

    pickle_folder = 'FZs/pickle_files_Mica_nPoints_5120'
    

    n_points = 5120
    optimazerAlg = 'Adam'
    LossF = 'mse'
    MetricsF = 'mae'
    learningRate = 0.0001
    batchSize = 32
    nEpochs = 6
    modelName = 'ConvNet-1D'
    modelVersion = 'v1'
    sample = 'Mica'

    fileModelEvaluation = f'evaluation_ models_trained_on_{sample}.csv'

    # If you want to train ConvNet-1D
    model = get_ConvNet1D(n_points)

    # If you want to train ResNet50-1D, just use:
    #model = get_ResNet50_1D()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
    model.compile(optimizer=optimizer, loss=LossF, metrics=[MetricsF])

    
    Xi_train = pickle.load(open(os.path.join(pickle_folder,'Xi_train'), 'rb'))
    zi_train = pickle.load(open(os.path.join(pickle_folder,'zi_train'), 'rb'))
    yi_train = pickle.load(open(os.path.join(pickle_folder,'yi_train'), 'rb'))
    X_train = pickle.load(open(os.path.join(pickle_folder,'X_train'), 'rb'))
    z_train = pickle.load(open(os.path.join(pickle_folder,'z_train'), 'rb'))
    y_train = pickle.load(open(os.path.join(pickle_folder,'y_train'), 'rb'))

    Xi_val = pickle.load(open(os.path.join(pickle_folder,'Xi_val'), 'rb'))
    zi_val = pickle.load(open(os.path.join(pickle_folder,'zi_val'), 'rb'))
    yi_val = pickle.load(open(os.path.join(pickle_folder,'yi_val'), 'rb'))
    X_val = pickle.load(open(os.path.join(pickle_folder,'X_val'), 'rb'))
    z_val = pickle.load(open(os.path.join(pickle_folder,'z_val'), 'rb'))
    y_val = pickle.load(open(os.path.join(pickle_folder,'y_val'), 'rb'))    

    Xi_train = np.array(Xi_train).reshape(-1,n_points,1)
    yi_train = np.asarray(yi_train)

    Xi_val = np.array(Xi_val).reshape(-1,n_points,1)
    yi_val = np.asarray(yi_val)

    if not os.path.exists('Models'):
        os.mkdir('Models')

    if not os.path.exists(f'Models/{sample}'):
        os.mkdir(f'Models/{sample}')
            


    if os.path.exists(f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}'):
        files = glob.glob(f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir(f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}')

    checkpoint_filepath = f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}/weights_{modelName}_{modelVersion}_trained_on_{sample}_best_val_loss.h5'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    history = model.fit(
        Xi_train,
        yi_train,
        epochs=nEpochs,
        batch_size=batchSize,
        validation_data=(Xi_val, yi_val),
        verbose=2,
        callbacks=[model_checkpoint_callback],
        shuffle=True
    )

    model.save_weights(os.path.join(
        f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}', f'weights_{modelName}_{modelVersion}_trained_on_{sample}.h5'
    ))

    model_data = {
        'model_name': modelName,
        'model_version': modelVersion,
        'sample': sample,
        'n_points': n_points,
        'optimazer': optimazerAlg,
        'loss': LossF,
        'metrics': MetricsF,
        'learning_rate': learningRate,
        'batch_size': batchSize,
        'epochs': nEpochs,
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_train_metrics': history.history[MetricsF][-1],
        'final_val_metrics': history.history[f'val_{MetricsF}'][-1],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'val_metrics': history.history[f'val_{MetricsF}']
        }

    if os.path.isfile(os.path.join(f'Models/{sample}', fileModelEvaluation)):
        df = pd.read_csv(os.path.join(f'Models/{sample}', fileModelEvaluation), index_col=False)
    else:
        df=pd.DataFrame(columns=[
            'model_name',
            'model_version',
            'sample',
            'n_points',
            'optimazer',
            'loss',
            'metrics',
            'learning_rate',
            'batch_size',
            'epochs',
            'final_train_loss',
            'final_val_loss',
            'final_train_metrics',
            'final_val_metrics',
            'train_loss',
            'val_loss',
            'val_metrics'
        ])

    df = df.append(model_data, ignore_index=True)

    df.to_csv(os.path.join(f'Models/{sample}', fileModelEvaluation), index=False)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'model loss ({LossF}) - {modelName}_{modelVersion}_trained_on_{sample}')
    plt.ylabel(f'loss ({LossF})')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(os.path.join(f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}', f'Evaluation_Loss_{modelName}_{modelVersion}_trained_on_{sample}.png'))
    plt.show()

    plt.plot(history.history[MetricsF])
    plt.plot(history.history[f'val_{MetricsF}'])
    plt.title(f'{MetricsF} - {modelName}_{modelVersion}_trained_on_{sample}')
    plt.ylabel(MetricsF)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(os.path.join(f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}', f'MAE_Interpolated_{modelName}_{modelVersion}_trained_on_{sample}.png'))
    plt.show()

    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(
            f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}',
            f'Architecture_{modelName}_{modelVersion}_trained_on_{sample}.png'),
        show_shapes=True)

    error_list = []

    y_hat = model.predict(Xi_val)
    y_hat = np.squeeze(y_hat)

    for q in range(yi_val.shape[0]):
        error_list.append(z_val[q][y_val[q]] - z_val[q][int(y_hat[q]*(len(z_val[q])-1))])
        

    nbins=40

    plt.hist(error_list, bins=nbins)
    plt.ylabel('counts')
    plt.xlabel('zc-zc_pred (nm)')
    plt.title(f'zc-zc_pred - {modelName}_{modelVersion}_trained_on_{sample}')
    plt.savefig(os.path.join(
        f'Models/{sample}/{modelName}_{modelVersion}_trained_on_{sample}',
        f'Error_{modelName}_{modelVersion}_trained_on_{sample}.png'))
    plt.show()

    
