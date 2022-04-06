import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform
import math
import os
import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

import pickle

from ResNet50_1D import *
from ConvNet1D import *



if __name__ == "__main__":

    pickle_folder = 'Pickle_Mica_nPoints_5120'
    pickle_folder_full = f'FZs/Mica/{pickle_folder}'

    Xi = pickle.load(open(os.path.join(pickle_folder_full,'Xi'), 'rb'))
    zi = pickle.load(open(os.path.join(pickle_folder_full,'zi'), 'rb'))
    yi = pickle.load(open(os.path.join(pickle_folder_full,'yi'), 'rb'))
    X = pickle.load(open(os.path.join(pickle_folder_full,'X'), 'rb'))
    z = pickle.load(open(os.path.join(pickle_folder_full,'z'), 'rb'))
    y = pickle.load(open(os.path.join(pickle_folder_full,'y'), 'rb'))

    n_points = 5120

    model_name_1 ='ConvNet1D_v2_trained_on_Mica'

    
    evaluation_folder_1 = f'Models/Mica/{model_name_1}/evaluation_on_{pickle_folder}'

    if os.path.exists(evaluation_folder_1):
        files1 = glob.glob(f'{evaluation_folder_1}/*')
        for f in files1:
            os.remove(f)
    else:
        os.mkdir(evaluation_folder_1)

    model1 = get_ConvNet1D(n_points)
    
    model1.load_weights(f'Models/Mica/{model_name_1}/weights_{model_name_1}_best_val_loss.h5')

    

    Xi = np.array(Xi).reshape(-1,n_points,1)
    yi = np.asarray(yi)

    

    y_hat1 = model1.predict(Xi)

    
    yi = np.squeeze(yi)
    y_hat1 = np.squeeze(y_hat1)

    
    mae_list1 = []
    fz_z_interval = []
    fz_z_range = []

    
    
    for q in range(yi.shape[0]):

        mae_list1.append ( (z[q][y[q]] - z[q][int(y_hat1[q]*(len(z[q])-1))] ))
        fz_z_interval.append ( np.abs (z[q][0]-z[q][1]))
        fz_z_range.append ( np.abs (z[q][0]-z[q][-1]))
    

        
    mean1 = sum(mae_list1) / len(mae_list1)
    variance1 = sum([((x - mean1) ** 2) for x in mae_list1]) / len(mae_list1)
    res1 = variance1 ** 0.5

    print("mean1: ", mean1)
    print("res1: ", res1)

    nbins=40
    histRange = (-6,6)
    
    plt.hist(mae_list1, bins=nbins, range=histRange)
    plt.ylabel('counts', fontsize=14)
    plt.xlabel('zc - zc_pred (nm)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.title(f'evaluation_on_{pickle_folder}')
    plt.title(f'ConvNet-1D: evaluation on FZs on Mica test set', fontsize=14)
    plt.savefig(os.path.join(
        f'{evaluation_folder_1}',
        f'MAE_Raw_{model_name_1}_on_{pickle_folder}.png'))
    plt.show()

    
    

    for q in range(len(y)):
        plt.scatter(
            z[q][y[q]], X[q][y[q]],
            s=120,
            c='g',
            marker='o',
            zorder=2,
            label='zc'
        )
        plt.scatter(z[q][int(y_hat1[q]*(len(z[q])-1))] , X[q][int(y_hat1[q]*(len(z[q])-1))],
            s=100,
            color='k',
            marker='v',
            zorder=3,
            label='zc_pred ConvNet-1D'
        )
        plt.scatter(
            z[q][int(y_hat2[q]*(len(z[q])-1))] , X[q][int(y_hat2[q]*(len(z[q])-1))],
            s=100,
            color='r',
            marker='x',
            zorder=4,
            label='zc_pred ResNet50-1D'
        )
        plt.plot(z[q], X[q], zorder=1, color='y')
        
        plt.xlabel('z (nm)', fontsize=14)
        plt.ylabel('PSD (V)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    
             
    
    
