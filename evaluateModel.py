import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle
from ResNet50_1D import *
from ConvNet_1D import *



if __name__ == "__main__":

    pickle_folder = 'pickle_files_Mica_nPoints_5120'    
    pickle_folder_full = f'FZs/{pickle_folder}'

    sample = 'Mica'
    model_name ='ConvNet-1D_v1_trained_on_Mica'
    model_name_short = 'ConvNet-1D'

    evaluation_folder = f'Models/{sample}/{model_name}/evaluation_on_{pickle_folder}'

    n_points = 5120

    model = get_ConvNet1D(n_points)
    # If you want to train ResNet50-1D, just use:
    #model = get_ResNet50_1D()

    model.load_weights(f'Models/{sample}/{model_name}/weights_{model_name}_best_val_loss.h5')
    

    Xi = pickle.load(open(os.path.join(pickle_folder_full,'Xi_val'), 'rb'))
    zi = pickle.load(open(os.path.join(pickle_folder_full,'zi_val'), 'rb'))
    yi = pickle.load(open(os.path.join(pickle_folder_full,'yi_val'), 'rb'))
    X = pickle.load(open(os.path.join(pickle_folder_full,'X_val'), 'rb'))
    z = pickle.load(open(os.path.join(pickle_folder_full,'z_val'), 'rb'))
    y = pickle.load(open(os.path.join(pickle_folder_full,'y_val'), 'rb'))

    if os.path.exists(evaluation_folder):
        files = glob.glob(f'{evaluation_folder}/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir(evaluation_folder)

    
    Xi = np.array(Xi).reshape(-1,n_points,1)
    yi = np.asarray(yi)
    yi = np.squeeze(yi)
    
    y_hat = model.predict(Xi)    
    y_hat = np.squeeze(y_hat)

    
    error_list = []    
    
    for q in range(yi.shape[0]):
        error_list.append ( (z[q][y[q]] - z[q][int(y_hat[q]*(len(z[q])-1))] ))
        
    mean = sum(error_list) / len(error_list)
    variance = sum([((x - mean) ** 2) for x in error_list]) / len(error_list)
    res = variance ** 0.5

    print("mean: ", mean)
    print("res: ", res)

    nbins=40
    histRange = (-6,6)
    
    plt.hist(error_list, bins=nbins, range=histRange)
    plt.ylabel('counts', fontsize=14)
    plt.xlabel('zc - zc_pred (nm)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'{model_name_short}: evaluation on FZs on {sample} val set', fontsize=14)
    plt.savefig(os.path.join(
        f'{evaluation_folder}',
        f'Error_{model_name}_on_{pickle_folder}.png'))
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
        plt.scatter(
            z[q][int(y_hat[q]*(len(z[q])-1))] , X[q][int(y_hat[q]*(len(z[q])-1))],
            s=100,
            color='r',
            marker='x',
            zorder=3,
            label=f'zc_pred {model_name_short}'
        )
        plt.plot(z[q], X[q], zorder=1, color='y')
        
        plt.xlabel('z (nm)', fontsize=14)
        plt.ylabel('PSD (V)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    
             
    
    
