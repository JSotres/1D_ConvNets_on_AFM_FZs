import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
import pickle

def interpolate(z0, fz, n_points):
    """
    Scipt that Interpolates a 1D numpy array to a provided number of 
    points. It also receives an index for a position within the 1D array,
    and returns the index that would compare to the same relative 
    position within the interpolated array.
    """
    
    z = np.arange(fz.shape[0])
    z_interpolated = np.linspace(
        np.min(z),
        np.max(z),
        n_points
    )
    
    fz_interpolated = np.interp(z_interpolated, z, fz)
    
    z0_interpolated_index = np.abs(z_interpolated-z0).argmin()
    z0_interpolated_value = z_interpolated[z0_interpolated_index]
    
    if fz_interpolated[z0_interpolated_index] > fz_interpolated[z0_interpolated_index-1]:
        z0_interpolated_index -= 1
            
    return(z0_interpolated_index,
           z_interpolated,
           fz_interpolated)

def readFZs(DIR, n_points):
    """
    Script that:
      - Reads force measurements, and their labelled events.
      - Interpolates the force measurements (1D arrays) to 
        n_points (input parameter). For this, it makes use of 
        the function interpolate.
      - Normalizes the force measurements.
      - Returns numpy arrays for the interpolated force 
        measurements and labelled positions, both the the 
        normalized and non-normalized formats.
    """
    
    X = []
    y = []
    z= []
    Xi = []
    yi = []
    zi = []

    DATADIR = DIR
    q_old = -1
    for file in sorted(os.listdir(DATADIR)):
        q = file.split('_')
        if int(q[1]) != int(q_old):
            q_old = q[1]
            z_original, fz = np.loadtxt(os.path.join(DATADIR,file))
        else:
            z0 = int(np.loadtxt(os.path.join(DATADIR,file)))            
            z0i, zit, fi = interpolate(z0, fz, n_points)
            fi = (fi-np.mean(fi))/(fi.max()-fi.min())
            z0i_norm = z0i/(zit.shape[0]-1)
            Xi.append(fi)
            zi.append(zit)
            yi.append(np.array(z0i_norm))
            X.append(fz)
            z.append(z_original)
            y.append(z0)
            
    return(X, z, y, Xi, zi, yi)

if __name__ == "__main__":
    """
    Script that: 
    - Reads AFM force measurements and corresponding labelled points
      (as obtained from https://github.com/JSotres/labelFZ).
    - Interpolate the force measurements to the input parameter n_points.
    - Normalizes vertical and horizontal dimensions of the force measurements.
    - Splits the raw and normalized force measurements into train and validation sets.
    - Saves train and validation sets, for both raw and normalized force measurements,
      as pickle files.
    """


    parentFolder = 'FZs'
    fzsFolder = 'Mica'
    full_fzs_folder = f'{parentFolder}/{fzsFolder}'

    n_points = 5120

    testSize = 0.2

    pickleFolder = f'{parentFolder}/pickle_files_{fzsFolder}_nPoints_{n_points}'

    if os.path.exists(pickleFolder):
        files = glob.glob(f'{pickleFolder}/*')
        for f in files:
            os.remove(f)
    else:
        os.mkdir(pickleFolder) 

    X, z, y, Xi, zi, yi = readFZs(full_fzs_folder, n_points)
    
    (Xi_train, Xi_val, zi_train, zi_val, yi_train, yi_val) = train_test_split(
        Xi,
        zi,
        yi,
        test_size=testSize,
        random_state=42,
        shuffle=True)
    
    (X_train, X_val, z_train, z_val, y_train, y_val) = train_test_split(
        X,
        z,
        y,
        test_size=testSize,
        random_state=42,
        shuffle=True)

    
    pickle_out = open(os.path.join(pickleFolder, 'Xi_train'), 'wb')
    pickle.dump(Xi_train, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'zi_train'), 'wb')
    pickle.dump(zi_train, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'yi_train'), 'wb')
    pickle.dump(yi_train, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'Xi_val'), 'wb')
    pickle.dump(Xi_val, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'zi_val'), 'wb')
    pickle.dump(zi_val, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'yi_val'), 'wb')
    pickle.dump(yi_val, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'X_train'), 'wb')
    pickle.dump(X_train, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'z_train'), 'wb')
    pickle.dump(z_train, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'y_train'), 'wb')
    pickle.dump(y_train, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'X_val'), 'wb')
    pickle.dump(X_val, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'z_val'), 'wb')
    pickle.dump(z_val, pickle_out)
    pickle_out.close()

    pickle_out = open(os.path.join(pickleFolder, 'y_val'), 'wb')
    pickle.dump(y_val, pickle_out)
    pickle_out.close()

    
