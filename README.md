# 1D_ConvNets_on_AFM_FZs

Repository for the manuscript: Locating critical events in AFM force measurements by means of one-dimensional convolutional neural networks.

More specific description will be uploded after the publication of the manuscript.

Briefly:

All data used in the manuscript can be found in the folder *FZs*.

The file *prepareDataSets.py* interpolates and normalizes the data, and saves it as pickle files that can be read by the *trainModel.py* file, which trains specific models on that data.

The Keras implementation of the two models used in the manuscript can be found in the files *ConvNet_1D.py* and *ResNet50_1D.py*.

The file *evaluateModel.py* can be used to evaluate the trained models