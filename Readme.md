# INLP ASSIGNMENT - 4

## Pre-training ELMO

1. **utils.py** - Contains the dataset class and the ElMO model and functions to train the model.
2. **ELMO.py** - Calls the functions in utils.py to train and save the model.

- To train the model, run the following command:
    ``` bash
    python3 ELMO.py
    ```
    The model will be saved as 'model.pt' in the current directory.

## Downstream Task

1. **classification_utils.py** - Contains the dataset class and the classifier model and functions to train and test the model.
2. **classification.py** - Loads the ELMO model and calls the functions in classification_utils.py to train and test the model.

- There is **method** argument passed to the classifier function. 

if method == '1': trainable lambdas   
if method == '2': fixed lambdas   
if method == '3': learnable function

- To train the model, run the following command:
    ``` bash
    python3 classification.py
    ```
    The model will be saved as 'classification_model_{method}.pt' in the current directory.

**Link for saved models:** [Google Drive](https://drive.google.com/drive/folders/1pPWVFrfkkxT4nLYgj5dueSqfdnXRZ_I4?usp=sharing)