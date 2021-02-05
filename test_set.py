"""
テストセットを使って検証します
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix

def load_data(path='./got_data/datasets288432.npz',height=288,width=432):
    """
    各データセットをロード
    """
    datasets = np.load(path)
    X_test,y_test = datasets['X_test'],datasets['y_test']
    X_test = X_test.reshape((len(X_test), height,width,1))
    y_test = y_test.reshape((len(y_test),1))
    datasets.close()
    return X_test,y_test

if __name__ == '__main__':
    X_test,y_test = load_data()
    model = tf.keras.models.load_model('./models/my_model288432.h5')
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(confusion_matrix(y_test, y_pred))
