"""
CNNで株価が上昇するか2項分類します
"""
import tensorflow as tf
from tensorflow import keras
import os
import time
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix

def make_model():
    model = keras.models.Sequential([
        # 64個のフィルタ,サイズ7x7,28x28ピクセルグレースケール(チャネル1)
        # keras.layers.Conv2D(64,7,activation='relu',padding='same',input_shape=[288,432,1]),
        keras.layers.Conv2D(64,7,activation='relu',padding='same',input_shape=[204,306,1]),
        # プーリングサイズ2 => サイズが1/2になる
        keras.layers.MaxPooling2D(2),
        # フィルタの数が2倍
        keras.layers.Conv2D(128,3,activation='relu',padding='same'),
        keras.layers.Conv2D(128,3,activation='relu',padding='same'),
        keras.layers.MaxPooling2D(2),
        # フィルタの数が更に2倍
        keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        keras.layers.Conv2D(256,3,activation='relu',padding='same'),
        keras.layers.MaxPooling2D(2),
        # 全結合層を作成する
        keras.layers.Flatten(),
        # keras.layers.Dense(128,activation='elu'),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dropout(0.5),
        # 出力層は1クラスの2項分類
        keras.layers.Dense(1,activation='sigmoid'),
    ])
    return model

def load_data(path='./got_data/datasets288432.npz',height=288,width=432):
    """
    各データセットをロード
    """
    datasets = np.load(path)
    X_train,y_train = datasets['X_train'],datasets['y_train']
    X_valid,y_valid = datasets['X_valid'],datasets['y_valid']
    datasets.close()
    X_train = X_train.reshape((-1, height,width,1))
    y_train = y_train.reshape((-1,1))
    X_valid = X_valid.reshape((-1, height,width,1))
    y_valid = y_valid.reshape((-1,1))
    total = len(y_train)
    neg = np.count_nonzero(y_train == 0)
    pos = total - neg
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f'class_weight:{class_weight}')
    return X_train,y_train,X_valid,y_valid,class_weight

def custom_total_true(y_true, y_pred):
    return K.sum(y_true)

def custom_total_pred(y_true, y_pred):
    return K.sum(y_pred)

def custom_recall(y_true, y_pred):
    """クラス1のRecall"""
    true_positives = K.sum(y_true * y_pred)
    total_positives = K.sum(y_true)
    return true_positives / (total_positives + K.epsilon())

def custom_precision(y_true, y_pred):
    """クラス1のPrecision"""
    total_1_predictions = K.sum(y_pred)
    total_true_predictions = K.sum(y_true*y_pred)
    return total_true_predictions/(total_1_predictions+K.epsilon())

def main():
    model = make_model()
    optimizer = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy",
                 custom_total_true,
                 custom_total_pred,
                 custom_recall,
                 custom_precision
                 ]
    )
    X_train,y_train,X_valid,y_valid,class_weight =\
        load_data(path='./drive/MyDrive/Colab Notebooks/datasets204306.npz',height=204,width=306)
    checkpoint = keras.callbacks.ModelCheckpoint('./drive/MyDrive/Colab Notebooks/my_model204306.h5')
    early_stopping = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
    print(model.summary())
    history = model.fit(X_train,
                        y_train,
                        batch_size=32,
                        epochs=30,
                        validation_data=(X_valid,y_valid),
                        class_weight=class_weight,
                        callbacks=[checkpoint,early_stopping])

if __name__ == '__main__':
    main()