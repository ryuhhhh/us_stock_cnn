"""
imageデータをndarrayに変換していきます
"""
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

IMAGES_PATH = './got_data/images'

def create_data_series(target_series,cmpress_ratio=0.71):
    """
    imageをndarrayに変換していく
    """
    X = []
    y = []
    count = 0
    for index in target_series.index:
        print(count)
        try:
            img_array = cv2.imread(os.path.join(IMAGES_PATH, index+'.jpeg'), cv2.IMREAD_GRAYSCALE)
            if type(img_array) is not np.ndarray or (img_array is None):
                print(index)
                print(type(img_array))
                continue
            y_value = target_series[index]
            if  type(y_value) is not np.int64 or np.isnan(y_value) or (y_value is None):
                print(index)
                print(y_value)
                continue
            # (288,432) -> (204,306)
            height,width = img_array.shape
            img_array2 = cv2.resize(img_array , (int(width*cmpress_ratio), int(height*cmpress_ratio)))
            # print(img_array2.shape)
            # plt.axis('off')
            # plt.imshow(img_array2, cmap='gray')
            # plt.show()
            count+=1
            X.append(img_array2)
            y.append(y_value)
            # if count==100:
            #     break
        except Exception as e:
            pass
    return X,y

if __name__ == '__main__':
    FILE_NAME = 'datasets204306'
    # target_series呼び出し
    target_series = pd.read_csv('./got_data/us_stock_target.csv',index_col=0,squeeze=True)
    # シャッフル
    target_series = target_series.sample(frac=1, random_state=42)
    # 訓練データ作成
    X,y = create_data_series(target_series)
    X_len = len(X)
    print(len(X),len(y))
    # 10% 15% 75% で分割
    X_test = X[:int(X_len*0.1)]
    y_test  = y[:int(X_len*0.1)]
    X_valid = X[int(X_len*0.1):int(X_len*0.25)]
    y_valid = y[int(X_len*0.1):int(X_len*0.25)]
    X_train = X[int(X_len*0.25):]
    y_train = y[int(X_len*0.25):]
    np.savez_compressed(f"./got_data/{FILE_NAME}.npz",X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid)
    np.savez_compressed(f"./got_data/{FILE_NAME}_test.npz",X_test=X_test,y_test=y_test)
