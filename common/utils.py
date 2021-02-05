import pandas as pd
import numpy as np
from scipy import stats

def read_csv(url,encoding='utf-8'):
    """
    時価総額低いランキングを取得します
    """
    df = pd.read_csv(url,encoding=encoding)
    return df


def get_moving_average_from_df(dataframe):
    """
    Dataframeからwindow.Rollingオブジェクトを取得し
    Series型の7日移動平均を返します
    """
    moving_average_rolling = dataframe.rolling(7)
    return moving_average_rolling.mean().dropna()

def get_slope(close_price_list,significant_digits=2):
    """
    終値のリストより1次近似を求めます
    """
    # print(close_price_list)
    close_price_list_num = len(close_price_list)
    try:
        slope,intercept = np.polyfit(list(range(close_price_list_num)), close_price_list, 1)
        # print(slope)
        slope = round_off(slope)
        intercept = round_off(intercept)
    except:
        print('1時近似を求めるのに失敗しました。スキップします。')
        return 0
    return slope,intercept

def round_off(num,significant_digits=2):
    return round(num,significant_digits)

def get_slope_dict(stock_info_df_list):
    """
    株価DataFrameリストより勾配Dictを求めます
    Return:
          dict:0番目に前半の値、1番目に後半の値
    """
    slope_dict = []
    for stock_info_df in stock_info_df_list:
        moving_average_series = get_moving_average_from_df(stock_info_df)

        moving_average_list = moving_average_series.values.tolist()

        slope = get_slope(close_price_list)
        slope = round(slope,2)
        slope_dict.append(slope)

    return {'first_half_slope':slope_dict[0],'latter_half_slope':slope_dict[1]}

def get_half_and_half_slope_dict(close_series):
    """
    終値のシリーズを受け取りそれを半分の期間に分割し
    それぞれの傾き及びその差を取得します
    args
        close_series(series): 日付が昇順になっている株価
    """

    half_num = close_series.size/2

    first_half_num = int(np.floor(half_num))
    latter_half_num = int(np.ceil(half_num))

    close_moving_average_series = get_moving_average_from_df(close_series)
    close_moving_average_list = close_moving_average_series.values.tolist()

    close_moving_average_list_first_half = close_moving_average_list[first_half_num:]
    close_moving_average_list_latter_half = close_moving_average_list[:latter_half_num]

    slope_first_half = get_slope(close_moving_average_list_first_half)
    slope_latter_half = get_slope(close_moving_average_list_latter_half)

    return {'slope_first_half':slope_first_half,
            'slope_latter_half':slope_latter_half,
            'slope_growth':slope_latter_half-slope_first_half}


def get_close_price_slope(close_price_list):
    """
    終値のリストより1次近似の傾きを計算します
    args:
       close_price_list(list): 終値のリスト
    returns:
       slope(float): 1次近似の傾き
    """
    pass

def get_coefficient_of_variation(close_series):
    """
    終値のシリーズを受け取り過去1か月と過去ヵ月の変動係数を求めます
    """
    # 過去1ヵ月の分の終値データを用意
    ## 1ヵ月分なければ0
    # 過去3ヵ月の分の終値データを用
    ## 3ヵ月分なければ0
    coefficient_of_variation = stats.variation(close_series)
    return coefficient_of_variation