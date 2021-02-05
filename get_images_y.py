import sys
import os
import traceback
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))
from utils import read_csv,get_slope,round_off
import VALUES
import pandas as pd
import datetime
import cv2
import numpy as np
from sklearn.datasets import load_sample_image

ENCODING_TYPE = 'utf-8-sig'

now = datetime.datetime.now()
DEST_CSV_PATH = f'{VALUES.CSV_ROOT_PATH}/{VALUES.US_FOLDER}/{VALUES.GOT_US_STOCK_LIST_CSV_NAME}'+\
                now.strftime('%Y%m%d_%H%M%S')+'.csv'
# TODO APIで取得
usd_jpy_rate = 103.35
def get_close_price_series_(code,date_num_length=60):
    """
    終値のシリーズを取得する
    args:
        code(str):銘柄コード
        data_num_length(int):何日前のモノを取得するか(非営業日入れて)
    returns:
        close_price_series(series):終値のシリーズ
    """
    # 終値を取得する
    end = datetime.date.today()
    start = end - datetime.timedelta(days=date_num_length)
    # 終値のシリーズを取得する
    close_price_series = pdr.DataReader(code, 'yahoo',start,end)['Close']
    return close_price_series

def get_close_price_series(code,start = '2018-01-01',end = '2021-02-03'):
    """
    終値・取引高のseriesを取得する
    args:
        code(str):銘柄コーtド
        data_num_length(int):何日前のモノを取得するか
    returns:
        close_price_and_volume_df(series):終値と出来高のdf
    """
    # 終値のシリーズを取得する
    try:
        close_price_series = pdr.DataReader(code, 'yahoo',start,end)['Close']
    except Exception as ex:
        print(f'終値取得時にエラー')
        print(traceback.format_exc())
    return close_price_series

def get_trendline_list_4_quarter(close_price_series):
    """
    日付降順の終値リストに指定されている期間で株価を取得し1時近似を取得する
    args:
        n日分の日付で昇順のリスト
    """
    QUARTER_NUM = 4
    close_price_series_length = len(close_price_series)
    # ループ用変数
    close_price_series_length_quarter = int(close_price_series_length/QUARTER_NUM)
    start_index = 0
    loop_length = close_price_series_length_quarter
    trendline_list = []
    slope_list = []
    intercept_list = []
    # 平均
    close_price_avg = close_price_series.mean()
    # データが少なすぎる場合はスキップ
    if close_price_series_length < QUARTER_NUM:
        print('データが少ないためスキップします')
        return trendline_list
    # 指定期間の4等分の上昇量を取得する
    for i in range(0,close_price_series_length,close_price_series_length_quarter):
        try:
            index = i
            if i>0:
                index = i-1
            slope,intercept =\
                 get_slope(close_price_series[index:i+close_price_series_length_quarter])
            # 傾きと切片を割合で示す
            slope_list.append(round(slope*100/close_price_avg,3))
            intercept_list.append(round(intercept*100/close_price_avg,3))
            loop_length += close_price_series_length_quarter
            trendline_list.append({'slope':slope,'intercept':intercept})
        except Exception:
            print(f'1次近似取得時にエラー発生。スキップします。')
            print(traceback.format_exc())
            continue
    return slope_list,intercept_list

def create_trend_line_image(img_name,slope_list,intercept_list):
    """
    傾きと切片のリストよりイメージを作成
    """
    y = []
    for i in range(0,len(slope_list),1):
        y.append(intercept_list[i])
        y.append(intercept_list[i] + slope_list[i])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylim(70, 140)
    ax.axis("off")
    plt.plot(range(len(y)),y,color='black')
    plt.savefig(f'images/{img_name}.jpeg')
    plt.clf()
    plt.close()

def get_is_up(close_price,close_price_list):
    """
    close_price_list
    """
    return 1 if np.max(close_price_list) > close_price*1.1 else 0

if __name__ == "__main__":
    """
    """
    us_stock_list_df = read_csv(f'{VALUES.CSV_ROOT_PATH}/{VALUES.US_STOCK_LIST_CSV_NAME}')

    # 結果用DataFrame 初期化
    result_series = pd.Series(dtype='int64')
    FUTURE_NUM = 10
    PAST_NUM = 40

    for index, row in us_stock_list_df.iterrows():
        # if index <= 1215:
        #     print(index)
        #     continue
        # 今は情報技術セクター固定で渡している
        if row[VALUES.INDUSTRY] != VALUES.IT_INDUSTRY:
            continue
        print(f'{index}番目 {row[VALUES.CODE]} {row[VALUES.NAME]}')
        try:
            # 指定期間の終値シリーズの取得(日付で昇順)
            close_price_series = get_close_price_series(row[VALUES.CODE])
            # 10日分ずつずらしながらループしていく => 過去40日未来10日分を見る
            for i in range(PAST_NUM,len(close_price_series),FUTURE_NUM):
                # 基準日の日付を取得
                base_date = close_price_series.index[i].strftime('%Y%m%d')
                id = row[VALUES.CODE] + base_date
                print(id)
                # 基準日の終値を取得
                base_date_close_price = close_price_series[i]
                # 前40日分を取得
                past_40_close_prices = close_price_series[i-PAST_NUM+1:i+1]
                # 後10日分を取得
                future_10_close_prices = close_price_series[i+1:i+FUTURE_NUM+1]
                if len(future_10_close_prices) < FUTURE_NUM:
                    break
                slope_list,intercept_list = get_trendline_list_4_quarter(past_40_close_prices)
                create_trend_line_image(id,slope_list,intercept_list)
                is_up = 1 if np.max(future_10_close_prices) > base_date_close_price*1.1 else 0
                result_series[id] = is_up
        except Exception as e:
            print(f'終値取得時にエラー発生。スキップします。')
            print(traceback.format_exc())
            continue
    result_series = result_series.dropna()
    print('\n')
    result_series.to_csv('./us_stock_target.csv',encoding=ENCODING_TYPE)
    print(f'終了しましたus_stock_target.csvをご確認ください。')
