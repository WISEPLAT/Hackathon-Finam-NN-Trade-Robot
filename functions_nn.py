import os
import pandas as pd
from PIL import Image, ImageDraw

cur_run_folder = os.path.abspath(os.getcwd())  # текущий каталог


def get_df_tf0(ticker, timeframe_0, period_sma_fast, period_sma_slow):
    """Считываем данные для обучения нейросети - вход - timeframe_0"""
    _filename = os.path.join(os.path.join(cur_run_folder, "csv"), f"{ticker}_{timeframe_0}.csv")
    df = pd.read_csv(_filename, sep=',')  # , index_col='datetime')
    if timeframe_0 in ["M1", "M10", "H1"]:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    df['sma_fast'] = df['close'].rolling(period_sma_fast).mean()  # формируем SMA fast
    df['sma_slow'] = df['close'].rolling(period_sma_slow).mean()  # формируем SMA slow
    df.dropna(inplace=True)  # удаляем все NULL значения
    return df.copy()


def get_df_t1(ticker, timeframe_1):
    """Считываем данные для обучения нейросети - выход - timeframe_1"""
    _filename = os.path.join(os.path.join(cur_run_folder, "csv"), f"{ticker}_{timeframe_1}.csv")
    df = pd.read_csv(_filename, sep=',')  # , index_col='datetime')
    if timeframe_1 in ["M1", "M10", "H1"]:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
    else:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')
    return df.copy()


def generate_img(_sma_fast_list, _sma_slow_list, _closes_list, draw_window):
    """Генерация картинки для обучения/теста нейросети"""
    _max = max(max(_sma_fast_list), max(_sma_slow_list), max(_closes_list))
    _min = min(min(_sma_fast_list), min(_sma_slow_list), min(_closes_list))
    _delta_h = _max - _min
    _k_h = (draw_window - 1) / _delta_h  # коэф. масштабирования по _h для помещения в квадрат
    w, h = draw_window, draw_window

    # creating new Image object - https://www.geeksforgeeks.org/python-pil-imagedraw-draw-line/
    img = Image.new("RGB", (w, h))
    img1 = ImageDraw.Draw(img)
    for i in range(1, w):
        # print(_sma_fast_list[i], _sma_slow_list[i])
        # будем использовать линии для масштабирования - а можно точки
        # выводим цену
        _h_1 = int((_closes_list[i - 1] - _min) * _k_h)
        _h = int((_closes_list[i] - _min) * _k_h)
        shape = [(i - 1, _h_1), (i, _h)]
        img1.line(shape, fill="red", width=0)
        # выводим SMA быструю
        _h_1 = int((_sma_fast_list[i - 1] - _min) * _k_h)
        _h = int((_sma_fast_list[i] - _min) * _k_h)
        shape = [(i - 1, _h_1), (i, _h)]
        img1.line(shape, fill="blue", width=0)
        # выводим SMA медленную
        _h_1 = int((_sma_slow_list[i - 1] - _min) * _k_h)
        _h = int((_sma_slow_list[i] - _min) * _k_h)
        shape = [(i - 1, _h_1), (i, _h)]
        img1.line(shape, fill="green", width=0)
    return img
