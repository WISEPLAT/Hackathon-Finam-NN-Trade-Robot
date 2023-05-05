"""
    В этом коде реализована live стратегия, мы однократно асинхронно
    получаем исторические данные с MOEX, и считаем что следующие асинхронно
    полученные исторические данные приходят нам в live режиме.
    Т.к. получаем их бесплатно, то есть задержка в полученных данных на 15 минут.

    Используем нейросеть для прогноза о вхождении в сделку:
    - нейросеть выбираем на шаге №3, по результатам loss, accuracy, val_loss, val_accuracy
    - открываем позицию по рынку, как только получаем сигнал от нейросети с классом 1 - на покупку 1 лотом
    - без стоп-лосса, т.к. закрывать позицию будем на следующем +1 баре старшего таймфрейма по рынку

    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

import asyncio
import os.path

import aiohttp
import aiomoex
import logging
import functions
import functions_nn
import pandas as pd
import numpy as np
from aiohttp import ClientSession
from datetime import datetime, timedelta
from typing import List, Optional

from FinamPy import FinamPy  # Коннект к Финам API - для выставления заявок на покупку/продажу
from FinamPy.proto.tradeapi.v1.common_pb2 import BUY_SELL_BUY, BUY_SELL_SELL

from my_config.Config import Config as ConfigAPI  # Файл конфигурации Финам API
from my_config.trade_config import Config  # Файл конфигурации торгового робота

from keras.models import load_model
from keras.utils.image_utils import img_to_array


logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class HackathonFinamStrategy:
    """Этот класс реализует нашу торговую стратегию."""

    def __init__(
        self,
        ticker: str,
        timeframe: str,
        days_back: int,
        check_interval: int,
        session: Optional[ClientSession],
        trading_hours_start: str,
        trading_hours_end: str,
        security_board: str,
        client_id: str,
    ):
        self.account_id = None
        self.ticker = ticker
        self.timeframe = timeframe
        self.days_back = days_back
        self.check_interval = check_interval
        self.session = session
        self.candles: List[List] = []
        self.live_mode = False
        self.trading_hours_start = trading_hours_start
        self.trading_hours_end = trading_hours_end
        self.security_board = security_board
        self.client_id = client_id
        self.order_time = None
        self.in_position = False

    async def get_all_candles(self, start, end):
        """Функция получения свечей с MOEX."""
        tf = functions.get_timeframe_moex(self.timeframe)
        data = await aiomoex.get_market_candles(self.session, self.ticker, interval=tf, start=start, end=end)  # M10
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['begin'], format='%Y-%m-%d %H:%M:%S')
        # для M1, M10, H1 - приводим дату свечи в правильный вид
        if tf in [1, 10, 60]:
            df['datetime'] = df['datetime'].apply(lambda x: x + timedelta(minutes=tf))
        df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
        # кроме последней свечи - т.к. у нее в рынке меняется объем
        df = df[:-1]
        return df.values.tolist()

    async def get_historical_data(self, model, fp_provider):
        """Получение исторических данных по тикеру."""
        logger.debug("Получение исторических данных по тикеру: %s", self.ticker)
        start = (datetime.now().date()-timedelta(days=self.days_back)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # print(start, end)
        _candles = await self.get_all_candles(start=start, end=end)
        # print(_candles)
        for candle in _candles:
            if candle not in self.candles:
                self.candles.append(candle)
                logger.debug("- найдено %s - тикер: %s - live: %s", candle, self.ticker, self.live_mode)

                # если уже live режим
                if self.live_mode:
                    await self.live_check_can_we_open_position(model, fp_provider)

    async def live_check_can_we_open_position(self, model, fp_provider):
        """В live проверяем, можем ли мы открыть позицию,
        если нейросеть на текущих данных выдаст класс 1"""
        # создаем текущую картинку для отправки в нейросеть

        df = pd.DataFrame(self.candles, columns=["datetime", "open", "high", "low", "close", "volume"])
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        # print(df)

        # не меняем
        period_sma_slow = 64
        period_sma_fast = 16
        draw_window = 128  # окно данных
        steps_skip = 16  # шаг сдвига окна данных
        draw_size = 128  # размер стороны квадратной картинки

        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        df['sma_fast'] = df['close'].rolling(period_sma_fast).mean()  # формируем SMA fast
        df['sma_slow'] = df['close'].rolling(period_sma_slow).mean()  # формируем SMA slow

        df.dropna(inplace=True)  # удаляем все NULL значения

        # print(df)

        df_in = df.copy()
        _close_in = df_in["close"].tolist()
        sma_fast = df_in["sma_fast"].tolist()
        sma_slow = df_in["sma_slow"].tolist()
        j = len(_close_in)

        _sma_fast_list = sma_fast[j - draw_window:j]
        _sma_slow_list = sma_slow[j - draw_window:j]
        _closes_list = _close_in[j - draw_window:j]

        # генерация картинки для обучения/теста нейросети
        img = functions_nn.generate_img(_sma_fast_list, _sma_slow_list, _closes_list, draw_window)
        # img.show()  # показать сгенерированную картинку

        # отправляем созданную картинку в нейросеть
        # model.summary()
        img_array = img_to_array(img)  # https://www.tensorflow.org/api_docs/python/tf/keras/utils/img_to_array
        # print(img_array.shape)
        img_array = np.expand_dims(img_array, axis=0)
        # print(img_array.shape)
        # print("Predicted: ", model.predict(img_array))
        _predict = model.predict(img_array, verbose=0)
        _class = 0
        if _predict[0][1] >= 0: _class = 1
        print("Predicted: ", model.predict(img_array), " class = ", _class)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # теперь реализуем простую торговую логику
        if not self.in_position:  # если нет открытой позиции
            if _class == 1:
                # покупаем, если еще не купили, когда нейросеть предсказывает рост
                rez = fp_provider.new_order(client_id=self.client_id, security_board=self.security_board,
                                            security_code=self.ticker,
                                            buy_sell=BUY_SELL_BUY, quantity=1,
                                            use_credit=True,
                                            )  # price не указываем, чтобы купить по рынку

                self.order_time = datetime.now()
                self.in_position = True
                print(f"Выставили заявку на покупку 1 лота {self.ticker}:", rez)
                print("\t - транзакция:", rez.transaction_id)
                print("\t - время:", self.order_time)
        else:  # если есть открытая позиция - проверка, может пора её закрыть
            _now = datetime.now()
            _timeframe_1 = Config.timeframe_1  # старший таймфрейм
            _delta = functions.get_timeframe_moex(tf=_timeframe_1)
            print(_now, _timeframe_1, _delta, self.order_time + timedelta(minutes=_delta), _now >= self.order_time + timedelta(minutes=_delta))
            if _now >= self.order_time + timedelta(minutes=_delta):
                # продаем, если прошло достаточно времени +1 бар старшего таймфрейма
                rez = fp_provider.new_order(client_id=self.client_id, security_board=self.security_board,
                                            security_code=self.ticker,
                                            buy_sell=BUY_SELL_SELL, quantity=1,
                                            use_credit=True,
                                            )  # price не указываем, чтобы купить по рынку
                self.order_time = None
                self.in_position = False
                print(f"Выставили заявку на продажу 1 лота {self.ticker}:", rez)
                print("\t - транзакция:", rez.transaction_id)
                print("\t - время:", self.order_time)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    async def ensure_market_open(self):
        """Ждем открытия рынка. Без учета выходных и праздников."""
        is_trading_hours = False
        while not is_trading_hours:
            logger.debug("Ждем открытия рынка. ticker=%s", self.ticker)
            now = datetime.now()
            now_start = datetime.fromisoformat(now.strftime("%Y-%m-%d") + " " + self.trading_hours_start)
            now_end = datetime.fromisoformat(now.strftime("%Y-%m-%d") + " " + self.trading_hours_end)
            if now_start <= now <= now_end: is_trading_hours = True
            if not is_trading_hours: await asyncio.sleep(60)

    async def main_cycle(self, model, fp_provider):
        """Основной цикл live стратегии."""
        while True:
            try:
                # await self.ensure_market_open()  # проверяем, что рынок открыт
                await self.get_historical_data(model, fp_provider)  # получаем исторические данные с MOEX

                # после первого получения исторических данных включаем live режим
                if not self.live_mode: self.live_mode = True

                # код стратегии для live
                # сигналы на покупку или продажу обрабатываются в live_check_can_we_open_position

                logger.debug("- live режим: запуск кода стратегии для покупки/продажи - тикер: %s", self.ticker)

            except Exception as are:
                logger.error("Client error %s", are)

            await asyncio.sleep(self.check_interval)

    async def start(self):
        """Запуск стратегии начинается с этой функции."""
        # здесь можно сделать некоторые инициализации переменных
        fp_provider = FinamPy(ConfigAPI.AccessToken)  # Коннект к Финам API - для выставления заявок на покупку/продажу
        model = load_model(os.path.join("NN_winner", "cnn_Open.hdf5"))  # загружаем обученную модель нейросети
        # # Check its architecture
        # model.summary()
        await self.main_cycle(model=model, fp_provider=fp_provider)  # запуск основного цикла стратегии


async def run_strategy(portfolio, timeframe, days_back, check_interval, trading_hours_start, trading_hours_end, security_board, client_id):
    """Запускаем асинхронно стратегию для каждого тикера из портфеля."""
    async with aiohttp.ClientSession() as session:
        strategy_tasks = []
        for instrument in portfolio:
            strategy = HackathonFinamStrategy(  # формируем стратегию для тикера
                ticker=instrument,
                timeframe=timeframe,
                days_back=days_back,
                check_interval=check_interval,
                session=session,
                trading_hours_start=trading_hours_start,
                trading_hours_end=trading_hours_end,
                security_board=security_board,
                client_id=client_id,
            )
            strategy_tasks.append(asyncio.create_task(strategy.start()))
        await asyncio.gather(*strategy_tasks)


if __name__ == "__main__":

    # применение настроек из config.py
    portfolio = Config.portfolio  # тикеры по которым торгуем
    security_board = Config.security_board  # класс тикеров
    timeframe_0 = Config.timeframe_0  # таймфрейм на котором торгуем == таймфрейму на котором обучали нейросеть
    client_id = ConfigAPI.ClientIds[0]  # id клиента

    trading_hours_start = Config.trading_hours_start  # время работы биржи - начало
    trading_hours_end = Config.trading_hours_end  # время работы биржи - конец

    # создаем необходимые каталоги
    functions.create_some_folders(timeframes=[timeframe_0])

    days_back = 1  # на сколько дней назад берем данные
    check_interval = 10  # интервал проверки в секундах на появление новой завершенной свечи

    # запуск асинхронного цикла получения исторических данных
    loop = asyncio.get_event_loop()  # создаем цикл
    task = loop.create_task(  # в цикл добавляем 1 задачу
        run_strategy(  # запуск стратегии
            portfolio=portfolio,
            timeframe=timeframe_0,  # на каком таймфрейме торгуем
            days_back=days_back,
            check_interval=check_interval,
            trading_hours_start=trading_hours_start,
            trading_hours_end=trading_hours_end,
            security_board=security_board,
            client_id=client_id,
        )
    )
    loop.run_until_complete(task)  # ждем окончания выполнения цикла
