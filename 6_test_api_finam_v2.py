"""
    В этом коде мы тестируем Финам API v2, библиотеку finam_trade_api.

    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

exit(777)  # для запрета запуска кода, иначе выставит заявку

import time
from decimal import Decimal
from my_config.Config import Config
from finam_trade_api.client import Client
from finam_trade_api.portfolio.model import PortfolioRequestModel
from finam_trade_api.order.model import (
    BoardType,
    CreateOrderRequestModel,
    CreateStopOrderRequestModel,
    DelOrderModel,
    OrdersRequestModel,
    OrderType,
    PropertyType,
    StopLossModel,
    StopQuantity,
    StopQuantityUnits,
    TakeProfitModel
)

token = Config.AccessToken
client_id = Config.ClientIds[0]
client = Client(token)


async def get_all_data():
    return await client.securities.get_data()


async def get_data_by_code(code: str):
    return await client.securities.get_data(code)


async def get_data_by_codes(board: str, tickers: list):
    """Получаем информацию обо всех тикерах"""
    _ticker_info = {}
    _all_info = await client.securities.get_data()
    for security in _all_info.data.securities:
        # print(security)
        if security.code in tickers and security.board == board:
            # print(security)
            decimals = int(security.decimals)
            min_step = security.minStep
            min_step = Decimal(10 ** -decimals * min_step).quantize(Decimal("1." + "0" * decimals))
            _ticker_info[security.code] = {"lot": int(security.lotSize), "nums": decimals, "step": min_step}
    return _ticker_info


async def get_portfolio():
    """Получаем информацию сколько у нас свободных денег"""
    params = PortfolioRequestModel(clientId=client_id)
    _portfolio_info = await client.portfolio.get_portfolio(params)
    _equity = _portfolio_info.data.equity
    _money_in_pos = 0
    for position in _portfolio_info.data.positions:
        _money_in_pos += position.equity
    return _equity-_money_in_pos


async def create_order(ticker: str, buy_sell: OrderType, quantity: int, price: float, use_credit: bool = False):
    """Функция выставления ордера"""
    payload = CreateOrderRequestModel(
        clientId=client_id,
        securityBoard=BoardType.TQBR,
        securityCode=ticker,
        buySell=buy_sell,
        quantity=quantity,
        price=price,
        useCredit=use_credit,
        property=PropertyType.PutInQueue,
        condition=None,
        validateBefore=None,
    )
    return await client.orders.create_order(payload)


async def del_order(transaction_id: str):
    """Функция снятия ордера"""
    params = DelOrderModel(clientId=client_id, transactionId=transaction_id)
    return await client.orders.del_order(params)


async def get_orders():
    """Функция получения ордеров"""
    params = OrdersRequestModel(
        clientId=client_id,
        includeActive="true",
        includeMatched="true",
    )
    return await client.orders.get_orders(params)


if __name__ == '__main__':
    import asyncio

    # print(asyncio.run(get_all_data()))
    # code_ = "SBER"
    # print(asyncio.run(get_data_by_code(code_)))

    # Получаем информацию обо всех тикерах
    tickers = ['SBER', 'VTBR']
    _ticker_info = asyncio.run(get_data_by_codes(board="TQBR", tickers=tickers))
    print("Информация по тикерам:", _ticker_info)

    _money = asyncio.run(get_portfolio())
    print('Свободные средства:', _money)

    # выставляем заявку на покупку SBER
    _order = asyncio.run(create_order(ticker="SBER", buy_sell=OrderType.Buy, quantity=1, price=230.05, use_credit=True))
    print(_order)
    print(_order.data.transactionId)
    tr_order = _order.data.transactionId

    # снимаем эту заявку через 3 секунды
    print("Отменяем заявку (через 3 секунды):")
    time.sleep(3)
    rez = asyncio.run(del_order(transaction_id=tr_order))
    print(rez)

    # получаем ордера
    print(asyncio.run(get_orders()))
