"""
    В этом коде мы тестируем Финам API v1, библиотеку FinamPy.

    Автор: Олег Шпагин
    Github: https://github.com/WISEPLAT
    Telegram: https://t.me/OlegSh777
"""

exit(777)  # для запрета запуска кода, иначе выставит заявку

# перед запуском скрипта - выполните команду ниже + настройте Config.py
# git clone https://github.com/cia76/FinamPy
import time

from FinamPy.FinamPy import FinamPy
from my_config.Config import Config as ConfigAPI  # Файл конфигурации Финам API
from decimal import Decimal
from FinamPy.proto.tradeapi.v1.common_pb2 import BUY_SELL_BUY, BUY_SELL_SELL

_price = 0

def get_info_by_tickers(fp_provider, symbols):
    """Получаем информацию обо всех тикерах"""
    _ticker_info = {}
    securities = fp_provider.get_securities()  # Получаем информацию обо всех тикерах
    # print('Ответ от сервера:', securities)
    for board, symbol in symbols:  # Пробегаемся по всем тикерам
        try:
            si = next(item for item in securities.securities if item.board == board and item.code == symbol)
            # print(si)
            # print(f'\nИнформация о тикере {si.board}.{si.code} ({si.short_name}, {fp_provider.markets[si.market]}):')
            # print(f'Валюта: {si.currency}')
            decimals = si.decimals
            # print(f'Кол-во десятичных знаков: {decimals}')
            # print(f'Лот: {si.lot_size}')
            min_step = Decimal(10 ** -decimals * si.min_step).quantize(Decimal("1."+"0"*decimals))
            # print(f'Шаг цены: {min_step}')
            _ticker_info[symbol] = {"lot": si.lot_size, "nums": decimals, "step": min_step}
        except:
            print(f'\nТикер {board}.{symbol} не найден')
    return _ticker_info


def get_current_price(order_book):
    """Получаем информацию по ближайшей текущей цене тикера - из стакана"""
    global _price
    print('ask:', order_book.asks[0].price, 'bid:', order_book.bids[0].price)  # Обработчик события прихода подписки на стакан
    _price = order_book.asks[0].price


def get_free_money(fp_provider, client_id):
    """Получаем информацию сколько у нас свободных денег"""
    portfolio = fp_provider.get_portfolio(client_id)  # Получаем портфель
    # print(portfolio)
    return portfolio.money[0].balance


if __name__ == '__main__':  # Точка входа при запуске этого скрипта
    fp_provider = FinamPy(ConfigAPI.AccessToken)  # Подключаемся
    client_id = ConfigAPI.ClientIds[0]
    security_board = "TQBR"

    symbols = (('TQBR', 'SBER'), ('TQBR', 'VTBR'))

    # Получаем информацию обо всех тикерах
    _ticker_info = get_info_by_tickers(fp_provider, symbols=symbols)
    print("Информация по тикерам:", _ticker_info)

    _money = get_free_money(fp_provider, client_id)
    print('Свободные средства:', _money)

    # берем цену инструмента из стакана
    fp_provider.on_order_book = get_current_price
    fp_provider.subscribe_order_book("SBER", "TQBR", 'orderbook1')  # Подписываемся на стакан тикера
    # Выход через 1 секунду
    time.sleep(1)
    fp_provider.unsubscribe_order_book('orderbook1', 'SBER', 'TQBR')  # Отписываемся от стакана тикера
    print(_price)  # глобальная переменная )) можно сделать и по другому - для теста ))

    # выставляем заявку на покупку SBER ниже на 5% по цене кратной шагу цены
    _step = _ticker_info["SBER"]["step"]
    price = (Decimal(_price * 0.95) //_step) * _step
    print(price)

    rez = fp_provider.new_order(client_id=client_id, security_board=security_board, security_code="SBER", buy_sell=BUY_SELL_BUY, quantity=1, price=price, use_credit=True)
    print("Выставили заявку:", rez)
    print("Транзакция:", rez.transaction_id)
    tr_order = rez.transaction_id

    # снимаем эту заявку через 3 секунды
    print("Отменяем заявку (через 3 секунды):")
    time.sleep(3)
    rez = fp_provider.cancel_order(client_id=client_id, transaction_id=tr_order)
    print(rez)

    # получаем ордера
    orders = fp_provider.get_orders(client_id).orders  # Получаем заявки
    for order in orders:  # Пробегаемся по всем заявкам
        print(f'  - Заявка номер {order.order_no} {"Покупка" if order.buy_sell == "Buy" else "Продажа"} {order.security_board}.{order.security_code} {order.quantity} @ {order.price}')
    stop_orders = fp_provider.get_stops(client_id).stops  # Получаем стоп заявки
    for stop_order in stop_orders:  # Пробегаемся по всем стоп заявкам
        print(f'  - Стоп заявка номер {stop_order.stop_id} {"Покупка" if stop_order.buy_sell == "Buy" else "Продажа"} {stop_order.security_board}.{stop_order.security_code} {stop_order.stop_loss.quantity} @ {stop_order.stop_loss.price}')

    fp_provider.close_channel()  # Закрываем канал перед выходом
