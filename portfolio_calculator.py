import yfinance as yf
import pandas as pd
from typing import List, Optional

def load_historical_data(tickers: List[str],
                         start_date: str,
                         end_date: str) -> Optional[pd.DataFrame]:
    """
    Загружает исторические данные (скорректированные цены закрытия)
    для указанных тикеров с Yahoo Finance.
    Использует auto_adjust=True (по умолчанию), поэтому 'Close' является скорректированной ценой.

    Args:
        tickers (List[str]): Список тикеров акций.
        start_date (str): Начальная дата в формате 'YYYY-MM-DD'.
        end_date (str): Конечная дата в формате 'YYYY-MM-DD'.

    Returns:
        Optional[pd.DataFrame]: DataFrame с историческими скорректированными
                                 ценами закрытия (столбец 'Close').
                                 Индекс - дата, колонки - тикеры.
                                 Возвращает None в случае ошибки загрузки
                                 или если данные не найдены.
    """
    print(f"Загрузка данных для: {tickers} с {start_date} по {end_date}...")
    try:
        # Загружаем данные с auto_adjust=True (по умолчанию)
        # В этом режиме 'Close' уже является скорректированной ценой
        data = yf.download(tickers, start=start_date, end=end_date)

        if data.empty:
            print(f"Ошибка: Не найдены данные для тикеров {tickers} в указанный период.")
            return None

        # Если только один тикер, yfinance возвращает простой DataFrame
        if len(tickers) == 1:
            # Нам нужен только столбец 'Close'
            if 'Close' in data.columns:
                # Создаем DataFrame только с нужной колонкой
                close_data = data[['Close']].copy()
                # Переименуем колонку в тикер для консистентности
                close_data.rename(columns={'Close': tickers[0]}, inplace=True)
            else:
                # Эта ошибка теперь маловероятна, но оставим проверку
                print(f"Ошибка: Не удалось найти столбец 'Close' для тикера {tickers[0]}. Структура данных: {data.columns}")
                return None
        # Если несколько тикеров, yfinance возвращает MultiIndex DataFrame
        else:
            # Выбираем 'Close' для всех тикеров
            # Проверяем, есть ли 'Close' среди названий столбцов верхнего уровня
            if 'Close' in data.columns.levels[0]:
                 close_data = data['Close'].copy()
            # Иногда, если загрузился только один из многих, структура может быть не MultiIndex
            elif isinstance(data.columns, pd.Index) and 'Close' in data.columns:
                 print("Предупреждение: Структура данных отличается от ожидаемой (возможно, загрузился только один тикер). Используется 'Close'.")
                 # Предполагаем, что названия колонок уже тикеры, выбираем 'Close' если она есть
                 if tickers[0] in data.columns and isinstance(data[tickers[0]], pd.DataFrame) and 'Close' in data[tickers[0]].columns:
                     # Очень редкий случай, структура вида Ticker -> OHLCV
                     close_data = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers if ticker in data.columns and 'Close' in data[ticker]})
                 elif 'Close' in data.columns:
                      # Если 'Close' просто столбец верхнего уровня
                      close_data = data[['Close']].copy()
                      # Попробуем переименовать, если только один столбец
                      if len(close_data.columns) == 1:
                           close_data.rename(columns={'Close': tickers[0]}, inplace=True) # Может быть неверно, если другой тикер загрузился
                 else:
                    print(f"Ошибка: Не удалось найти данные 'Close' для тикеров {tickers} в ожидаемом формате MultiIndex или простом DataFrame.")
                    return None

            else: # Fallback если структура совсем иная
                print(f"Ошибка: Не удалось найти данные 'Close' для тикеров {tickers} в ожидаемом формате MultiIndex. Структура колонок: {data.columns}")
                return None

            # Проверим, есть ли полностью пустые столбцы (если какой-то тикер не загрузился)
            # Это нужно делать после извлечения 'Close'
            missing_tickers = close_data.columns[close_data.isna().all()].tolist()
            if missing_tickers:
                print(f"Предупреждение: Не удалось загрузить данные для тикеров: {missing_tickers}")
                # Удаляем полностью пустые столбцы
                close_data.drop(columns=missing_tickers, inplace=True)
                # Если после удаления не осталось столбцов
                if close_data.empty:
                     print(f"Ошибка: Не удалось загрузить данные ни для одного из указанных тикеров после удаления пустых столбцов.")
                     return None

        # Дополнительная проверка на случай, если DataFrame оказался пустым после манипуляций
        if close_data.empty:
             print(f"Ошибка: DataFrame с ценами закрытия оказался пустым после обработки.")
             return None

        print("Данные успешно загружены!")
        return close_data

    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc() # Выводим полный traceback для диагностики
        return None

# # --- Пример использования ---
# if __name__ == "__main__":
#     # stock_tickers = ['AAPL', 'MSFT', 'GOOG'] # Несколько тикеров
#     stock_tickers = ['NVDA'] # Один тикер
#     # stock_tickers = ['FAKE', 'AAPL'] # Один неверный тикер
#     # stock_tickers = ['FAKE', 'INVALID'] # Все неверные
#     start = '2021-01-01'
#     end = '2023-12-31'

#     historical_prices = load_historical_data(stock_tickers, start, end)

#     if historical_prices is not None:
#         print("\nЗагруженные данные (первые 5 строк):")
#         print(historical_prices.head())
#         print("\nЗагруженные данные (последние 5 строк):")
#         print(historical_prices.tail())
#         print(f"\nРазмер DataFrame: {historical_prices.shape}")
#         print(f"\nКоличество пропусков (NaN) по каждому тикеру:\n{historical_prices.isnull().sum()}")
#     else:
#         print("\nНе удалось получить данные.")