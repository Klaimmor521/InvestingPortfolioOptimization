import yfinance as yf
import pandas as pd
from typing import List, Optional
import numpy as np

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
    print("\nНе удалось получить данные.")

def calculate_periodic_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает периодические (например, дневные) доходности для активов.

    Args:
        prices_df (pd.DataFrame): DataFrame с историческими ценами закрытия.
                                  Индексы - даты, колонки - тикеры.

    Returns:
        pd.DataFrame: DataFrame с рассчитанными периодическими доходностями.
                      Первая строка с NaN удалена.
    """
    if not isinstance(prices_df, pd.DataFrame):
        raise TypeError("Входные данные должны быть pandas DataFrame.")
    if prices_df.empty:
        # Возвращаем пустой DataFrame, если на входе пустой
        return pd.DataFrame()
    if prices_df.isnull().values.any():
        # Это более сложный случай, если в ценах есть пропуски.
        # Для простоты пока можно или бросить ошибку, или просто продолжить,
        # .pct_change() обработает NaN внутри столбца, но это может повлиять на результаты.
        # print("Предупреждение: Во входных данных цен обнаружены NaN значения.")
        pass # .pct_change() сам вернет NaN где нужно

    # Рассчитываем процентное изменение цен
    # formula: (price_t / price_t-1) - 1
    returns_df = prices_df.pct_change()

    # Удаляем первую строку, так как для нее нет предыдущего значения для расчета доходности
    # (она будет содержать NaN)
    returns_df = returns_df.dropna(how='all') # Удаляем строки, где ВСЕ значения NaN (это первая строка)

    return returns_df

# # Пример использования (для тестирования этой функции):
# if __name__ == '__main__':
#     # Создадим тестовый DataFrame с ценами
#     data = {
#         'AAPL': [150, 152, 151, 155, 154],
#         'MSFT': [300, 303, 302, 305, 306]
#     }
#     dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
#     prices_test_df = pd.DataFrame(data, index=dates)

#     print("Исходный DataFrame цен:")
#     print(prices_test_df)
#     print("-" * 30)

#     # Проверим случай с пустым DataFrame
#     empty_df = pd.DataFrame()
#     print("\nТест с пустым DataFrame:")
#     returns_from_empty = calculate_periodic_returns(empty_df)
#     print(returns_from_empty)
#     print("-" * 30)

#     # Проверим основной функционал
#     try:
#         returns_test_df = calculate_periodic_returns(prices_test_df.copy()) # .copy() чтобы не менять исходный
#         print("\nРассчитанный DataFrame доходностей:")
#         print(returns_test_df)
#         print("-" * 30)

#         # Проверим, что первая дата исходного DataFrame отсутствует в доходностях
#         if not returns_test_df.empty:
#             assert prices_test_df.index[0] not in returns_test_df.index
#             print(f"Первая дата ({prices_test_df.index[0].date()}) из цен отсутствует в доходностях: OK")

#         # Проверим расчет для одного значения вручную
#         # AAPL: (152 - 150) / 150 = 0.013333...
#         expected_aapl_return_day2 = (152 - 150) / 150
#         if not returns_test_df.empty: # Добавим проверку, что DataFrame не пустой
#             actual_aapl_return_day2 = returns_test_df.loc[dates[1], 'AAPL'] # Доходность для второй даты
#             assert np.isclose(actual_aapl_return_day2, expected_aapl_return_day2), \
#                 f"Ошибка в расчете! Ожидалось: {expected_aapl_return_day2}, Получено: {actual_aapl_return_day2}"
#             print(f"Ручная проверка значения для AAPL на {dates[1].date()}: OK")

#     except Exception as e:
#         print(f"Произошла ошибка: {e}")

#     # Тест с NaN внутри данных
#     data_with_nan = {
#         'GOOG': [2000, np.nan, 2010, 2005, 2020],
#         'AMZN': [3000, 3010, 3005, np.nan, 3030]
#     }
#     prices_nan_df = pd.DataFrame(data_with_nan, index=dates)
#     print("\nТест с DataFrame с NaN внутри:")
#     print(prices_nan_df)
#     returns_nan_df = calculate_periodic_returns(prices_nan_df.copy())
#     print("\nРассчитанный DataFrame доходностей с NaN:")
#     print(returns_nan_df)
#     # .pct_change() сам обрабатывает NaN: результат до и после NaN будет NaN для этого периода.

def calculate_statistics(returns_df: pd.DataFrame, trading_days_per_year: int = 252):
    """
    Рассчитывает основные статистические показатели для моделирования портфеля:
    средние (ожидаемые) доходности и ковариационную матрицу доходностей.
    Также выполняет аннуализацию показателей.
    Args:
        returns_df (pd.DataFrame): DataFrame с периодическими доходностями активов.
                                   Индексы - даты, колонки - тикеры.
        trading_days_per_year (int): Количество торговых дней в году для аннуализации.
                                     По умолчанию 252.
    Returns:
        tuple: Кортеж, содержащий:
            - mean_returns (pd.Series): Серия аннуализированных средних доходностей для каждого актива.
            - cov_matrix (pd.DataFrame): Аннуализированная ковариационная матрица доходностей.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("Входные данные доходностей должны быть pandas DataFrame.")
    if returns_df.empty:
        # Если DataFrame доходностей пуст, возвращаем пустые структуры
        return pd.Series(dtype='float64'), pd.DataFrame()
    if returns_df.isnull().values.any():
        # В реальном проекте здесь нужна более тщательная обработка NAN.
        print("Предупреждение: Обнаружены NaN в DataFrame доходностей. Результаты могут быть неточными.")
        # Можно добавить .dropna() еще раз, чтобы попытаться убрать строки/столбцы с NaN,
        # но это может удалить полезные данные, если NaN не систематические.
        # returns_df = returns_df.dropna() # Быть осторожным!

    # Рассчитываем средние периодические доходности
    # .mean() для DataFrame по умолчанию считает среднее по столбцам (axis=0)
    mean_periodic_returns = returns_df.mean()

    # Аннуализируем средние доходности
    # Умножаем на количество торговых периодов в году
    # (если доходности дневные, то на кол-во торг. дней)
    mean_annual_returns = mean_periodic_returns * trading_days_per_year

    # Рассчитываем ковариационную матрицу периодических доходностей
    # .cov() для DataFrame рассчитывает попарные ковариации между всеми столбцами
    cov_periodic_matrix = returns_df.cov()

    # Аннуализируем ковариационную матрицу
    # Умножаем на количество торговых периодов в году
    cov_annual_matrix = cov_periodic_matrix * trading_days_per_year

    return mean_annual_returns, cov_annual_matrix

# # Пример использования (для тестирования этой функции):
# if __name__ == '__main__':
#     # --- Код для тестирования calculate_periodic_returns остался выше ---
#     # ... (скопируй сюда тестовый блок из предыдущего ответа, если нужно)
#     # Создадим тестовый DataFrame с ценами для нового теста
#     data_prices = {
#         'STOCK_A': [100, 101, 102, 101, 103, 105, 104],
#         'STOCK_B': [200, 200, 201, 203, 202, 205, 206]
#     }
#     dates_prices = pd.to_datetime([
#         '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
#         '2023-01-05', '2023-01-06', '2023-01-07'
#     ])
#     prices_df_for_stats = pd.DataFrame(data_prices, index=dates_prices)

#     print("\n" + "="*50)
#     print("Тестирование функции calculate_statistics:")
#     print("="*50)
#     print("Исходный DataFrame цен для статистики:")
#     print(prices_df_for_stats)

#     # Сначала получим доходности
#     returns_df_for_stats = calculate_periodic_returns(prices_df_for_stats.copy())
#     print("\nРассчитанный DataFrame периодических доходностей:")
#     print(returns_df_for_stats)

#     if returns_df_for_stats.empty:
#         print("\nDataFrame доходностей пуст, статистика не может быть рассчитана.")
#     else:
#         # Рассчитаем статистику
#         try:
#             mean_returns, cov_matrix = calculate_statistics(returns_df_for_stats)

#             print("\nАннуализированные средние доходности (pd.Series):")
#             print(mean_returns)
#             print(f"Тип: {type(mean_returns)}")

#             print("\nАннуализированная ковариационная матрица (pd.DataFrame):")
#             print(cov_matrix)
#             print(f"Тип: {type(cov_matrix)}")

#             # Проверим размерности
#             assert len(mean_returns) == len(returns_df_for_stats.columns)
#             assert cov_matrix.shape == (len(returns_df_for_stats.columns), len(returns_df_for_stats.columns))
#             print("\nПроверка размерностей: OK")

#             # Дисперсия отдельного актива - это диагональный элемент ковариационной матрицы
#             variance_stock_a_annual = cov_matrix.loc['STOCK_A', 'STOCK_A']
#             # Стандартное отклонение - корень из дисперсии
#             std_dev_stock_a_annual = np.sqrt(variance_stock_a_annual)
#             print(f"\nПример: Аннуализированная Дисперсия STOCK_A: {variance_stock_a_annual:.6f}")
#             print(f"Пример: Аннуализированное Станд.Откл. (Волатильность) STOCK_A: {std_dev_stock_a_annual:.6f}")

#         except Exception as e:
#             print(f"Произошла ошибка при расчете статистики: {e}")

#     # Тест с пустым DataFrame доходностей
#     print("\nТест calculate_statistics с пустым DataFrame доходностей:")
#     empty_returns_df = pd.DataFrame()
#     mean_empty, cov_empty = calculate_statistics(empty_returns_df)
#     print("Средние доходности:", mean_empty)
#     print("Ковариационная матрица:", cov_empty)
#     assert mean_empty.empty
#     assert cov_empty.empty
#     print("Обработка пустого DataFrame доходностей: OK")




