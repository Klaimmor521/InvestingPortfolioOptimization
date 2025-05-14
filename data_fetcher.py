# data_fetcher.py
import yfinance as yf
import pandas as pd
from typing import List, Optional
import logging # Используем logging, который уже был

# Настройка логирования (если еще не настроено в основном модуле, но лучше централизованно)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_historical_data(tickers: List[str],
                         start_date: str,
                         end_date: str) -> Optional[pd.DataFrame]:
    """
    Загружает исторические данные (скорректированные цены закрытия)
    для указанных тикеров с Yahoo Finance.
    """
    logging.info(f"Загрузка данных для: {tickers} с {start_date} по {end_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False) # Добавил auto_adjust и progress

        if data.empty:
            logging.error(f"Не найдены данные для тикеров {tickers} в указанный период.")
            return None

        # Если только один тикер, yfinance возвращает DataFrame, где цены в 'Close'
        # Если несколько, то MultiIndex DataFrame, где цены в ('Close', ticker_name)
        # Нам нужен DataFrame, где колонки - это тикеры, а значения - цены закрытия.

        if isinstance(data.columns, pd.MultiIndex):
            # Для нескольких тикеров извлекаем 'Close'
            close_data = data['Close'].copy()
        elif len(tickers) == 1 and 'Close' in data.columns:
            # Для одного тикера DataFrame уже содержит 'Close', просто переименуем
            close_data = data[['Close']].copy()
            close_data.rename(columns={'Close': tickers[0]}, inplace=True)
        elif len(tickers) > 1 and all(ticker in data.columns for ticker in tickers):
            # Если yfinance вернул данные не в MultiIndex, а сразу колонками тикеров
            # (такое бывает, если некоторые тикеры не загрузились, а один загрузился)
            # И если это DataFrame с ценами OHLCV, то нужно извлечь 'Close'
            # Это более сложный случай, пока упростим, предполагая, что если не MultiIndex, то уже нужный формат
            # или yf.download сам вернул только 'Close' для тех, что загрузились.
            # Важно проверить, что это действительно цены закрытия.
            # Для простоты, если не MultiIndex и не один тикер, то это уже готовые 'Close' цены
            # по тем тикерам, которые загрузились (yfinance может так вернуть)
            close_data = data.copy()
        else:
            logging.error(f"Неожиданная структура данных от yfinance для тикеров: {tickers}. Колонки: {data.columns}")
            return None

        # Удаляем строки, где все значения NaN (могут появиться, если у тикеров разная история)
        close_data.dropna(how='all', inplace=True)

        # Проверка на полностью пустые столбцы (если какой-то тикер не загрузился)
        missing_tickers = close_data.columns[close_data.isna().all()].tolist()
        if missing_tickers:
            logging.warning(f"Не удалось загрузить данные для тикеров: {missing_tickers}")
            close_data.drop(columns=missing_tickers, inplace=True)

        if close_data.empty:
            logging.error("DataFrame с ценами закрытия оказался пустым после обработки.")
            return None

        logging.info(f"Данные успешно загружены и обработаны. Финальные тикеры: {close_data.columns.tolist()}")
        return close_data

    except Exception as e:
        logging.error(f"Непредвиденная ошибка при загрузке данных: {e}", exc_info=True)
        return None