# statistics_calculator.py
import pandas as pd
import numpy as np
from typing import Tuple # Добавляем Tuple
import logging

# Настройка логирования
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRADING_PERIODS_PER_YEAR = 252

def calculate_periodic_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Рассчитывает периодические (например, дневные) доходности для активов.
    """
    if not isinstance(prices_df, pd.DataFrame):
        logging.error("Входные данные цен должны быть pandas DataFrame.")
        raise TypeError("Входные данные цен должны быть pandas DataFrame.")
    if prices_df.empty:
        logging.warning("Входной DataFrame цен пуст, возвращается пустой DataFrame доходностей.")
        return pd.DataFrame()

    returns_df = prices_df.pct_change()
    returns_df = returns_df.dropna(how='all')
    logging.info(f"Периодические доходности рассчитаны. Форма: {returns_df.shape}")
    return returns_df

def calculate_annualized_statistics(returns_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Рассчитывает аннуализированные средние доходности и ковариационную матрицу.
    """
    if not isinstance(returns_df, pd.DataFrame):
        logging.error("Входные данные доходностей должны быть pandas DataFrame.")
        raise TypeError("Входные данные доходностей должны быть pandas DataFrame.")
    if returns_df.empty:
        logging.warning("DataFrame доходностей пуст, возвращаются пустые структуры статистики.")
        return pd.Series(dtype='float64'), pd.DataFrame()

    mean_periodic_returns = returns_df.mean()
    mean_annual_returns = mean_periodic_returns * TRADING_PERIODS_PER_YEAR

    cov_periodic_matrix = returns_df.cov()
    cov_annual_matrix = cov_periodic_matrix * TRADING_PERIODS_PER_YEAR
    logging.info("Аннуализированная статистика (средние доходности, ковариация) рассчитана.")
    return mean_annual_returns, cov_annual_matrix