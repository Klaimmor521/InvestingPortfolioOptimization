import yfinance as yf
import pandas as pd
from typing import List, Optional
import numpy as np
import logging
from scipy.optimize import minimize

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRADING_PERIODS_PER_YEAR = 252  # Пример для дневных данных

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

def  calculate_portfolio_return(daily_returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """
    Рассчитывает годовую ожидаемую доходность портфеля.

    Args:
        daily_returns_df (pd.DataFrame): DataFrame с историческими дневными доходностями активов.
                                         Колонки - активы, индексы - даты.
        weights (np.ndarray): Массив весов активов в портфеле.

    Returns:
        float: Годовая ожидаемая доходность портфеля.
    """
    if not isinstance(daily_returns_df, pd.DataFrame):
        logging.error("daily_returns_df должен быть pandas DataFrame.")
        raise TypeError("daily_returns_df должен быть pandas DataFrame.")
    if not isinstance(weights, np.ndarray):
        logging.error("weights должен быть numpy ndarray.")
        raise TypeError("weights должен быть numpy ndarray.")
    if daily_returns_df.shape[1] != len(weights):
        logging.error("Количество активов в daily_returns_df и weights должно совпадать.")
        raise ValueError("Количество активов в daily_returns_df и weights должно совпадать.")
    if not np.isclose(np.sum(weights), 1.0):
        logging.warning(f"Сумма весов ({np.sum(weights):.4f}) не равна 1.0.")
        # Можно добавить raise ValueError, если сумма весов *строго* должна быть 1.0

    try:
        # Средняя дневная доходность каждого актива
        mean_daily_returns = daily_returns_df.mean()
        # Ожидаемая дневная доходность портфеля
        portfolio_mean_daily_return = np.dot(weights, mean_daily_returns)
        # Годовая доходность
        annualized_return = portfolio_mean_daily_return * TRADING_PERIODS_PER_YEAR
        logging.info(f"Рассчитана годовая доходность: {annualized_return:.4f}")
        return annualized_return
    except Exception as e:
        logging.error(f"Ошибка при расчете годовой доходности: {e}")
        raise

def calculate_annualized_volatility(daily_returns_df: pd.DataFrame, weights: np.ndarray) -> float:
    """
    Рассчитывает годовую волатильность (стандартное отклонение) портфеля.

    Args:
        daily_returns_df (pd.DataFrame): DataFrame с историческими дневными доходностями активов.
        weights (np.ndarray): Массив весов активов в портфеле.

    Returns:
        float: Годовая волатильность портфеля.
    """
    if not isinstance(daily_returns_df, pd.DataFrame):
        logging.error("daily_returns_df должен быть pandas DataFrame.")
        raise TypeError("daily_returns_df должен быть pandas DataFrame.")
    if not isinstance(weights, np.ndarray):
        logging.error("weights должен быть numpy ndarray.")
        raise TypeError("weights должен быть numpy ndarray.")
    if daily_returns_df.shape[1] != len(weights):
        logging.error("Количество активов в daily_returns_df и weights должно совпадать.")
        raise ValueError("Количество активов в daily_returns_df и weights должно совпадать.")

    try:
        # Ковариационная матрица дневных доходностей
        cov_matrix_daily = daily_returns_df.cov()
        # Дисперсия портфеля
        portfolio_variance_daily = np.dot(weights.T, np.dot(cov_matrix_daily, weights))
        # Годовая волатильность (стандартное отклонение)
        # Дисперсию умножаем на TRADING_PERIODS_PER_YEAR, а затем берем корень
        annualized_volatility = np.sqrt(portfolio_variance_daily * TRADING_PERIODS_PER_YEAR)
        logging.info(f"Рассчитана годовая волатильность: {annualized_volatility:.4f}")
        return annualized_volatility
    except Exception as e:
        logging.error(f"Ошибка при расчете годовой волатильности: {e}")
        raise

def calculate_sharpe_ratio(annualized_return: float, annualized_volatility: float, risk_free_rate: float) -> float:
    """
    Рассчитывает коэффициент Шарпа.

    Args:
        annualized_return (float): Годовая ожидаемая доходность портфеля.
        annualized_volatility (float): Годовая волатильность портфеля.
        risk_free_rate (float): Безрисковая годовая ставка доходности.

    Returns:
        float: Коэффициент Шарпа.
    """
    if np.isclose(annualized_volatility, 0.0): 
        logging.warning("Волатильность близка к нулю. Коэффициент Шарпа не может быть рассчитан (деление на ноль). Возвращаем NaN.")
        return np.nan
    try:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        logging.info(f"Для доходности {annualized_return:.4f} и волатильности {annualized_volatility:.4f} рассчитан коэффициент Шарпа: {sharpe_ratio:.4f}")
        return sharpe_ratio
    except Exception as e:
        logging.error(f"Ошибка при расчете коэффициента Шарпа: {e}")
        raise

def get_portfolio_performance(daily_returns_df: pd.DataFrame, weights: np.ndarray, risk_free_rate: float) -> tuple[float, float, float]:
    """
    Рассчитывает годовую доходность, годовую волатильность и коэффициент Шарпа для портфеля.
    """
    logging.debug(f"Расчет производительности для весов: {weights}")
    try:
        # ИСПОЛЬЗУЕМ НОВОЕ ИМЯ ФУНКЦИИ
        ann_return = calculate_portfolio_return(daily_returns_df, weights)
        ann_volatility = calculate_annualized_volatility(daily_returns_df, weights) # Эта функция уже была для портфеля
        sharpe = calculate_sharpe_ratio(ann_return, ann_volatility, risk_free_rate)
        return ann_return, ann_volatility, sharpe
    except Exception as e:
        logging.error(f"Ошибка в get_portfolio_performance для весов {weights}: {e}")
        return np.nan, np.nan, np.nan

def portfolio_return(weights, expected_returns):
    """
    Рассчитывает ожидаемую доходность портфеля.
    Args:
        weights (np.array): Вектор весов активов.
        expected_returns (np.array): Вектор ожидаемых доходностей активов.
    Returns:
        float: Ожидаемая доходность портфеля.
    """
    return np.sum(expected_returns * weights)

def portfolio_volatility(weights, cov_matrix):
    """
    Рассчитывает волатильность (стандартное отклонение) портфеля.
    Args:
        weights (np.array): Вектор весов активов.
        cov_matrix (np.array): Ковариационная матрица доходностей активов.
    Returns:
        float: Волатильность портфеля.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def minimize_volatility_for_target_return(expected_returns, cov_matrix, target_return):
    num_assets = len(expected_returns)
    # args убираем отсюда, так как cov_matrix будет передана явно в objective
    
    # Целевая функция для минимизации (волатильность)
    def objective(weights, cov_matrix_arg): # Добавляем cov_matrix_arg
        return portfolio_volatility(weights, cov_matrix_arg) # Используем cov_matrix_arg

    # Ограничения
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights, er=expected_returns: portfolio_return(weights, er) - target_return}
    )
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    result = minimize(objective, initial_weights, args=(cov_matrix,), # Передаем cov_matrix как второй аргумент для objective
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return None

def calculate_efficient_frontier(expected_returns, cov_matrix, num_points=100):
    """
    Рассчитывает точки для построения Границы Эффективности.
    """
    results_volatility = []
    results_returns = []
    results_weights = []

    # Определяем диапазон целевых доходностей (можно настроить)
    min_ret = np.min(expected_returns)
    max_ret = np.max(expected_returns)
    target_returns_range = np.linspace(min_ret, max_ret, num_points)

    for target_ret in target_returns_range:
        optimal_weights = minimize_volatility_for_target_return(expected_returns, cov_matrix, target_ret)
        if optimal_weights is not None:
            results_weights.append(optimal_weights)
            results_returns.append(portfolio_return(optimal_weights, expected_returns))
            results_volatility.append(portfolio_volatility(optimal_weights, cov_matrix))

    return np.array(results_returns), np.array(results_volatility), np.array(results_weights)

def minimum_variance_portfolio(expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    # args убираем
    
    def objective(weights, cov_matrix_arg): # Добавляем cov_matrix_arg
        return portfolio_volatility(weights, cov_matrix_arg) # Используем cov_matrix_arg

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    result = minimize(objective, initial_weights, args=(cov_matrix,), # Передаем cov_matrix
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        mvp_return = portfolio_return(result.x, expected_returns) # expected_returns здесь доступна из замыкания
        mvp_volatility = portfolio_volatility(result.x, cov_matrix) # cov_matrix здесь доступна из замыкания
        return result.x, mvp_return, mvp_volatility
    else:
        return None, None, None

def sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    """
    Рассчитывает коэффициент Шарпа.
    """
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    if p_volatility == 0: # Избегаем деления на ноль
        return -np.inf if (p_return - risk_free_rate) < 0 else np.inf
    return (p_return - risk_free_rate) / p_volatility

def max_sharpe_ratio_portfolio(expected_returns, cov_matrix, risk_free_rate):
    num_assets = len(expected_returns)

    # Целевая функция: минимизируем -SharpeRatio
    def objective(weights, er_arg, cov_arg, rfr_arg): # Принимаем все нужные аргументы
        return -sharpe_ratio(weights, er_arg, cov_arg, rfr_arg)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = num_assets * [1. / num_assets,]

    # Передаем все необходимые аргументы для objective через args
    result = minimize(objective, initial_weights, args=(expected_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        msr_weights = result.x
        msr_return = portfolio_return(msr_weights, expected_returns)
        msr_volatility = portfolio_volatility(msr_weights, cov_matrix)
        msr_sharpe = sharpe_ratio(msr_weights, expected_returns, cov_matrix, risk_free_rate)
        return msr_weights, msr_return, msr_volatility, msr_sharpe
    else:
        return None, None, None, None

def calculate_portfolio_optimization_results(prices_df: pd.DataFrame,
                                             risk_free_rate: float = 0.02, # Ставка по умолчанию 2%
                                             num_frontier_points: int = 50): # Кол-во точек для границы
    """
    Координирует весь процесс расчета: от цен до результатов оптимизации.

    Args:
        prices_df (pd.DataFrame): DataFrame с историческими ценами закрытия.
        risk_free_rate (float): Годовая безрисковая ставка.
        num_frontier_points (int): Количество точек для расчета границы эффективности.

    Returns:
        dict: Словарь с результатами или None в случае ошибки.
              Структура словаря:
              {
                  'mvp': {'weights': np.array, 'return': float, 'volatility': float},
                  'msr': {'weights': np.array, 'return': float, 'volatility': float, 'sharpe': float},
                  'frontier': {'returns': np.array, 'volatilities': np.array, 'weights': np.array},
                  'stats': {'mean_returns': pd.Series, 'cov_matrix': pd.DataFrame} # Добавим статистику
              }
    """
    logging.info("Начало процесса оптимизации портфеля...")
    try:
        # 1. Расчет доходностей
        logging.info("Расчет периодических доходностей...")
        returns_df = calculate_periodic_returns(prices_df)
        if returns_df.empty:
            logging.error("Не удалось рассчитать доходности (DataFrame пуст).")
            return None
        logging.info(f"Доходности рассчитаны. Форма: {returns_df.shape}")

        # 2. Расчет статистики (аннуализированной)
        logging.info("Расчет аннуализированной статистики (средние доходности, ковариация)...")
        # Используем существующую функцию calculate_statistics
        expected_returns, cov_matrix = calculate_statistics(returns_df, trading_days_per_year=TRADING_PERIODS_PER_YEAR)
        if expected_returns.empty or cov_matrix.empty:
             logging.error("Не удалось рассчитать статистику.")
             return None
        logging.info("Статистика рассчитана.")
        # print("Средние годовые доходности:\n", expected_returns) # Для отладки
        # print("Годовая ковариационная матрица:\n", cov_matrix) # Для отладки


        # 3. Расчет портфеля минимальной дисперсии (MVP) - Часть MID-11
        logging.info("Расчет портфеля минимальной дисперсии (MVP)...")
        mvp_weights, mvp_return, mvp_volatility = minimum_variance_portfolio(expected_returns, cov_matrix)
        if mvp_weights is None:
            logging.warning("Не удалось рассчитать портфель минимальной дисперсии.")
            mvp_results = None
        else:
            mvp_results = {'weights': mvp_weights, 'return': mvp_return, 'volatility': mvp_volatility}
            logging.info("MVP рассчитан.")

        # 4. Расчет портфеля с максимальным коэфф. Шарпа (MSR) - Часть MID-11
        logging.info("Расчет портфеля с максимальным коэффициентом Шарпа (MSR)...")
        msr_weights, msr_return, msr_volatility, msr_sharpe = max_sharpe_ratio_portfolio(expected_returns, cov_matrix, risk_free_rate)
        if msr_weights is None:
            logging.warning("Не удалось рассчитать портфель с максимальным коэфф. Шарпа.")
            msr_results = None
        else:
            msr_results = {'weights': msr_weights, 'return': msr_return, 'volatility': msr_volatility, 'sharpe': msr_sharpe}
            logging.info("MSR рассчитан.")

        # 5. Расчет точек границы эффективности - Часть MID-12
        logging.info(f"Расчет границы эффективности ({num_frontier_points} точек)...")
        ef_returns, ef_volatilities, ef_weights = calculate_efficient_frontier(expected_returns, cov_matrix, num_points=num_frontier_points)
        if len(ef_returns) == 0:
             logging.warning("Не удалось рассчитать точки для границы эффективности.")
             frontier_results = None
        else:
             frontier_results = {'returns': ef_returns, 'volatilities': ef_volatilities, 'weights': ef_weights}
             logging.info("Граница эффективности рассчитана.")

        # 6. Сбор результатов
        final_results = {
            'mvp': mvp_results,
            'msr': msr_results,
            'frontier': frontier_results,
            'stats': {'mean_returns': expected_returns, 'cov_matrix': cov_matrix} # Возвращаем статистику, может пригодиться
        }
        logging.info("Процесс оптимизации портфеля завершен.")
        return final_results

    except Exception as e:
        logging.error(f"Критическая ошибка в процессе оптимизации портфеля: {e}", exc_info=True) # Добавим exc_info=True
        # import traceback # Можно и так, если нет logging
        # traceback.print_exc()
        return None