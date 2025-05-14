import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, List
import logging

import statistics_calculator

def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float: # Уточнил тип expected_returns
    """Рассчитывает ожидаемую доходность портфеля."""
    return np.sum(expected_returns.values * weights) # Используем .values для совместимости, если Series

def portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

def _sharpe_ratio_objective(weights: np.ndarray,
                            expected_returns: pd.Series,
                            cov_matrix: pd.DataFrame,
                            risk_free_rate: float) -> float:
    p_return = portfolio_return(weights, expected_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    if np.isclose(p_volatility, 0):
        return -np.inf if (p_return - risk_free_rate) >= 0 else np.inf
    return -(p_return - risk_free_rate) / p_volatility


def minimize_volatility_for_target_return(expected_returns: pd.Series,
                                          cov_matrix: pd.DataFrame,
                                          target_return: float) -> Optional[np.ndarray]:
    """Находит портфель с минимальной волатильностью для заданной целевой доходности."""
    num_assets = len(expected_returns)
    
    def objective_vol(weights_arg): # Убрал cov_matrix_arg, он доступен из замыкания
        return portfolio_volatility(weights_arg, cov_matrix)

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w, expected_returns) - target_return}
    )
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets,])

    result = minimize(objective_vol, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        logging.warning(f"Оптимизация для целевой доходности {target_return:.4f} не удалась: {result.message}")
        return None


def calculate_efficient_frontier(expected_returns: pd.Series,
                                 cov_matrix: pd.DataFrame,
                                 num_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Рассчитывает точки для построения Границы Эффективности."""
    results_volatility = []
    results_returns = []
    results_weights = []

    min_possible_ret, max_possible_ret = _get_achievable_return_range(expected_returns, cov_matrix)
    
    if min_possible_ret is None or max_possible_ret is None:
         logging.warning("Не удалось определить диапазон достижимых доходностей для границы эффективности.")
         return np.array([]), np.array([]), np.array([])

    # target_returns_range = np.linspace(expected_returns.min(), expected_returns.max(), num_points)
    target_returns_range = np.linspace(min_possible_ret, max_possible_ret, num_points)


    for target_ret in target_returns_range:
        optimal_weights = minimize_volatility_for_target_return(expected_returns, cov_matrix, target_ret)
        if optimal_weights is not None:
            # Проверяем сумму весов после оптимизации
            if not np.isclose(np.sum(optimal_weights), 1.0):
                 logging.debug(f"Сумма весов {np.sum(optimal_weights):.4f} не 1.0 для target_ret={target_ret:.4f}, пропускаем точку.")
                 continue
            if np.any(optimal_weights < -1e-5) or np.any(optimal_weights > 1.0 + 1e-5): # Небольшой допуск
                 logging.debug(f"Веса вышли за пределы [0,1] для target_ret={target_ret:.4f}, пропускаем точку. Веса: {optimal_weights}")
                 continue

            actual_return = portfolio_return(optimal_weights, expected_returns)
            actual_volatility = portfolio_volatility(optimal_weights, cov_matrix)
            
            # Дополнительная проверка, что фактическая доходность близка к целевой
            if not np.isclose(actual_return, target_ret, atol=1e-3): # Допуск на точность
                logging.debug(f"Фактическая доходность {actual_return:.4f} далека от целевой {target_ret:.4f}, пропускаем точку.")
                continue

            results_weights.append(optimal_weights)
            results_returns.append(actual_return)
            results_volatility.append(actual_volatility)
        else:
            logging.debug(f"Не удалось найти оптимальные веса для целевой доходности: {target_ret:.4f}")


    # Сортируем по волатильности для гладкого графика
    if results_volatility:
        sorted_indices = np.argsort(results_volatility)
        results_returns = np.array(results_returns)[sorted_indices]
        results_volatility = np.array(results_volatility)[sorted_indices]
        results_weights = np.array(results_weights)[sorted_indices]

    return results_returns, results_volatility, results_weights

def _get_achievable_return_range(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Вспомогательная функция для определения реалистичного диапазона доходностей."""
    mvp_w, mvp_r, _ = minimum_variance_portfolio(expected_returns, cov_matrix)
    if mvp_w is None:
        return None, None # Не удалось найти MVP

    # Максимальная доходность - это доходность актива с максимальной ожидаемой доходностью
    # (при условии, что можно вложить 100% в один актив)
    max_ind_return = expected_returns.max()
    
    # Теоретически, граница эффективности может идти выше max_ind_return
    # за счет кредитного плеча или коротких продаж, но мы их не используем.
    # Попробуем найти портфель для доходности чуть выше максимальной индивидуальной,
    # чтобы убедиться, что оптимизатор может ее достичь.
    # Однако, для простоты, можно ограничиться диапазоном от MVP до max(expected_returns)
    
    # Если MVP дает доходность выше, чем максимальная индивидуальная (маловероятно, но возможно при сложной ковариации)
    # то берем от MVP до доходности, которую дает 100% вложение в самый доходный актив
    
    # Диапазон должен быть от доходности MVP до доходности самого доходного актива
    # или немного выше, если оптимизатор может это сделать без коротких продаж
    
    # Более безопасный подход:
    # Найти портфель для самой низкой индивидуальной доходности и самой высокой
    # min_ret_port_w = minimize_volatility_for_target_return(expected_returns, cov_matrix, expected_returns.min())
    # max_ret_port_w = minimize_volatility_for_target_return(expected_returns, cov_matrix, expected_returns.max())

    # if min_ret_port_w is not None and max_ret_port_w is not None:
    #     min_achievable = portfolio_return(min_ret_port_w, expected_returns)
    #     max_achievable = portfolio_return(max_ret_port_w, expected_returns)
    #     return min(min_achievable, mvp_r), max(max_achievable, mvp_r) # Берем диапазон от MVP до более широких границ

    return mvp_r, max_ind_return # Упрощенный диапазон, который должен работать

def minimum_variance_portfolio(expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """Находит Портфель Минимальной Дисперсии (MVP)."""
    num_assets = len(expected_returns)
    
    def objective_vol(weights_arg):
        return portfolio_volatility(weights_arg, cov_matrix)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets,])

    result = minimize(objective_vol, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        mvp_weights = result.x
        mvp_return = portfolio_return(mvp_weights, expected_returns)
        mvp_volatility = portfolio_volatility(mvp_weights, cov_matrix)
        return mvp_weights, mvp_return, mvp_volatility
    else:
        logging.error(f"Оптимизация для MVP не удалась: {result.message}")
        return None, None, None


def max_sharpe_ratio_portfolio(expected_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[float]]:
    """Находит портфель с максимальным коэффициентом Шарпа."""
    num_assets = len(expected_returns)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = np.array(num_assets * [1. / num_assets,])

    result = minimize(_sharpe_ratio_objective, initial_weights,
                      args=(expected_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        msr_weights = result.x
        msr_return = portfolio_return(msr_weights, expected_returns)
        msr_volatility = portfolio_volatility(msr_weights, cov_matrix)
        msr_sharpe = (msr_return - risk_free_rate) / msr_volatility if not np.isclose(msr_volatility, 0) else np.nan
        return msr_weights, msr_return, msr_volatility, msr_sharpe
    else:
        logging.error(f"Оптимизация для Max Sharpe Ratio не удалась: {result.message}")
        return None, None, None, None

# --- Главная координирующая функция (изменена для вызова из statistics_calculator) ---
def calculate_portfolio_optimization_results(prices_df: pd.DataFrame,
                                             risk_free_rate: float = 0.02,
                                             num_frontier_points: int = 50) -> Optional[Dict]:
    logging.info("Начало процесса оптимизации портфеля...")
    try:
        logging.info("Расчет периодических доходностей...")
        returns_df = statistics_calculator.calculate_periodic_returns(prices_df) # ИЗМЕНЕНО

        if returns_df.empty:
            logging.error("Не удалось рассчитать доходности (DataFrame пуст).")
            return None
        logging.info(f"Доходности рассчитаны. Форма: {returns_df.shape}")

        logging.info("Расчет аннуализированной статистики...")
        expected_returns, cov_matrix = statistics_calculator.calculate_annualized_statistics(returns_df) # ИЗМЕНЕНО

        if expected_returns.empty or cov_matrix.empty:
             logging.error("Не удалось рассчитать статистику.")
             return None
        logging.info("Статистика рассчитана.")

        # ... (остальная часть функции с расчетом MVP, MSR, Frontier - БЕЗ ИЗМЕНЕНИЙ) ...
        logging.info("Расчет портфеля минимальной дисперсии (MVP)...")
        mvp_weights, mvp_return, mvp_volatility = minimum_variance_portfolio(expected_returns, cov_matrix)
        mvp_results = {'weights': mvp_weights, 'return': mvp_return, 'volatility': mvp_volatility} if mvp_weights is not None else None
        if mvp_results: logging.info("MVP рассчитан.")
        else: logging.warning("Не удалось рассчитать MVP.")

        logging.info("Расчет портфеля с максимальным коэффициентом Шарпа (MSR)...")
        msr_weights, msr_return, msr_volatility, msr_sharpe = max_sharpe_ratio_portfolio(expected_returns, cov_matrix, risk_free_rate)
        msr_results = {'weights': msr_weights, 'return': msr_return, 'volatility': msr_volatility, 'sharpe': msr_sharpe} if msr_weights is not None else None
        if msr_results: logging.info("MSR рассчитан.")
        else: logging.warning("Не удалось рассчитать MSR.")

        logging.info(f"Расчет границы эффективности ({num_frontier_points} точек)...")
        ef_returns, ef_volatilities, ef_weights = calculate_efficient_frontier(expected_returns, cov_matrix, num_points=num_frontier_points)
        frontier_results = {'returns': ef_returns, 'volatilities': ef_volatilities, 'weights': ef_weights} if len(ef_returns) > 0 else None
        if frontier_results: logging.info("Граница эффективности рассчитана.")
        else: logging.warning("Не удалось рассчитать границу эффективности.")

        final_results = {
            'mvp': mvp_results,
            'msr': msr_results,
            'frontier': frontier_results,
            'stats': {'mean_returns': expected_returns, 'cov_matrix': cov_matrix}
        }
        logging.info("Процесс оптимизации портфеля завершен.")
        return final_results

    except Exception as e:
        logging.error(f"Критическая ошибка в процессе оптимизации портфеля: {e}", exc_info=True)
        return None