import unittest
import numpy as np
import pandas as pd
from portfolio_calculator import (
    calculate_portfolio_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    get_portfolio_performance,
    TRADING_PERIODS_PER_YEAR
)

class TestPortfolioCalculations(unittest.TestCase):

    def setUp(self):
        """Настройка тестовых данных перед каждым тестом."""
        np.random.seed(0) # Для консистентности тестов
        # Искусственные дневные доходности для 2 активов
        data_asset1 = np.array([0.01, -0.005, 0.015, 0.002, -0.008] * (TRADING_PERIODS_PER_YEAR // 5))
        data_asset2 = np.array([-0.002, 0.01, -0.003, 0.005, 0.001] * (TRADING_PERIODS_PER_YEAR // 5))

        # Убедимся, что длина данных равна TRADING_PERIODS_PER_YEAR
        if len(data_asset1) < TRADING_PERIODS_PER_YEAR:
            data_asset1 = np.pad(data_asset1, (0, TRADING_PERIODS_PER_YEAR - len(data_asset1)), 'edge')
            data_asset2 = np.pad(data_asset2, (0, TRADING_PERIODS_PER_YEAR - len(data_asset2)), 'edge')


        self.returns_df = pd.DataFrame({
            'AssetA': data_asset1[:TRADING_PERIODS_PER_YEAR],
            'AssetB': data_asset2[:TRADING_PERIODS_PER_YEAR]
        })
        self.weights_valid = np.array([0.6, 0.4])
        self.weights_invalid_sum = np.array([0.5, 0.4]) # Сумма не 1
        self.weights_invalid_len = np.array([0.5, 0.3, 0.2]) # Не совпадает с кол-вом активов
        self.risk_free_rate = 0.01 # 1% годовых

    def test_calculate_portfolio_return_valid(self):
        """Тест корректного расчета годовой доходности."""
        expected_mean_returns = self.returns_df.mean()
        expected_portfolio_daily_return = np.dot(self.weights_valid, expected_mean_returns)
        expected_annualized_return = expected_portfolio_daily_return * TRADING_PERIODS_PER_YEAR
        
        actual_return =  calculate_portfolio_return(self.returns_df, self.weights_valid)
        self.assertAlmostEqual(actual_return, expected_annualized_return, places=6)

    def test_calculate_portfolio_return_invalid_input_type(self):
        """Тест на неверный тип входных данных для доходности."""
        with self.assertRaises(TypeError):
             calculate_portfolio_return("not a dataframe", self.weights_valid)
        with self.assertRaises(TypeError):
             calculate_portfolio_return(self.returns_df, [0.6, 0.4]) # не ndarray

    def test_calculate_portfolio_return_mismatch_assets_weights(self):
        """Тест на несоответствие количества активов и весов."""
        with self.assertRaises(ValueError):
             calculate_portfolio_return(self.returns_df, self.weights_invalid_len)
            
    def test_calculate_portfolio_return_weights_sum_warning(self):
        """Тест предупреждения, если сумма весов не равна 1 (но расчет продолжается)."""
        # Проверяем, что функция не падает и возвращает значение,
        # и что было вызвано предупреждение (сложнее протестировать стандартным unittest,
        # но для этого можно использовать unittest.mock.patch на logging.warning)
        with self.assertLogs(level='WARNING') as log: # Проверяем логи
            result =  calculate_portfolio_return(self.returns_df, self.weights_invalid_sum)
            self.assertTrue(any("Сумма весов" in message for message in log.output))
        self.assertIsNotNone(result)

    def test_calculate_annualized_volatility_valid(self):
        """Тест корректного расчета годовой волатильности."""
        cov_matrix = self.returns_df.cov()
        portfolio_variance_daily = np.dot(self.weights_valid.T, np.dot(cov_matrix, self.weights_valid))
        expected_annualized_volatility = np.sqrt(portfolio_variance_daily * TRADING_PERIODS_PER_YEAR)

        actual_volatility = calculate_annualized_volatility(self.returns_df, self.weights_valid)
        self.assertAlmostEqual(actual_volatility, expected_annualized_volatility, places=6)

    def test_calculate_annualized_volatility_invalid_input_type(self):
        """Тест на неверный тип входных данных для волатильности."""
        with self.assertRaises(TypeError):
            calculate_annualized_volatility(list(), self.weights_valid)
        with self.assertRaises(TypeError):
            calculate_annualized_volatility(self.returns_df, (0.6, 0.4))

    def test_calculate_sharpe_ratio_valid(self):
        """Тест корректного расчета коэффициента Шарпа."""
        # Используем заранее рассчитанные значения для простоты
        ann_return = 0.10  # 10%
        ann_volatility = 0.15 # 15%
        expected_sharpe = (ann_return - self.risk_free_rate) / ann_volatility
        
        actual_sharpe = calculate_sharpe_ratio(ann_return, ann_volatility, self.risk_free_rate)
        self.assertAlmostEqual(actual_sharpe, expected_sharpe, places=6)

    def test_calculate_sharpe_ratio_zero_volatility(self):
        """Тест расчета Шарпа при нулевой волатильности."""
        ann_return = 0.05
        ann_volatility_zero = 0.0
        # Проверяем, что возвращается NaN и было предупреждение
        with self.assertLogs(level='WARNING') as log:
            sharpe_nan = calculate_sharpe_ratio(ann_return, ann_volatility_zero, self.risk_free_rate)
            self.assertTrue(any("Волатильность близка к нулю" in message for message in log.output),
                "Ожидалось предупреждение о близкой к нулю волатильности в логах")
        self.assertTrue(np.isnan(sharpe_nan))

    def test_get_portfolio_performance_valid_data(self):
        """
        Тест корректного возврата всех метрик производительности для валидных данных.
        """
        # Рассчитываем ожидаемые значения, вызывая уже протестированные функции
        expected_return = calculate_portfolio_return(self.returns_df, self.weights_valid)
        expected_volatility = calculate_annualized_volatility(self.returns_df, self.weights_valid)
        expected_sharpe = calculate_sharpe_ratio(expected_return, expected_volatility, self.risk_free_rate)

        # Вызываем тестируемую функцию
        actual_return, actual_volatility, actual_sharpe = get_portfolio_performance(
            self.returns_df, self.weights_valid, self.risk_free_rate
        )

        # Сравниваем результаты
        self.assertAlmostEqual(actual_return, expected_return, places=6)
        self.assertAlmostEqual(actual_volatility, expected_volatility, places=6)
        self.assertAlmostEqual(actual_sharpe, expected_sharpe, places=6)

    def test_get_portfolio_performance_handles_errors_from_children(self):
        """
        Тест, что get_portfolio_performance возвращает NaN, если внутренние функции
        (например, calculate_portfolio_return) вызывают ошибку из-за неверных данных.
        """
        # 1. Случай, когда daily_returns_df - не DataFrame (вызовет TypeError в дочерних функциях)
        # Используем type: ignore, чтобы подавить ошибку типа от статического анализатора,
        # так как мы специально передаем неверный тип для теста.
        ret_nan1, vol_nan1, shp_nan1 = get_portfolio_performance(
            "not_a_dataframe", self.weights_valid, self.risk_free_rate # type: ignore
        )
        self.assertTrue(np.isnan(ret_nan1), "Return должен быть NaN при ошибке в daily_returns_df")
        self.assertTrue(np.isnan(vol_nan1), "Volatility должен быть NaN при ошибке в daily_returns_df")
        self.assertTrue(np.isnan(shp_nan1), "Sharpe должен быть NaN при ошибке в daily_returns_df")

        # 2. Случай, когда длина весов не совпадает с количеством активов (вызовет ValueError)
        ret_nan2, vol_nan2, shp_nan2 = get_portfolio_performance(
            self.returns_df, self.weights_invalid_len, self.risk_free_rate
        )
        self.assertTrue(np.isnan(ret_nan2), "Return должен быть NaN при несоответствии длин")
        self.assertTrue(np.isnan(vol_nan2), "Volatility должен быть NaN при несоответствии длин")
        self.assertTrue(np.isnan(shp_nan2), "Sharpe должен быть NaN при несоответствии длин")

    def test_get_portfolio_performance_zero_volatility(self):
        """
        Тест, что get_portfolio_performance корректно обрабатывает нулевую волатильность
        (sharpe_ratio должен быть NaN).
        """
        # Создаем данные, где волатильность будет 0 (все доходности одинаковые)
        constant_returns_data = np.full((TRADING_PERIODS_PER_YEAR, 2), 0.001) # Все доходности 0.1%
        constant_returns_df = pd.DataFrame(constant_returns_data, columns=['AssetA', 'AssetB'])

        ret, vol, shp = get_portfolio_performance(
            constant_returns_df, self.weights_valid, self.risk_free_rate
        )

        expected_return = calculate_portfolio_return(constant_returns_df, self.weights_valid)
        self.assertAlmostEqual(ret, expected_return, places=6)
        self.assertAlmostEqual(vol, 0.0, places=6, msg="Ожидалась нулевая волатильность") # Волатильность должна быть близка к 0
        self.assertTrue(np.isnan(shp), "Коэффициент Шарпа должен быть NaN при нулевой волатильности")

if __name__ == '__main__':
    unittest.main()