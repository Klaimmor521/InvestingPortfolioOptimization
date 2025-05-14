# file_operations.py
import json
# import tkinter.filedialog as fd # Больше не нужен для автоматического сохранения
from typing import Dict, Optional, List
import logging
import os # Для работы с путями

# Имя файла для автоматического сохранения
DEFAULT_RESULTS_FILENAME = "last_optimization_results.json"

def auto_save_results_to_json(
        optimization_results: Optional[Dict],
        tickers_input: List[str],
        start_date_input: str,
        end_date_input: str,
        risk_free_rate_input: float,
        filename: str = DEFAULT_RESULTS_FILENAME # Можно передать имя файла, если нужно
    ) -> bool: # Возвращает True при успехе, False при ошибке
    """
    Автоматически сохраняет данные оптимизации в JSON файл с фиксированным именем.
    Перезаписывает файл при каждом вызове.
    """
    if optimization_results is None:
        logging.warning("Нет результатов оптимизации для автоматического сохранения.")
        return False

    data_to_save = {
        "request_parameters": {
            "tickers": tickers_input,
            "start_date": start_date_input,
            "end_date": end_date_input,
            "risk_free_rate_percent": risk_free_rate_input * 100
        },
        "results": {}
    }
    # ... (логика сбора mvp_data и msr_data остается такой же, как в предыдущем примере) ...
    asset_names_from_stats = []
    if optimization_results.get('stats') and \
       optimization_results['stats'].get('mean_returns') is not None and \
       not optimization_results['stats']['mean_returns'].empty:
        asset_names_from_stats = optimization_results['stats']['mean_returns'].index.tolist()

    mvp_data = optimization_results.get('mvp')
    if mvp_data and mvp_data.get('weights') is not None:
        mvp_weights_dict = {}
        # Используем asset_names_from_stats для имен, если они есть и совпадают по длине
        current_asset_names_mvp = asset_names_from_stats if asset_names_from_stats and len(asset_names_from_stats) == len(mvp_data['weights']) else [f"Актив_{i+1}" for i in range(len(mvp_data['weights']))]
        for ticker, weight in zip(current_asset_names_mvp, mvp_data['weights']):
            if weight > 1e-5:
                mvp_weights_dict[ticker] = round(weight, 4)
        data_to_save["results"]["mvp"] = {
            "return_annual_percent": round(mvp_data['return'] * 100, 2),
            "volatility_annual_percent": round(mvp_data['volatility'] * 100, 2),
            "weights": mvp_weights_dict
        }

    msr_data = optimization_results.get('msr')
    if msr_data and msr_data.get('weights') is not None:
        msr_weights_dict = {}
        current_asset_names_msr = asset_names_from_stats if asset_names_from_stats and len(asset_names_from_stats) == len(msr_data['weights']) else [f"Актив_{i+1}" for i in range(len(msr_data['weights']))]
        for ticker, weight in zip(current_asset_names_msr, msr_data['weights']):
            if weight > 1e-5:
                msr_weights_dict[ticker] = round(weight, 4)
        data_to_save["results"]["msr"] = {
            "return_annual_percent": round(msr_data['return'] * 100, 2),
            "volatility_annual_percent": round(msr_data['volatility'] * 100, 2),
            "sharpe_ratio": round(msr_data['sharpe'], 4),
            "weights": msr_weights_dict
        }

    if not data_to_save["results"]:
        logging.warning("Нет данных MVP или MSR для автоматического сохранения.")
        return False

    try:
        # Получаем путь к папке, где находится скрипт main_app.py
        # Это нужно, чтобы файл сохранялся рядом с программой, а не в текущей рабочей директории,
        # которая может быть разной в зависимости от того, как запускается скрипт.
        # Для этого main_app.py должен знать свой путь. Проще всего сохранять в текущую рабочую директорию,
        # но если запускать из другого места, файл окажется там.
        # Пока оставим сохранение в текущую рабочую директорию:
        file_path = filename # Используем переданное имя файла

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4, sort_keys=True)
        logging.info(f"Результаты автоматически сохранены в: {os.path.abspath(file_path)}") # Показываем полный путь
        return True
    except Exception as e:
        logging.error(f"Ошибка при автоматическом сохранении результатов в JSON ({filename}): {e}", exc_info=True)
        return False