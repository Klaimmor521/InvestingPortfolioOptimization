import json
from typing import Dict, Optional, List
import logging
import os # Для работы с путями

# Имя файла для автоматического сохранения
DEFAULT_RESULTS_FILENAME = "calculation_history.json"

def auto_save_results_to_json(
        optimization_results: Optional[Dict],
        # tickers_input: List[str], # Убираем этот параметр, если хотим сохранять только обработанные
        processed_tickers: List[str], # НОВЫЙ ПАРАМЕТР
        start_date_input: str,
        end_date_input: str,
        risk_free_rate_input: float,
        filename: str = DEFAULT_RESULTS_FILENAME
    ) -> bool:
    if optimization_results is None:
        logging.warning("Нет результатов оптимизации для автоматического сохранения.")
        return False
    if not processed_tickers: # Если список обработанных тикеров пуст
         logging.warning("Нет обработанных тикеров для сохранения результатов.")
         return False


    data_to_save = {
        "request_parameters": {
            "tickers": processed_tickers, # ИСПОЛЬЗУЕМ processed_tickers
            "start_date": start_date_input,
            "end_date": end_date_input,
            "risk_free_rate_percent": risk_free_rate_input * 100
        },
        "results": {}
    }

    # asset_names_from_stats должен теперь совпадать с processed_tickers, если все хорошо
    asset_names_to_use = processed_tickers # Используем этот список для сопоставления с весами

    mvp_data = optimization_results.get('mvp')
    if mvp_data and mvp_data.get('weights') is not None:
        mvp_weights_dict = {}
        # Убедимся, что количество весов совпадает с количеством обработанных тикеров
        if len(asset_names_to_use) == len(mvp_data['weights']):
            for ticker, weight in zip(asset_names_to_use, mvp_data['weights']):
                if weight > 1e-5:
                    mvp_weights_dict[ticker] = round(weight, 4)
        else:
            logging.warning("Несоответствие количества тикеров и весов MVP при сохранении.")
            # Можно добавить заглушки, как раньше, или не сохранять веса, если есть несоответствие
            for i, weight in enumerate(mvp_data['weights']): # Fallback
                if weight > 1e-5: mvp_weights_dict[f"Вес_{i+1}"] = round(weight, 4)


        data_to_save["results"]["mvp"] = {
            "return_annual_percent": round(mvp_data['return'] * 100, 2),
            "volatility_annual_percent": round(mvp_data['volatility'] * 100, 2),
            "weights": mvp_weights_dict
        }

    msr_data = optimization_results.get('msr')
    if msr_data and msr_data.get('weights') is not None:
        msr_weights_dict = {}
        if len(asset_names_to_use) == len(msr_data['weights']):
            for ticker, weight in zip(asset_names_to_use, msr_data['weights']):
                if weight > 1e-5:
                    msr_weights_dict[ticker] = round(weight, 4)
        else:
            logging.warning("Несоответствие количества тикеров и весов MSR при сохранении.")
            for i, weight in enumerate(msr_data['weights']): # Fallback
                if weight > 1e-5: msr_weights_dict[f"Вес_{i+1}"] = round(weight, 4)


        data_to_save["results"]["msr"] = {
            "return_annual_percent": round(msr_data['return'] * 100, 2),
            "volatility_annual_percent": round(msr_data['volatility'] * 100, 2),
            "sharpe_ratio": round(msr_data['sharpe'], 4),
            "weights": msr_weights_dict
        }

    if not data_to_save["results"]:
        logging.warning("Нет данных MVP или MSR для автоматического сохранения (после фильтрации).")
        return False

    try:
        file_path = filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4, sort_keys=True)
        logging.info(f"Результаты автоматически сохранены в: {os.path.abspath(file_path)}")
        return True
    except Exception as e:
        logging.error(f"Ошибка при автоматическом сохранении результатов в JSON ({filename}): {e}", exc_info=True)
        return False

def clear_calculation_history(filename: str = DEFAULT_RESULTS_FILENAME) -> bool:
    """
    Очищает файл истории расчетов (удаляет его или записывает пустой список).
    """
    try:
        if os.path.exists(filename):
            os.remove(filename) # Удаляем файл, если он существует
            logging.info(f"Файл истории {filename} успешно удален (очищен).")
        else:
            logging.info(f"Файл истории {filename} не найден, очистка не требуется.")
        return True
    except Exception as e:
        logging.error(f"Ошибка при очистке файла истории {filename}: {e}", exc_info=True)
        return False