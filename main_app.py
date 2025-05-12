import customtkinter as ctk
import portfolio_calculator 
import pandas as pd 
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np

# Импортируем функцию настройки UI
import ui_setup 

# Настройки внешнего вида
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("green") 

# --- Основное приложение ---
app = ctk.CTk()
app.title("Оптимизатор Портфеля v0.2")
app.geometry("900x650")

# --- Создаем виджеты с помощью функции из ui_setup ---
# Функция setup_main_window возвращает словарь с виджетами
widgets = ui_setup.setup_main_window(app) 

# --- Функция-обработчик нажатия кнопки ---
def calculate_button_callback():
    print("Нажата кнопка 'Рассчитать'")
    widgets['status_label'].configure(text="Обработка запроса...", text_color="gray")
    widgets['result_display_label'].configure(text="") # Очищаем старые результаты
    # Очистка графика (потребуется реализовать, пока просто выводим сообщение)
    # clear_plot() 
    app.update_idletasks()

    # 1. Получаем тикеры (код без изменений)
    tickers_string = widgets['ticker_entry'].get() 
    if not tickers_string:
        widgets['status_label'].configure(text="Ошибка: Введите тикеры акций.", text_color="red")
        return
    tickers_list = [ticker.strip().upper() for ticker in tickers_string.split(',') if ticker.strip()] # Сразу в верхний регистр
    if not tickers_list:
        widgets['status_label'].configure(text="Ошибка: Введите корректные тикеры.", text_color="red")
        return

    # 2. Получаем и проверяем даты (код без изменений)
    start_date_str = widgets['start_date_entry'].get() 
    end_date_str = widgets['end_date_entry'].get()
    try:
        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
        if start_date_obj >= end_date_obj: # >= чтобы был хотя бы один день разницы
             widgets['status_label'].configure(text="Ошибка: Конечная дата должна быть позже начальной.", text_color="red")
             return
    except ValueError:
        widgets['status_label'].configure(text="Ошибка: Неверный формат даты. Используйте ГГГГ-ММ-ДД.", text_color="red")
        return
    start_date = start_date_str
    end_date = end_date_str
    
    # *** НОВОЕ: Устанавливаем безрисковую ставку (пока по умолчанию) ***
    risk_free_rate = 0.02 # 2%

    # 3. Вызываем функцию загрузки данных (код без изменений)
    widgets['status_label'].configure(text=f"Загрузка данных для: {tickers_list}...", text_color="gray")
    app.update_idletasks()
    
    # Используем исправленный загрузчик
    historical_data_df = portfolio_calculator.load_historical_data(
        tickers=tickers_list,
        start_date=start_date,
        end_date=end_date
    )

    # 4. Обрабатываем результат загрузки
    if historical_data_df is None or not isinstance(historical_data_df, pd.DataFrame) or historical_data_df.empty:
        widgets['status_label'].configure(text="Ошибка при загрузке данных. Смотрите консоль.", text_color="red")
        return # Прерываем выполнение, если данные не загружены

    widgets['status_label'].configure(text=f"Данные загружены ({historical_data_df.shape[0]} строк). Выполняется оптимизация...", text_color="gray")
    app.update_idletasks()
    
    # *** НОВОЕ: Вызов функции-координатора оптимизации ***
    optimization_results = portfolio_calculator.calculate_portfolio_optimization_results(
        prices_df=historical_data_df,
        risk_free_rate=risk_free_rate
    )
    
    # 5. Обработка результатов оптимизации
    if optimization_results:
        widgets['status_label'].configure(text="Оптимизация завершена!", text_color="green")
        
        # Вызываем функции отображения (которые пока плейсхолдеры)
        display_text_results(optimization_results.get('mvp'), optimization_results.get('msr'), optimization_results.get('stats')) # Передаем тикеры для заголовков
        display_plot(optimization_results.get('frontier'), optimization_results.get('mvp'), optimization_results.get('msr'), optimization_results.get('stats'))

        # Для отладки можно вывести в консоль
        print("\n--- Результаты Оптимизации ---")
        if optimization_results.get('mvp'):
            print("MVP:", optimization_results['mvp'])
        if optimization_results.get('msr'):
            print("MSR:", optimization_results['msr'])
        if optimization_results.get('frontier'):
            print("Frontier points:", len(optimization_results['frontier']['returns']))
            
    else:
        widgets['status_label'].configure(text="Ошибка в процессе оптимизации портфеля. Смотрите консоль.", text_color="red")

# --- Функции отображения результатов (могут быть здесь или тоже вынесены) ---

def display_text_results(mvp_results: Optional[Dict], msr_results: Optional[Dict], stats_data: Optional[Dict]): # Убираем tickers, добавляем stats_data
    """Обновляет текстовую область результатами MVP и MSR."""
    result_label = widgets['result_display_label']
    text_output = "--- Результаты Оптимизации ---\n\n"

    # Получаем имена активов из статистики (это самый надежный источник)
    asset_names = []
    if stats_data and 'mean_returns' in stats_data and not stats_data['mean_returns'].empty:
         asset_names = stats_data['mean_returns'].index.tolist()
    else:
         # Если статистики нет, пытаемся получить из весов (менее надежно)
         if mvp_results and 'weights' in mvp_results and len(mvp_results['weights']) > 0:
             # Нужен способ узнать имена из весов, если они не переданы как индекс Series
             # Пока оставим пустым или используем заглушки, если статистики нет
             asset_names = [f"Актив {i+1}" for i in range(len(mvp_results['weights']))] # Пример заглушки
         elif msr_results and 'weights' in msr_results and len(msr_results['weights']) > 0:
              asset_names = [f"Актив {i+1}" for i in range(len(msr_results['weights']))] # Пример заглушки


    if not asset_names:
         text_output += "Не удалось определить названия активов для отображения весов.\n\n"


    if mvp_results:
        text_output += "**Портфель Минимальной Дисперсии (MVP):**\n"
        text_output += f"  Ожидаемая Доходность: {mvp_results['return']:.2%}\n"
        text_output += f"  Волатильность (Риск): {mvp_results['volatility']:.2%}\n"
        if asset_names and len(asset_names) == len(mvp_results['weights']): # Проверяем совпадение длины
            text_output += "  Состав портфеля:\n"
            for ticker, weight in zip(asset_names, mvp_results['weights']):
                if weight > 1e-4:
                    text_output += f"    - {ticker}: {weight:.2%}\n"
        else:
             text_output += "  (Не удалось сопоставить веса с активами)\n"
        text_output += "\n"
    else:
        text_output += "**Портфель Минимальной Дисперсии (MVP):** Не удалось рассчитать.\n\n"

    if msr_results:
        text_output += "**Портфель Макс. Коэфф. Шарпа (MSR):**\n"
        text_output += f"  Ожидаемая Доходность: {msr_results['return']:.2%}\n"
        text_output += f"  Волатильность (Риск): {msr_results['volatility']:.2%}\n"
        text_output += f"  Коэффициент Шарпа: {msr_results['sharpe']:.4f}\n"
        if asset_names and len(asset_names) == len(msr_results['weights']): # Проверяем совпадение длины
            text_output += "  Состав портфеля:\n"
            for ticker, weight in zip(asset_names, msr_results['weights']):
                if weight > 1e-4:
                    text_output += f"    - {ticker}: {weight:.2%}\n"
        else:
            text_output += "  (Не удалось сопоставить веса с активами)\n"

    else:
        text_output += "**Портфель Макс. Коэфф. Шарпа (MSR):** Не удалось рассчитать.\n"

    result_label.configure(text=text_output)
    print("Текстовые результаты подготовлены (см. GUI).")


def display_plot(frontier_data: Optional[dict], mvp_results: Optional[dict], msr_results: Optional[dict], stats_data: Optional[dict]):
     """Отображает график Границы Эффективности."""
     # !!! ЭТУ ФУНКЦИЮ НУЖНО БУДЕТ РЕАЛИЗОВАТЬ С MATPLOTLIB + CUSTOMTKINTER !!!
     # Сейчас это просто плейсхолдер
     print("\nОтображение графика (пока не реализовано в GUI)")
     if frontier_data:
         print(f"  Получено {len(frontier_data['returns'])} точек для границы эффективности.")
     if mvp_results:
         print(f"  Точка MVP: Vol={mvp_results['volatility']:.4f}, Ret={mvp_results['return']:.4f}")
     if msr_results:
         print(f"  Точка MSR: Vol={msr_results['volatility']:.4f}, Ret={msr_results['return']:.4f}")
     if stats_data and 'mean_returns' in stats_data and 'cov_matrix' in stats_data:
         print("  Статистика для отдельных активов:")
         num_assets = len(stats_data['mean_returns'])
         for i in range(num_assets):
              ticker = stats_data['mean_returns'].index[i]
              ret = stats_data['mean_returns'].iloc[i]
              # Волатильность = корень из диагонального элемента ковариационной матрицы
              vol = np.sqrt(stats_data['cov_matrix'].iloc[i, i])
              print(f"    - {ticker}: Vol={vol:.4f}, Ret={ret:.4f}")

def clear_plot():
    print("Очистка области графика (пока не реализовано)")
    pass

# --- Привязываем команду к кнопке ПОСЛЕ ее создания ---
widgets['calculate_button'].configure(command=calculate_button_callback)

# --- Запуск главного цикла приложения ---
app.mainloop()