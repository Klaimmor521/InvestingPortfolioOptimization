import customtkinter as ctk
import portfolio_calculator 
import pandas as pd 
from datetime import datetime

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
    # Доступ к виджетам теперь через словарь 'widgets'
    widgets['status_label'].configure(text="Обработка запроса...", text_color="gray") 
    app.update_idletasks() 

    # 1. Получаем тикеры
    tickers_string = widgets['ticker_entry'].get() # Используем widgets['ticker_entry']
    if not tickers_string:
        widgets['status_label'].configure(text="Ошибка: Введите тикеры акций.", text_color="red")
        return 
    tickers_list = [ticker.strip() for ticker in tickers_string.split(',') if ticker.strip()]
    if not tickers_list:
        widgets['status_label'].configure(text="Ошибка: Введите корректные тикеры.", text_color="red")
        return 

    # 2. Получаем и проверяем даты
    start_date_str = widgets['start_date_entry'].get() # Используем widgets['start_date_entry']
    end_date_str = widgets['end_date_entry'].get()   # Используем widgets['end_date_entry']
    
    try:
        start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
        if start_date_obj > end_date_obj:
             widgets['status_label'].configure(text="Ошибка: Начальная дата не может быть позже конечной.", text_color="red")
             return
    except ValueError:
        widgets['status_label'].configure(text="Ошибка: Неверный формат даты. Используйте ГГГГ-ММ-ДД.", text_color="red")
        return
    start_date = start_date_str
    end_date = end_date_str

    # 3. Вызываем функцию загрузки данных 
    widgets['status_label'].configure(text=f"Загрузка данных для: {tickers_list}...", text_color="gray")
    app.update_idletasks() 
    
    historical_data_df = portfolio_calculator.load_historical_data(
        tickers=tickers_list, 
        start_date=start_date, 
        end_date=end_date
    )

    # 4. Обрабатываем результат
    if historical_data_df is not None and isinstance(historical_data_df, pd.DataFrame):
        widgets['status_label'].configure(text=f"Данные успешно загружены! Получено {historical_data_df.shape[0]} строк.", text_color="green")
        
        # --- ОСТАЛЬНОЙ ЛОГИКА И ОТОБРАЖЕНИЯ ---
        # Например:
        # results_data = portfolio_calculator.calculate_efficient_frontier(...)
        # display_text_results(results_data['mvp'], results_data['max_sharpe']) 
        # display_plot(results_data['frontier_points'], ...) 
        
        print("\n--- Данные для дальнейшей обработки ---")
        print(historical_data_df.head()) 
        
    else:
        widgets['status_label'].configure(text="Ошибка при загрузке данных. Смотрите консоль.", text_color="red")

# --- Функции отображения результатов (могут быть здесь или тоже вынесены) ---

def display_text_results(mvp_results: dict, max_sharpe_results: dict):
    # Эта функция будет обновлять текстовую область в GUI
    # Например, widgets['result_display_label'].configure(text=...)
    print("Отображение текстовых результатов (пока не реализовано)")
    pass 

def display_plot(frontier_data: pd.DataFrame, mvp_results: dict, max_sharpe_results: dict):
     # Эта функция будет рисовать график и встраивать его в widgets['plot_frame']
     print("Отображение графика (пока не реализовано)")
     pass

# --- Привязываем команду к кнопке ПОСЛЕ ее создания ---
widgets['calculate_button'].configure(command=calculate_button_callback)

# --- Запуск главного цикла приложения ---
app.mainloop()