import customtkinter as ctk
import portfolio_calculator 
import pandas as pd 
from typing import Optional, List, Dict
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import gc

# Импортируем функцию настройки UI
import ui_setup 

# Импорты для графика
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Настройки внешнего вида
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("green") 

# --- Основное приложение ---
app = ctk.CTk()
app.title("Оптимизатор Портфеля v0.2")
app.geometry("1000x750")

# --- Глобальные переменные для холста и панели инструментов графика ---
canvas_widget = None
toolbar_widget = None
plot_no_data_label_widget = None
current_fig = None

# --- Создаем виджеты с помощью функции из ui_setup ---
# Функция setup_main_window возвращает словарь с виджетами
widgets = ui_setup.setup_main_window(app) 

# --- Функция-обработчик нажатия кнопки ---
def calculate_button_callback():
    print("Нажата кнопка 'Рассчитать'")

    # --- Шаг 0: Первичная очистка и установка начального статуса ---
    widgets['status_label'].configure(text="Обработка запроса...", text_color="gray")
    widgets['result_display_textbox'].configure(state="normal")
    widgets['result_display_textbox'].delete("1.0", "end")
    widgets['result_display_textbox'].configure(state="disabled")
    clear_plot() # Очищаем график
    app.update_idletasks() # Обновляем интерфейс

    # --- Шаг 1: Получение и валидация тикеров ---
    tickers_string = widgets['ticker_entry'].get()
    if not tickers_string:
        widgets['status_label'].configure(text="Ошибка: Введите тикеры акций.", text_color="red")
        return
    tickers_list = [ticker.strip().upper() for ticker in tickers_string.split(',') if ticker.strip()]
    if not tickers_list:
        widgets['status_label'].configure(text="Ошибка: Введите корректные тикеры.", text_color="red")
        return
    if len(tickers_list) > 10:
        widgets['status_label'].configure(text="Ошибка: Введите не более 10 тикеров.", text_color="red")
        return

    # --- Шаг 2: Получение и валидация дат ---
    start_date_str_input = widgets['start_date_entry'].get()
    end_date_str_input = widgets['end_date_entry'].get()
    try:
        start_date_obj = datetime.strptime(start_date_str_input, '%Y-%m-%d').date()
        end_date_obj = datetime.strptime(end_date_str_input, '%Y-%m-%d').date()
    except ValueError:
        widgets['status_label'].configure(text="Ошибка: Неверный формат даты. Используйте ГГГГ-ММ-ДД.", text_color="red")
        return

    today = date.today()
    yesterday = today - timedelta(days=1)
    final_end_date_obj = end_date_obj
    corrected_date_message_part = ""

    if start_date_obj > today:
        widgets['status_label'].configure(text="Ошибка: Начальная дата не может быть в будущем.", text_color="red")
        return
    if end_date_obj >= today:
        final_end_date_obj = yesterday
        corrected_date_message_part = f"Конечная дата скорректирована на {final_end_date_obj.strftime('%Y-%m-%d')}. "
        widgets['end_date_entry'].delete(0, "end")
        widgets['end_date_entry'].insert(0, final_end_date_obj.strftime('%Y-%m-%d'))
    if start_date_obj >= final_end_date_obj:
        widgets['status_label'].configure(text="Ошибка: Начальная дата должна быть раньше конечной (учитывая коррекцию).", text_color="red")
        return
    min_calendar_days_for_period = 42
    if (final_end_date_obj - start_date_obj).days < min_calendar_days_for_period:
        widgets['status_label'].configure(text=f"Ошибка: Период должен быть не менее {min_calendar_days_for_period} календарных дней.", text_color="red")
        return
    MAX_PERIOD_YEARS = 8
    MAX_PERIOD_DAYS = MAX_PERIOD_YEARS * 365.25
    if (final_end_date_obj - start_date_obj).days > MAX_PERIOD_DAYS:
        widgets['status_label'].configure(text=f"Ошибка: Период слишком большой (макс. {MAX_PERIOD_YEARS} лет).", text_color="red")
        return

    start_date_for_yfinance = start_date_obj.strftime('%Y-%m-%d')
    end_date_for_yfinance = final_end_date_obj.strftime('%Y-%m-%d')

    # --- Шаг 2.1: Получение и валидация безрисковой ставки ---
    risk_free_rate_str = widgets['risk_free_rate_entry'].get()
    try:
        risk_free_rate_percent = float(risk_free_rate_str)
        if risk_free_rate_percent < 0:
            widgets['status_label'].configure(text="Ошибка: Безрисковая ставка не может быть отрицательной.", text_color="red")
            return
        MAX_PROC = 50.0
        if risk_free_rate_percent > MAX_PROC:
            widgets['status_label'].configure(text=f"Ошибка: Безрисковая ставка слишком высока (макс. {MAX_PROC}%).", text_color="red")
            return
        risk_free_rate = risk_free_rate_percent / 100.0
    except ValueError:
        widgets['status_label'].configure(text="Ошибка: Неверный формат безрисковой ставки. Введите число (%).", text_color="red")
        return

    # --- Шаг 3: Загрузка данных ---
    status_before_load = corrected_date_message_part + f"Загрузка данных для: {tickers_list} ({start_date_for_yfinance} по {end_date_for_yfinance})..."
    widgets['status_label'].configure(text=status_before_load, text_color="gray")
    app.update_idletasks()

    historical_data_df = portfolio_calculator.load_historical_data(
        tickers=tickers_list,
        start_date=start_date_for_yfinance,
        end_date=end_date_for_yfinance
    )

    # --- Шаг 4: Обработка результатов загрузки и формирование статуса ---
    current_status_text = corrected_date_message_part # Начинаем с сообщения о коррекции даты, если оно было
    current_status_color = "gray" # По умолчанию
    can_proceed_to_optimization = True

    min_data_points_required = 20 # Минимальное количество дней с данными

    if historical_data_df is None or not isinstance(historical_data_df, pd.DataFrame) or historical_data_df.empty:
        current_status_text = "Ошибка: Не удалось загрузить данные ни для одного из тикеров. Проверьте тикеры и интернет-соединение."
        current_status_color = "red"
        can_proceed_to_optimization = False
    elif historical_data_df.shape[0] < min_data_points_required:
        current_status_text = f"Ошибка: Загружено слишком мало данных ({historical_data_df.shape[0]} дн.). Требуется минимум {min_data_points_required} дн."
        current_status_color = "red"
        can_proceed_to_optimization = False
    else:
        loaded_tickers = historical_data_df.columns.tolist()
        all_input_tickers_set = set(tickers_list)
        loaded_tickers_set = set(loaded_tickers)
        failed_to_load_tickers = list(all_input_tickers_set - loaded_tickers_set)

        if failed_to_load_tickers:
            current_status_text += f"Предупреждение: Не удалось загрузить: {failed_to_load_tickers}. "
            current_status_color = "orange"
        
        if not loaded_tickers: # Этот случай уже отловлен выше, но для полной уверенности
            current_status_text += "Ошибка: Нет данных для расчета после фильтрации."
            current_status_color = "red"
            can_proceed_to_optimization = False
        elif len(loaded_tickers) == 1:
            current_status_text += f"Расчет для одного актива: {loaded_tickers[0]}. Для диверсификации необходимо >1 актива. "
            current_status_color = "orange" # Даже если не было failed_to_load, это предупреждение
        elif len(loaded_tickers) > 1:
            if failed_to_load_tickers:
                current_status_text += f"Расчет для успешно загруженных: {loaded_tickers}. "
            else: # Все запрошенные и загруженные тикеры совпали
                current_status_text = corrected_date_message_part + f"Данные для {loaded_tickers} успешно загружены ({historical_data_df.shape[0]} строк). "
                current_status_color = "gray" # Или "green"

    # Устанавливаем финальный статус ПЕРЕД оптимизацией (или ошибку, если ее нельзя проводить)
    widgets['status_label'].configure(text=current_status_text.strip(), text_color=current_status_color)
    app.update_idletasks()

    if not can_proceed_to_optimization:
        return # Выход, если оптимизацию проводить нельзя

    # --- Шаг 4.1: Добавляем "Выполняется оптимизация" к текущему статусу ---
    # (только если нет критической ошибки красного цвета)
    if widgets['status_label'].cget("text_color") != "red":
        existing_status_text = widgets['status_label'].cget("text")
        widgets['status_label'].configure(
            text=existing_status_text + "Выполняется оптимизация...",
            text_color=widgets['status_label'].cget("text_color")
        )
        app.update_idletasks()

    # --- Шаг 5: Вызов функции-координатора оптимизации ---
    optimization_results = portfolio_calculator.calculate_portfolio_optimization_results(
        prices_df=historical_data_df,
        risk_free_rate=risk_free_rate
    )

    # --- Шаг 6: Обработка результатов оптимизации ---
    if optimization_results:
        final_status_message_parts = []
        final_status_color_after_opt = "green"

        # Если были предупреждения на этапе загрузки, добавим их к финальному сообщению
        if corrected_date_message_part:
             final_status_message_parts.append(corrected_date_message_part.strip())
        if failed_to_load_tickers:
            final_status_message_parts.append(f"Обработаны не все тикеры (не загружены: {failed_to_load_tickers}).")
            final_status_color_after_opt = "orange"
        if len(loaded_tickers) == 1:
            final_status_message_parts.append(f"Расчет для одного актива ({loaded_tickers[0]}).")
            final_status_color_after_opt = "orange"
        
        final_status_message_parts.append("Оптимизация завершена!")
        widgets['status_label'].configure(text=" ".join(final_status_message_parts), text_color=final_status_color_after_opt)

        display_text_results(optimization_results.get('mvp'),
                             optimization_results.get('msr'),
                             optimization_results.get('stats'))
        display_plot(optimization_results.get('frontier'),
                     optimization_results.get('mvp'),
                     optimization_results.get('msr'),
                     optimization_results.get('stats'))
        
        print("\n--- Результаты Оптимизации (из callback) ---") # Для консольной отладки
        if optimization_results.get('mvp'): print("MVP:", optimization_results['mvp'])
        if optimization_results.get('msr'): print("MSR:", optimization_results['msr'])
        if optimization_results.get('frontier') and 'returns' in optimization_results['frontier']:
            print("Frontier points:", len(optimization_results['frontier']['returns']))
    else:
        # Если оптимизация не удалась, но загрузка данных прошла (статус не красный)
        if widgets['status_label'].cget("text_color") != "red":
             # Берем текст ДО "Выполняется оптимизация..." и добавляем ошибку
             text_before_opt_attempt = widgets['status_label'].cget("text").split("Выполняется оптимизация...")[0]
             widgets['status_label'].configure(text=text_before_opt_attempt + " Ошибка в процессе оптимизации. Смотрите консоль.", text_color="red")
        # Если уже была красная ошибка загрузки, она останется

    gc.collect()
    print("Сборщик мусора вызван.")

def display_text_results(mvp_results: Optional[Dict], msr_results: Optional[Dict], stats_data: Optional[Dict]):
    textbox = widgets['result_display_textbox']
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")

    textbox.tag_config("h1", foreground="#64FFDA", underline=True)   # Бирюзовый, подчеркнутый
    textbox.tag_config("h2", foreground="#FFFFFF", underline=True) # Белый, подчеркнутый (можно сделать менее ярким, если нужно)
    textbox.tag_config("metric_label", foreground="lightgray")    # Метки типа "Ожидаемая Доходность:"
    textbox.tag_config("metric_value", foreground="#FFEB3B")      # Желтоватый для значений
    textbox.tag_config("asset_name", foreground="#E0E0E0")       # Светло-серый для тикеров
    textbox.tag_config("asset_weight", foreground="#FFFFFF")      # Белый для весов
    textbox.tag_config("info_text", foreground="gray") 

    textbox.insert("end", "--- Результаты Оптимизации ---\n\n", "h1")
    asset_names = []
    if not asset_names:
         if mvp_results and 'weights' in mvp_results and mvp_results['weights'] is not None:
              asset_names = [f"Актив {i+1}" for i in range(len(mvp_results['weights']))]
         elif msr_results and 'weights' in msr_results and msr_results['weights'] is not None:
              asset_names = [f"Актив {i+1}" for i in range(len(msr_results['weights']))]
         else:
            textbox.insert("end", "Не удалось определить активы или нет данных для результатов.\n", "info_text")
            textbox.configure(state="disabled")
            return

    if stats_data and 'mean_returns' in stats_data and not stats_data['mean_returns'].empty:
         asset_names = stats_data['mean_returns'].index.tolist()
    else: # Пытаемся получить из весов, если статистики нет
        if mvp_results and 'weights' in mvp_results and mvp_results['weights'] is not None:
            asset_names = [f"Актив {i+1}" for i in range(len(mvp_results['weights']))]
        elif msr_results and 'weights' in msr_results and msr_results['weights'] is not None:
            asset_names = [f"Актив {i+1}" for i in range(len(msr_results['weights']))]

    if not asset_names and (mvp_results or msr_results) : # Добавил проверку, что хотя бы один результат есть
         text_output += "Не удалось определить названия активов для отображения весов.\n\n"

    if mvp_results:
        textbox.insert("end", "Портфель Минимальной Дисперсии (MVP):\n", "h2")
        textbox.insert("end", "  Ожидаемая Доходность: ", "metric_label")
        textbox.insert("end", f"{mvp_results['return']:.2%}\n", "metric_value")
        textbox.insert("end", "  Волатильность (Риск): ", "metric_label")
        textbox.insert("end", f"{mvp_results['volatility']:.2%}\n", "metric_value")

        if asset_names and mvp_results.get('weights') is not None and len(asset_names) == len(mvp_results['weights']):
            textbox.insert("end", "  Состав портфеля:\n", "metric_label")
            for ticker, weight in zip(asset_names, mvp_results['weights']):
                if weight > 1e-4:
                    textbox.insert("end", "    - ", "metric_label")
                    textbox.insert("end", f"{ticker}", "asset_name")
                    textbox.insert("end", ": ", "metric_label")
                    textbox.insert("end", f"{weight:.2%}\n", "asset_weight")
        elif mvp_results.get('weights') is not None:
             textbox.insert("end", "  (Веса не сопоставлены с именами активов)\n", "info_text")
        textbox.insert("end", "\n")
    else:
        textbox.insert("end", "Портфель Минимальной Дисперсии (MVP): Не удалось рассчитать.\n\n", "info_text") # h2 убрал, это инфо

    if msr_results:
        textbox.insert("end", "Портфель Макс. Коэфф. Шарпа (MSR):\n", "h2")
        textbox.insert("end", "  Ожидаемая Доходность: ", "metric_label")
        textbox.insert("end", f"{msr_results['return']:.2%}\n", "metric_value")
        textbox.insert("end", "  Волатильность (Риск): ", "metric_label")
        textbox.insert("end", f"{msr_results['volatility']:.2%}\n", "metric_value")
        textbox.insert("end", "  Коэффициент Шарпа: ", "metric_label")
        textbox.insert("end", f"{msr_results['sharpe']:.4f}\n", "metric_value")

        if asset_names and msr_results.get('weights') is not None and len(asset_names) == len(msr_results['weights']):
            textbox.insert("end", "  Состав портфеля:\n", "metric_label")
            for ticker, weight in zip(asset_names, msr_results['weights']):
                if weight > 1e-4:
                    textbox.insert("end", "    - ", "metric_label")
                    textbox.insert("end", f"{ticker}", "asset_name")
                    textbox.insert("end", ": ", "metric_label")
                    textbox.insert("end", f"{weight:.2%}\n", "asset_weight")
        elif msr_results.get('weights') is not None:
             textbox.insert("end", "  (Веса не сопоставлены с именами активов)\n", "info_text")
    else:
        textbox.insert("end", "Портфель Макс. Коэфф. Шарпа (MSR): Не удалось рассчитать.\n", "info_text") # h2 убрал

    textbox.configure(state="disabled")
    print("Текстовые результаты (с форматированием цветом/подчеркиванием) выведены в CTkTextbox.")

def clear_plot():
    global canvas_widget, toolbar_widget, plot_no_data_label_widget, current_fig
    if canvas_widget:
        canvas_widget.get_tk_widget().destroy()
        canvas_widget = None
    if toolbar_widget:
        toolbar_widget.destroy()
        toolbar_widget = None
    if plot_no_data_label_widget and plot_no_data_label_widget.winfo_exists():
        plot_no_data_label_widget.destroy()
        plot_no_data_label_widget = None
    
    # --- НОВОЕ: Явно закрываем предыдущую фигуру Matplotlib ---
    if current_fig:
        plt.close(current_fig) # Закрываем фигуру
        current_fig = None
    # --- КОНЕЦ НОВОГО ---
    print("Область графика и фигура Matplotlib очищены.")
    gc.collect()
    print("Сборщик мусора вызван.")

def display_plot(frontier_data: Optional[Dict],
                 mvp_results: Optional[Dict],
                 msr_results: Optional[Dict],
                 stats_data: Optional[Dict]):
    global canvas_widget, toolbar_widget, plot_no_data_label_widget, current_fig
    clear_plot()
    plot_frame = widgets['plot_frame']

    # Проверяем, есть ли хоть какие-то данные для графика
    has_frontier = frontier_data and 'volatilities' in frontier_data and 'returns' in frontier_data and len(frontier_data['returns']) > 0
    has_mvp = mvp_results and 'volatility' in mvp_results and 'return' in mvp_results
    has_msr = msr_results and 'volatility' in msr_results and 'return' in msr_results
    has_individual_assets = stats_data and 'mean_returns' in stats_data and 'cov_matrix' in stats_data and not stats_data['mean_returns'].empty

    if not (has_frontier or has_mvp or has_msr or has_individual_assets):
        print("Нет данных для построения графика.")
        plot_no_data_label_widget = ctk.CTkLabel(plot_frame, text="Нет данных для отображения графика.")
        # Размещаем метку по центру plot_frame
        plot_no_data_label_widget.grid(row=0, column=0, sticky="nsew")
        plot_frame.grid_rowconfigure(0, weight=1) # Чтобы метка была по центру вертикально
        plot_frame.grid_columnconfigure(0, weight=1) # Чтобы метка была по центру горизонтально
        return

    fig_face_color = '#2B2B2B'
    current_fig = Figure(figsize=(7, 5), dpi=100, facecolor=fig_face_color) 
    fig = Figure(figsize=(7, 5), dpi=100, facecolor=fig_face_color)
    ax = current_fig.add_subplot(111)

    ax.set_facecolor(fig_face_color)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

    if has_frontier:
        ax.plot(frontier_data['volatilities'], frontier_data['returns'], 'b-', label='Граница Эффективности', linewidth=2)
    if has_mvp:
        ax.scatter(mvp_results['volatility'], mvp_results['return'], marker='o', color='red', s=100, label='MVP (Мин. Риск)')
    if has_msr:
        ax.scatter(msr_results['volatility'], msr_results['return'], marker='*', color='green', s=150, label='MSR (Макс. Шарп)')
    if has_individual_assets:
        asset_returns = stats_data['mean_returns']
        asset_cov_matrix = stats_data['cov_matrix']
        for i, ticker in enumerate(asset_returns.index):
            asset_volatility = np.sqrt(asset_cov_matrix.iloc[i, i])
            ax.scatter(asset_volatility, asset_returns.iloc[i], marker='x', s=80, label=ticker)

    ax.set_title('Граница Эффективности Портфеля', color='white')
    ax.set_xlabel('Волатильность (Риск)', color='white')
    ax.set_ylabel('Ожидаемая Доходность', color='white')
    if has_frontier or has_mvp or has_msr or has_individual_assets: # Отображаем легенду, только если есть что отображать
        ax.legend(facecolor='#404040', edgecolor='white', labelcolor='white', loc='best')
    ax.grid(True, linestyle='--', alpha=0.5, color='#808080') # Используем HEX для цвета сетки

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(current_fig, master=plot_frame)
    canvas_widget = canvas
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=False)
    toolbar_widget = toolbar
    toolbar.update()
    toolbar.grid(row=1, column=0, sticky="ew")
    # plot_frame.grid_rowconfigure(1, weight=0) # Это можно убрать, если toolbar нормально размещается

    print("График отображен в GUI (новый макет).")

# --- Привязываем команду к кнопке ПОСЛЕ ее создания ---
widgets['calculate_button'].configure(command=calculate_button_callback)

# --- Запуск главного цикла приложения ---
app.mainloop()