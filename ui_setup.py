import customtkinter as ctk

def setup_main_window(app: ctk.CTk) -> dict:
    """
    Создает и настраивает основные виджеты главного окна приложения.

    Args:
        app (ctk.CTk): Главный объект приложения (окно).

    Returns:
        dict: Словарь, содержащий созданные виджеты для доступа к ним 
              из основного файла приложения. Ключи - имена виджетов (строки), 
              значения - объекты виджетов CTk.
    """
    widgets = {} # Словарь для хранения виджетов

    # Заголовок/Инструкция для тикеров
    widgets['ticker_title_label'] = ctk.CTkLabel(app, text="Введите тикеры акций через запятую:")
    widgets['ticker_title_label'].pack(pady=(10, 0)) 

    # Поле для ввода тикеров
    widgets['ticker_entry'] = ctk.CTkEntry(app, placeholder_text="AAPL, MSFT, GOOG", width=300)
    widgets['ticker_entry'].pack(pady=(5, 10)) 

    # --- Поля для дат ---
    date_frame = ctk.CTkFrame(app, fg_color="transparent") 
    date_frame.pack(pady=5)

    start_date_label = ctk.CTkLabel(date_frame, text="Начальная дата (ГГГГ-ММ-ДД):")
    start_date_label.pack(side=ctk.LEFT, padx=(0, 5)) 
    widgets['start_date_entry'] = ctk.CTkEntry(date_frame, placeholder_text="2021-01-01", width=120)
    widgets['start_date_entry'].pack(side=ctk.LEFT)

    end_date_label = ctk.CTkLabel(date_frame, text="Конечная дата (ГГГГ-ММ-ДД):")
    end_date_label.pack(side=ctk.LEFT, padx=(15, 5)) 
    widgets['end_date_entry'] = ctk.CTkEntry(date_frame, placeholder_text="2023-12-31", width=120)
    widgets['end_date_entry'].pack(side=ctk.LEFT)
    # --- Конец полей для дат ---

    # Кнопка для запуска расчета (command пока не привязываем здесь)
    widgets['calculate_button'] = ctk.CTkButton(app, text="Рассчитать портфель") 
    widgets['calculate_button'].pack(pady=20) 

    # Метка для отображения статуса или ошибок
    widgets['status_label'] = ctk.CTkLabel(app, text="", text_color="gray") 
    widgets['status_label'].pack(pady=5)
    
    # --- Место для будущих результатов ---
    # Можно сразу создать фреймы или текстовые поля
    
    # Фрейм для текстовых результатов
    widgets['results_frame'] = ctk.CTkFrame(app, fg_color="transparent")
    widgets['results_frame'].pack(pady=10, padx=10, fill="x") 
    # Внутри этого фрейма потом можно разместить метки или текстовое поле для весов
    
    # Фрейм для графика
    # widgets['plot_frame'] = ctk.CTkFrame(app, fg_color="lightgray")
    #widgets['plot_frame'].pack(pady=10, padx=10, fill="both", expand=True) 
    # Холст из Matplotlib
    
    # Добавим начальный текст или метки в results_frame, если нужно
    initial_result_label = ctk.CTkLabel(widgets['results_frame'], text="Результаты расчета появятся здесь...")
    initial_result_label.pack()
    widgets['result_display_label'] = initial_result_label # Сохраняем ссылку для обновления

    print("Виджеты GUI созданы и настроены.")
    return widgets