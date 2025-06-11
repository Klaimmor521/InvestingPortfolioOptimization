# ui_setup.py
import customtkinter as ctk
from typing import Dict

def setup_main_window(app: ctk.CTk) -> Dict[str, ctk.CTkBaseClass]:
    widgets = {}

    # --- Верхняя часть: Ввод данных ---
    input_frame = ctk.CTkFrame(app, fg_color="transparent")
    input_frame.pack(pady=10, padx=10, fill="x")

    widgets['ticker_title_label'] = ctk.CTkLabel(input_frame, text="Введите тикеры акций через запятую:")
    widgets['ticker_title_label'].pack(pady=(0, 0)) # Меньший отступ снизу для компактности
    widgets['ticker_entry'] = ctk.CTkEntry(input_frame, placeholder_text="GOOG, AAPL, MSFT", width=350)
    widgets['ticker_entry'].pack(pady=(5, 10))

    date_input_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    date_input_frame.pack(pady=5)

    start_date_label = ctk.CTkLabel(date_input_frame, text="Начальная дата (ГГГГ-ММ-ДД):")
    start_date_label.pack(side=ctk.LEFT, padx=(0, 5))
    widgets['start_date_entry'] = ctk.CTkEntry(date_input_frame, placeholder_text="2020-01-01", width=120)
    widgets['start_date_entry'].pack(side=ctk.LEFT)

    end_date_label = ctk.CTkLabel(date_input_frame, text="Конечная дата (ГГГГ-ММ-ДД):")
    end_date_label.pack(side=ctk.LEFT, padx=(15, 5))
    widgets['end_date_entry'] = ctk.CTkEntry(date_input_frame, placeholder_text="2023-12-31", width=120)
    widgets['end_date_entry'].pack(side=ctk.LEFT)

    risk_free_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    risk_free_frame.pack(pady=(10,0))

    risk_free_label = ctk.CTkLabel(risk_free_frame, text="Безриск. ставка (% годовых):")
    risk_free_label.pack(side=ctk.LEFT, padx=(0, 5))
    widgets['risk_free_rate_entry'] = ctk.CTkEntry(risk_free_frame, placeholder_text="2.0", width=60)
    widgets['risk_free_rate_entry'].pack(side=ctk.LEFT)
    widgets['risk_free_rate_entry'].insert(0, "2.0")

    # --- НОВОЕ: Поле для ввода количества точек для Границы Эффективности ---
    frontier_points_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
    frontier_points_frame.pack(pady=(5,0)) # Небольшой отступ сверху

    frontier_points_label = ctk.CTkLabel(frontier_points_frame, text="Точек для Границы Эффективности:")
    frontier_points_label.pack(side=ctk.LEFT, padx=(0, 5))
    widgets['frontier_points_entry'] = ctk.CTkEntry(frontier_points_frame, placeholder_text="50", width=60)
    widgets['frontier_points_entry'].pack(side=ctk.LEFT)
    widgets['frontier_points_entry'].insert(0, "50") # Значение по умолчанию 50 точек
    # --- КОНЕЦ НОВОГО ---

    # --- Кнопки управления ---
    # Фрейм-контейнер для кнопок, чтобы он мог быть отцентрирован
    buttons_container_frame = ctk.CTkFrame(app, fg_color="transparent")
    buttons_container_frame.pack(pady=10) # Отступы вокруг группы кнопок

    widgets['calculate_button'] = ctk.CTkButton(buttons_container_frame, text="Рассчитать портфель")
    widgets['calculate_button'].pack(side=ctk.LEFT, padx=(0, 10)) # Отступ справа

    widgets['clear_history_button'] = ctk.CTkButton(buttons_container_frame, text="Очистить Историю",
                                                    fg_color="firebrick", hover_color="darkred")
    widgets['clear_history_button'].pack(side=ctk.LEFT)

    # --- Статус-метка ---
    # Фрейм для статус-метки, чтобы она была под кнопками и могла растягиваться
    status_frame = ctk.CTkFrame(app, fg_color="transparent")
    status_frame.pack(pady=(0, 10), padx=10, fill="x") # Отступ снизу, растягивается по ширине

    widgets['status_label'] = ctk.CTkLabel(status_frame, text="", text_color="gray")
    widgets['status_label'].pack() # По умолчанию будет по центру status_frame

    # --- Основная область: График слева, Результаты справа ---
    main_content_frame = ctk.CTkFrame(app, fg_color="transparent")
    main_content_frame.pack(pady=(0,10), padx=10, fill="both", expand=True) # Убрал верхний отступ, добавил нижний

    main_content_frame.grid_columnconfigure(0, weight=3) # График чуть больше
    main_content_frame.grid_columnconfigure(1, weight=2) # Текст чуть меньше
    main_content_frame.grid_rowconfigure(0, weight=1)

    widgets['plot_frame'] = ctk.CTkFrame(main_content_frame, fg_color="gray20") # Можно оставить так или подобрать цвет
    widgets['plot_frame'].grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")

    widgets['result_display_textbox'] = ctk.CTkTextbox(main_content_frame, wrap="word", state="disabled", font=("Arial", 12))
    widgets['result_display_textbox'].grid(row=0, column=1, padx=(5, 0), pady=0, sticky="nsew")

    print("Виджеты GUI созданы и настроены с центрированными кнопками.")
    return widgets