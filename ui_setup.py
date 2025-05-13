# В файле ui_setup.py
import customtkinter as ctk
from typing import Dict

def setup_main_window(app: ctk.CTk) -> Dict[str, ctk.CTkBaseClass]:
    widgets = {}

    # --- Верхняя часть: Ввод данных ---
    input_frame = ctk.CTkFrame(app, fg_color="transparent")
    input_frame.pack(pady=10, padx=10, fill="x")

    widgets['ticker_title_label'] = ctk.CTkLabel(input_frame, text="Введите тикеры акций через запятую:")
    widgets['ticker_title_label'].pack(pady=(0, 0))
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

    # --- НОВОЕ: Поле для ввода безрисковой ставки ---
    risk_free_frame = ctk.CTkFrame(input_frame, fg_color="transparent") # Добавляем в input_frame
    risk_free_frame.pack(pady=(10,0)) # Небольшой отступ сверху

    risk_free_label = ctk.CTkLabel(risk_free_frame, text="Безриск. ставка (% годовых):")
    risk_free_label.pack(side=ctk.LEFT, padx=(0, 5))
    widgets['risk_free_rate_entry'] = ctk.CTkEntry(risk_free_frame, placeholder_text="2.0", width=60) # Меньше ширина
    widgets['risk_free_rate_entry'].pack(side=ctk.LEFT)
    widgets['risk_free_rate_entry'].insert(0, "2.0") # Значение по умолчанию 2.0%
    # --- КОНЕЦ НОВОГО ---

    # --- Кнопка и статус под вводом ---
    control_frame = ctk.CTkFrame(app, fg_color="transparent")
    control_frame.pack(pady=10, padx=10, fill="x")
    widgets['calculate_button'] = ctk.CTkButton(control_frame, text="Рассчитать портфель")
    widgets['calculate_button'].pack() # Будет по центру этого фрейма
    widgets['status_label'] = ctk.CTkLabel(control_frame, text="", text_color="gray")
    widgets['status_label'].pack(pady=5)

    # --- Основная область: График слева, Результаты справа ---
    main_content_frame = ctk.CTkFrame(app, fg_color="transparent")
    main_content_frame.pack(pady=10, padx=10, fill="both", expand=True)

    # Настраиваем колонки для main_content_frame, чтобы они делили пространство
    main_content_frame.grid_columnconfigure(0, weight=2) # График займет больше места (например, 2/3)
    main_content_frame.grid_columnconfigure(1, weight=1) # Текст займет меньше (например, 1/3)
    main_content_frame.grid_rowconfigure(0, weight=1)    # Одна строка, растягивается по вертикали

    # Фрейм для графика (слева)
    widgets['plot_frame'] = ctk.CTkFrame(main_content_frame, fg_color="gray20")
    widgets['plot_frame'].grid(row=0, column=0, padx=(0, 5), pady=0, sticky="nsew")

    # Фрейм или Текстовое поле для результатов (справа)
    # Используем CTkTextbox для возможности скроллинга, если текста много
    widgets['result_display_textbox'] = ctk.CTkTextbox(main_content_frame, wrap="word", state="disabled", font=("Arial", 12))
    widgets['result_display_textbox'].grid(row=0, column=1, padx=(5, 0), pady=0, sticky="nsew")


    print("Виджеты GUI созданы и настроены для нового макета.")
    return widgets