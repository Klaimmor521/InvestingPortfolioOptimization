import customtkinter

# Задаем базовые настройки внешнего вида
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("green")

class PortfolioApp(customtkinter.CTk):
    """ Основной класс приложения Portfolio Optimizer """

    def __init__(self):
        # Инициализируем родительский класс (CTk - это само окно)
        super().__init__()

        # --- Настройка основного окна ---
        self.title("Portfolio Optimizer v0.1") # Заголовок окна
        self.geometry("900x600")             # Размер окна (ширина x высота)

        # --- Виджеты (поля, кнопки, графики) ---
        # Создаем текстовую метку
        self.label = customtkinter.CTkLabel(master=self, text="Hi! This is my app :)")

        self.label.pack(pady=20, padx=20) # Добавим отступы сверху/снизу (pady) и слева/справа (padx)

# --- Вход в приложение ---
if __name__ == "__main__":
    # Создаем экземпляр нашего приложения
    app = PortfolioApp()
    # Запускаем главный цикл обработки событий Tkinter при запуске
    app.mainloop()