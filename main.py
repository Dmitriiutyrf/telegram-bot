import logging
import os
import asyncio
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Dummy функция, имитирующая обучение модели на загруженных данных
def train_model_on_data(file_path: str):
    # Здесь должна быть логика обучения (например, чтение CSV, обучение модели и т.д.)
    # Для демонстрации мы просто ждем 3 секунды и возвращаем сообщение.
    import time
    logger.info(f"Начинается обучение модели на данных из файла: {file_path}")
    time.sleep(3)
    logger.info("Обучение модели завершено успешно.")
    return "Модель успешно обучена на данных."

# Обработчик команды /upload, который просит пользователя отправить файл
async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пожалуйста, отправьте документ с обучающими данными.")

# Обработчик сообщений с документами
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        document = update.message.document
        file = await document.get_file()
        # Создаем папку "data", если ее нет
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", document.file_name)
        # Скачиваем файл
        await file.download_to_drive(custom_path=file_path)
        await update.message.reply_text(f"Файл '{document.file_name}' успешно загружен. Начинается обучение модели...")
        
        # Запуск обучения модели в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, train_model_on_data, file_path)
        await update.message.reply_text(result)
    else:
        await update.message.reply_text("Документ не найден.")

# Основная функция для запуска бота
def main():
    BOT_TOKEN = '7886745169:AAEXTW3SpmnCJ2FaHwffdCVwrY1ZLOGylGA'  # Ваш токен

    application = Application.builder().token(BOT_TOKEN).build()

    # Обработчик команды /upload
    application.add_handler(CommandHandler("upload", upload_command))
    # Обработчик для сообщений с документами
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))

    # Запуск бота
    application.run_polling()

if __name__ == '__main__':
    main()
