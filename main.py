import logging
import os
import pickle
import csv
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
import requests
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- Настройка логирования ------------------- #
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------- Настройка OpenAI (GPT-4) ------------------- #
openai.api_base = "https://api.openai.com/v1"
openai.api_key = "sk-dfaf43362cde486d9590ff6a44e5bdc3"
openai.organization = "org-Pq3AjEI4F8tSxYmwskn3PTZw"

# ------------------- Настройки аналитики (заглушки) ------------------- #
GA_MEASUREMENT_ID = "G-XXXXXXXXXX"    # Замените на свои данные
GA_API_SECRET = "YOUR_API_SECRET"      # Замените на свои данные
YANDEX_COUNTER_ID = "YOUR_COUNTER_ID"  # Замените на свои данные
YANDEX_OAUTH_TOKEN = "YOUR_YANDEX_OAUTH_TOKEN"  # Замените на свои данные

# ------------------- Глобальные переменные ------------------- #
MODEL_FILE = "model.pkl"
TRAINING_DATA_FILE = "training_data.csv"  # Файл для внешних обучающих данных
CONV_KB_FILE = "conversational_kb.csv"      # Файл базы знаний разговорного характера
CONFIDENCE_THRESHOLD = 0.7

# ------------------- Инициализация глобальной модели ------------------- #
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=200))
])

def load_model():
    global model
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        logger.info("Модель загружена из %s", MODEL_FILE)
    else:
        logger.info("Файл модели не найден. Используется начальная модель.")

def save_model():
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    logger.info("Модель сохранена в %s", MODEL_FILE)

def train_model_on_data(file_path: str):
    try:
        extension = os.path.splitext(file_path)[1].lower()
        if extension == '.csv':
            df = pd.read_csv(file_path)
        elif extension == '.json':
            df = pd.read_json(file_path)
        elif extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            return f"Unsupported file format: {extension}"
        if df.empty:
            logger.info("Обучающие данные пусты.")
            return "Обучающие данные пусты."
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(solver='liblinear', max_iter=200))
        ])
        pipeline.fit(texts, labels)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(pipeline, f)
        global model
        model = pipeline
        logger.info("Модель успешно обучена на данных из %s", file_path)
        return "Модель успешно обучена на данных!"
    except Exception as e:
        logger.error("Ошибка обучения модели: %s", e)
        return f"Ошибка обучения модели: {e}"

def retrain_model():
    if os.path.exists(TRAINING_DATA_FILE):
        try:
            df = pd.read_csv(TRAINING_DATA_FILE)
            if df.empty:
                logger.info("Нет данных для переобучения.")
                return "Нет данных для переобучения."
            texts = df["text"].tolist()
            labels = df["label"].tolist()
            new_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(solver='liblinear', max_iter=200))
            ])
            new_pipeline.fit(texts, labels)
            global model
            model = new_pipeline
            save_model()
            logger.info("Модель переобучена на %d примерах.", len(texts))
            return f"Модель переобучена на {len(texts)} примерах."
        except Exception as e:
            logger.error("Ошибка переобучения модели: %s", e)
            return f"Ошибка переобучения модели: {e}"
    else:
        logger.info("Файл обучающих данных не найден.")
        return "Файл обучающих данных не найден."

# Функция для фонового переобучения через JobQueue (синхронная функция)
def training_job(context: ContextTypes.DEFAULT_TYPE):
    result = retrain_model()
    logger.info("Фоновое обучение: %s", result)

# ------------------- Модуль базы знаний (разговорной) ------------------- #
kb_vectorizer = None
kb_matrix = None
kb_data = None

def load_conversational_kb():
    global kb_data, kb_vectorizer, kb_matrix
    if os.path.exists(CONV_KB_FILE):
        kb_data = pd.read_csv(CONV_KB_FILE)
        if not kb_data.empty:
            kb_vectorizer = TfidfVectorizer()
            kb_matrix = kb_vectorizer.fit_transform(kb_data['question'].tolist())
            logger.info("База знаний загружена из %s", CONV_KB_FILE)
            return "База знаний загружена."
        else:
            return "База знаний пуста."
    else:
        return "Файл базы знаний не найден."

def query_knowledge_base(query):
    global kb_vectorizer, kb_matrix, kb_data
    if kb_vectorizer is None or kb_matrix is None:
        load_conversational_kb()
    if kb_vectorizer is None or kb_matrix is None:
        return "База знаний не доступна."
    query_vec = kb_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, kb_matrix)
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[0, best_idx])
    if best_score < 0.3:
        return "Извините, я не нашёл релевантного ответа."
    else:
        return kb_data.iloc[best_idx]['answer']

async def kb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(context.args)
    if not query:
        await update.message.reply_text("Пожалуйста, укажите вопрос для поиска в базе знаний, например: /kb Как работает блокчейн?")
        return
    answer = query_knowledge_base(query)
    await update.message.reply_text(answer)

# ------------------- Обработчики Telegram-бота ------------------- #
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я умный бот, который автоматически понимает, о чем идет речь. "
        "Просто пишите, и я постараюсь помочь!\n"
        "Для загрузки базы знаний используйте команду /upload, для поиска в базе знаний — /kb, "
        "для ручного переобучения — /train."
    )
    send_event_to_google("start_command", {"message": "Бот запущен"})
    send_event_to_yandex("start_command", {"message": "Бот запущен"})

async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Пожалуйста, отправьте документ с обучающими данными.\nПоддерживаемые форматы: CSV, JSON, Excel (XLS/XLSX)."
    )

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.document:
        document = update.message.document
        file = await document.get_file()
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", document.file_name)
        await file.download_to_drive(custom_path=file_path)
        await update.message.reply_text(f"Файл '{document.file_name}' успешно загружен. Начинается обучение модели...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, train_model_on_data, file_path)
        await update.message.reply_text(result)
        send_event_to_google("upload_command", {"filename": document.file_name})
        send_event_to_yandex("upload_command", {"filename": document.file_name})
    else:
        await update.message.reply_text("Документ не найден.")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = ' '.join(context.args)
    if not text:
        await update.message.reply_text("Введите текст для анализа, например: /analyze Пример текста")
        return
    load_model()
    try:
        probabilities = model.predict_proba([text])[0]
        max_prob = max(probabilities)
        if max_prob < CONFIDENCE_THRESHOLD:
            with open(TRAINING_DATA_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not os.path.exists(TRAINING_DATA_FILE) or os.path.getsize(TRAINING_DATA_FILE) == 0:
                    writer.writerow(["text", "label", "timestamp"])
                writer.writerow([text, 1, datetime.utcnow().isoformat()])
            await update.message.reply_text(
                f"Низкая уверенность модели ({max_prob:.2f}). Пример сохранён для дообучения."
            )
        else:
            prediction = model.predict([text])[0]
            await update.message.reply_text(f"Анализ: {prediction} (уверенность: {max_prob:.2f})")
        send_event_to_google("analyze_command", {"text_length": len(text)})
        send_event_to_yandex("analyze_command", {"text_length": len(text)})
    except Exception as e:
        await update.message.reply_text("Ошибка при анализе текста: " + str(e))

async def get_transaction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Введите хеш транзакции, например: /transaction 0x...")
        return
    tx_hash = context.args[0]
    try:
        response = requests.get(
            f'https://api.etherscan.io/api?module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}'
        )
        data = response.json()
        await update.message.reply_text(str(data))
        send_event_to_google("transaction_command", {"tx_hash": tx_hash})
        send_event_to_yandex("transaction_command", {"tx_hash": tx_hash})
    except Exception as e:
        await update.message.reply_text("Ошибка при получении данных транзакции: " + str(e))

async def chat_with_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message['content']
        await update.message.reply_text(reply)
        send_event_to_google("chat_command", {"message_length": len(user_message)})
        send_event_to_yandex("chat_command", {"message_length": len(user_message)})
    except Exception as e:
        await update.message.reply_text("Ошибка при общении с ИИ: " + str(e))

async def manual_train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    result = retrain_model()
    await update.message.reply_text(result)

async def kb_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = ' '.join(context.args)
    if not query:
        await update.message.reply_text("Пожалуйста, укажите вопрос для поиска в базе знаний, например: /kb Как работает блокчейн?")
        return
    answer = query_knowledge_base(query)
    await update.message.reply_text(answer)

async def auto_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message['content']
        await update.message.reply_text(reply)
    except Exception as e:
        await update.message.reply_text("Ошибка при общении с ИИ: " + str(e))

# ------------------- Функции аналитики (заглушки) ------------------- #
def send_event_to_google(event_name, event_params, client_id="555"):
    payload = {
      "client_id": client_id,
      "events": [
         {
            "name": event_name,
            "params": event_params
         }
      ]
    }
    url = f"https://www.google-analytics.com/mp/collect?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}"
    try:
        response = requests.post(url, json=payload)
        logger.info("Google Analytics event sent: %s, status: %s", event_name, response.status_code)
        return response.status_code, response.text
    except Exception as e:
        logger.error("Ошибка отправки события в GA: %s", e)
        return None, str(e)

def send_event_to_yandex(event_name, event_params):
    url = f"https://api-metrika.yandex.net/stat/v1/data?ids={YANDEX_COUNTER_ID}"
    payload = {
         "event_name": event_name,
         "event_params": event_params
    }
    headers = {"Authorization": f"OAuth {YANDEX_OAUTH_TOKEN}"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        logger.info("Yandex Metrica event sent: %s, status: %s", event_name, response.status_code)
        return response.status_code, response.text
    except Exception as e:
        logger.error("Ошибка отправки события в Yandex: %s", e)
        return None, str(e)

# ------------------- Основная функция для запуска бота ------------------- #
def main():
    BOT_TOKEN = "7886745169:AAEXTW3SpmnCJ2FaHwffdCVwrY1ZLOGylGA"
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("upload", upload_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("transaction", get_transaction))
    application.add_handler(CommandHandler("chat", chat_with_ai))
    application.add_handler(CommandHandler("train", manual_train))
    application.add_handler(CommandHandler("kb", kb_command))
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auto_chat))

    # Используем JobQueue для фонового переобучения модели каждые 3600 секунд (1 час)
    application.job_queue.run_repeating(training_job, interval=3600, first=10)

    application.run_polling()

if __name__ == '__main__':
    main()
