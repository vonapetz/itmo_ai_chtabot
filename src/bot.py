# src/bot.py
import os
import json
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, ConversationHandler
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv # Для загрузки переменных окружения

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Конфигурация ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") # <-- Читаем токен из .env
MODEL_NAME = 'intfloat/multilingual-e5-large' # Хорошая модель для русского
FAISS_INDEX_PATH = 'models/faiss_index.bin'
CHUNKS_PATH = 'models/chunks.json'
DATA_FILE_PATH = 'data/programs_data.json'

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Глобальные переменные для модели и индекса ---
model = None
index = None
chunks = None

# --- Состояния для рекомендаций ---
BACKGROUND, INTERESTS, CAREER = range(3)

# --- Сообщения бота на русском ---
START_MESSAGE = (
    "🤖 Привет! Я чат-бот для абитуриентов магистратур ИТМО по направлениям AI.\n\n"
    "Я могу ответить на вопросы о программах:\n"
    "🔹 'Искусственный интеллект'\n"
    "🔹 'AI и ML в технических системах'\n\n"
    "Просто задай мне вопрос!\n\n"
    "Хочешь узнать, какая программа тебе больше подходит? Напиши /recommend."
)

RECOMMEND_START_MESSAGE = (
    "Давай подберу тебе подходящую программу! Пожалуйста, ответь на несколько вопросов.\n\n"
    "1️⃣ Какой у тебя бэкграунд? (например, программирование, математика, физика, экономика)"
)
RECOMMEND_INTERESTS_MESSAGE = "2️⃣ Какие области ИИ тебе интересны больше всего? (например, Computer Vision, NLP, Reinforcement Learning, Data Engineering, AI Product)"
RECOMMEND_CAREER_MESSAGE = "3️⃣ Какую карьеру ты хочешь построить? (например, ML Engineer, Data Scientist, AI Product Manager, Data Analyst)"
RECOMMEND_RESULT_MESSAGE = (
    "На основе твоих ответов:\n"
    "- Бэкграунд: {background}\n"
    "- Интересы: {interests}\n"
    "- Карьерная цель: {career_goal}\n\n"
    "📊 Рекомендация: {recommended_program}\n"
    "📌 Причина: {reason}\n\n"
    "Теперь ты можешь задать мне вопросы об этой программе!"
)
RECOMMEND_CANCEL_MESSAGE = "Рекомендация отменена. Можешь задать любой вопрос о программах."
NOT_RELEVANT_MESSAGE = "Извините, я не могу найти информацию по вашему вопросу в данных о магистратурах ИТМО. Пожалуйста, задайте вопрос, связанный с программами 'Искусственный интеллект' или 'AI и ML в технических системах'."
ERROR_MESSAGE = "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
NO_DATA_MESSAGE = "Извините, бот еще не готов. Попробуйте позже."
BOT_READY_MESSAGE = "✅ Бот готов к работе!"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    await context.bot.send_message(chat_id=update.effective_chat.id, text=START_MESSAGE)

# --- Логика рекомендаций ---
async def recommend_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Начинает процесс рекомендации."""
    await update.message.reply_text(RECOMMEND_START_MESSAGE)
    return BACKGROUND

async def recommend_background(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получает бэкграунд."""
    context.user_data['background'] = update.message.text
    await update.message.reply_text(RECOMMEND_INTERESTS_MESSAGE)
    return INTERESTS

async def recommend_interests(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получает интересы."""
    context.user_data['interests'] = update.message.text
    await update.message.reply_text(RECOMMEND_CAREER_MESSAGE)
    return CAREER

async def recommend_career(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Получает карьерную цель и выдает рекомендацию."""
    context.user_data['career_goal'] = update.message.text
    
    background = context.user_data.get('background', '').lower()
    interests = context.user_data.get('interests', '').lower()
    career_goal = context.user_data.get('career_goal', '').lower()

    # Простое сопоставление ключевых слов
    ai_keywords = ['ml engineer', 'data engineer', 'data scientist', 'computer vision', 'nlp', 'reinforcement learning', 'машинное обучение', 'глубокое обучение', 'нейронные сети', 'алгоритмы', 'программирование', 'математика', 'статистика']
    ai_product_keywords = ['product manager', 'ai product', 'product', 'бизнес', 'аналитик', 'data analyst', 'системный подход', 'экономика', 'менеджмент', 'управление проектами']

    ai_score = sum(1 for keyword in ai_keywords if keyword in interests or keyword in career_goal or keyword in background)
    ai_product_score = sum(1 for keyword in ai_product_keywords if keyword in interests or keyword in career_goal or keyword in background)

    if ai_score > ai_product_score:
        recommended_program = "Искусственный интеллект"
        reason = "так как ты упомянул роли или области, связанные с разработкой и инженерией ИИ (ML Engineer, Data Engineer, Computer Vision и т.д.)."
    elif ai_product_score > ai_score:
        recommended_program = "AI и ML в технических системах"
        reason = "так как ты упомянул интерес к продуктам, бизнесу или анализу данных (AI Product Manager, Data Analyst и т.д.)."
    else:
        # Если счет равный или нулевой, можно предложить обе или одну по умолчанию
        recommended_program = "Искусственный интеллект"
        reason = "так как эта программа имеет более широкий охват технических ролей в области ИИ."

    final_message = RECOMMEND_RESULT_MESSAGE.format(
        background=context.user_data['background'],
        interests=context.user_data['interests'],
        career_goal=context.user_data['career_goal'],
        recommended_program=recommended_program,
        reason=reason
    )
    
    await update.message.reply_text(final_message)
    return ConversationHandler.END

async def recommend_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отменяет процесс рекомендации."""
    await update.message.reply_text(RECOMMEND_CANCEL_MESSAGE)
    return ConversationHandler.END

# --- Логика обработки вопросов ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик всех текстовых сообщений."""
    user_question = update.message.text
    logger.info(f"Получен вопрос от пользователя {update.effective_user.first_name}: {user_question}")

    if not model or not index or not chunks:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=NO_DATA_MESSAGE)
        return

    try:
        # 1. Создание эмбеддинга вопроса
        question_embedding = model.encode([user_question])
        question_embedding = np.array(question_embedding).astype('float32')

        # 2. Поиск похожих чанков
        k = 3
        distances, indices = index.search(question_embedding, k)
        
        # 3. Фильтрация по релевантности
        relevant_chunks = []
        for i, idx in enumerate(indices[0]):
            if distances[0][i] < 100: # Эмпирический порог
                 relevant_chunks.append(chunks[idx])

        if not relevant_chunks:
             await context.bot.send_message(chat_id=update.effective_chat.id, text=NOT_RELEVANT_MESSAGE)
             return

        # 4. Формирование контекста и генерация ответа
        # Для простоты и без внешней LLM, создаем ответ из найденных фрагментов
        answer_parts = ["Вот информация, которая может быть вам полезна:\n"]
        for chunk in relevant_chunks:
            # Форматируем ответ, чтобы он был читаемым
            source_info = f"🔹 Источник: {chunk['source']}"
            if chunk['field'] == 'about':
                answer_parts.append(f"{source_info}\n📄 О программе: {chunk['text']}\n")
            elif chunk['field'] == 'career':
                answer_parts.append(f"{source_info}\n💼 Карьера: {chunk['text']}\n")
            elif chunk['field'] == 'directions':
                answer_parts.append(f"{source_info}\n🧭 Направления: {chunk['text']}\n")
            elif chunk['field'] == 'scholarships':
                answer_parts.append(f"{source_info}\n💰 Стипендии: {chunk['text']}\n")
            elif chunk['field'] == 'international':
                answer_parts.append(f"{source_info}\n🌍 Международные возможности: {chunk['text']}\n")
            elif chunk['field'] == 'companies':
                 answer_parts.append(f"{source_info}\n🏢 Компании: {chunk['text']}\n")
            else:
                answer_parts.append(f"{source_info}\n📝 {chunk['text']}\n")
        
        answer = "\n".join(answer_parts)
        
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ERROR_MESSAGE)

async def post_init(application: ApplicationBuilder) -> None:
    """Функция, вызываемая при запуске бота для загрузки модели и индекса."""
    global model, index, chunks
    logger.info("Загрузка модели SentenceTransformer...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Модель загружена.")

    logger.info("Загрузка FAISS индекса и чанков...")
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info("FAISS индекс и чанки загружены.")
        logger.info(BOT_READY_MESSAGE)
    else:
        logger.error("Не найдены файлы векторной базы знаний. Пожалуйста, сначала запустите data_processor.py")

def main():
    """Главная функция для запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN не найден. Пожалуйста, создайте файл .env и укажите токен.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    # --- Обработчики ---
    app.add_handler(CommandHandler("start", start))
    
    # ConversationHandler для рекомендаций
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('recommend', recommend_start)],
        states={
            BACKGROUND: [MessageHandler(filters.TEXT & ~filters.COMMAND, recommend_background)],
            INTERESTS: [MessageHandler(filters.TEXT & ~filters.COMMAND, recommend_interests)],
            CAREER: [MessageHandler(filters.TEXT & ~filters.COMMAND, recommend_career)],
        },
        fallbacks=[CommandHandler('cancel', recommend_cancel)],
    )
    app.add_handler(conv_handler)

    # Обработчик всех остальных текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Бот запущен. Ожидание сообщений...")
    app.run_polling()

if __name__ == '__main__':
    main()