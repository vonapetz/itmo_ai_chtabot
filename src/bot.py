# src/bot.py
import os
import json
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters, ConversationHandler
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
# Импорты для OpenAI
from openai import OpenAI, APIError, RateLimitError, AuthenticationError

# --- Загрузка переменных окружения ---
load_dotenv()

# --- Конфигурация ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = 'intfloat/multilingual-e5-large'
FAISS_INDEX_PATH = 'models/faiss_index.bin'
CHUNKS_PATH = 'models/chunks.json'
DATA_FILE_PATH = 'data/programs_data.json'

# --- Настройка логирования ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Глобальные переменные ---
model = None
index = None
chunks = None
programs_data = None
# Инициализация клиента OpenAI
client = None

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
    "📊 Рекомендация: *{recommended_program}*\n"
    "📌 Причина: {reason}\n\n"
    "Теперь ты можешь задать мне вопросы об этой программе!"
)
RECOMMEND_CANCEL_MESSAGE = "Рекомендация отменена. Можешь задать любой вопрос о программах."

# --- Сообщение, если OpenAI API не настроен ---
NO_LLM_MESSAGE = (
    "Бот настроен, но API-ключ для генерации ответов (OpenAI) не найден.\n"
    "Пожалуйста, укажите `OPENAI_API_KEY` в файле `.env` для полноценной работы.\n"
    "В качестве альтернативы, вы можете использовать предыдущую версию бота без LLM."
)

# --- Сообщения об ошибках API ---
LLM_API_ERROR_MESSAGE = "К сожалению, возникла ошибка при обращении к сервису генерации ответов. Попробуйте задать вопрос позже."
LLM_AUTH_ERROR_MESSAGE = "Ошибка аутентификации с API генерации ответов. Обратитесь к администратору бота."
LLM_RATE_LIMIT_MESSAGE = "Превышен лимит запросов к сервису генерации ответов. Попробуйте задать вопрос через несколько минут."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start."""
    await context.bot.send_message(chat_id=update.effective_chat.id, text=START_MESSAGE, parse_mode='Markdown')

# --- Логика рекомендаций (без изменений) ---
async def recommend_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(RECOMMEND_START_MESSAGE)
    return BACKGROUND

async def recommend_background(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['background'] = update.message.text
    await update.message.reply_text(RECOMMEND_INTERESTS_MESSAGE)
    return INTERESTS

async def recommend_interests(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['interests'] = update.message.text
    await update.message.reply_text(RECOMMEND_CAREER_MESSAGE)
    return CAREER

async def recommend_career(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['career_goal'] = update.message.text
    
    background = context.user_data.get('background', '').lower()
    interests = context.user_data.get('interests', '').lower()
    career_goal = context.user_data.get('career_goal', '').lower()

    ai_keywords = ['ml engineer', 'data engineer', 'data scientist', 'computer vision', 'nlp', 'reinforcement learning', 'машинное обучение', 'глубокое обучение', 'нейронные сети', 'алгоритмы', 'программирование', 'математика', 'статистика', ' middle', 'middle']
    ai_product_keywords = ['product manager', 'ai product', 'product', 'бизнес', 'аналитик', 'data analyst', 'системный подход', 'экономика', 'менеджмент', 'управление проектами', 'developer']

    ai_score = sum(1 for keyword in ai_keywords if keyword in interests or keyword in career_goal or keyword in background)
    ai_product_score = sum(1 for keyword in ai_product_keywords if keyword in interests or keyword in career_goal or keyword in background)

    if ai_score > ai_product_score:
        recommended_program = "Искусственный интеллект"
        reason = "так как ты упомянул роли или области, связанные с разработкой и инженерией ИИ (ML Engineer, Data Engineer, Computer Vision и т.д.)."
    elif ai_product_score > ai_score:
        recommended_program = "AI и ML в технических системах"
        reason = "так как ты упомянул интерес к продуктам, бизнесу или анализу данных (AI Product Manager, Data Analyst и т.д.)."
    else:
        recommended_program = "Искусственный интеллект"
        reason = "так как эта программа имеет более широкий охват технических ролей в области ИИ."

    final_message = RECOMMEND_RESULT_MESSAGE.format(
        background=context.user_data['background'],
        interests=context.user_data['interests'],
        career_goal=context.user_data['career_goal'],
        recommended_program=recommended_program,
        reason=reason
    )
    
    await update.message.reply_text(final_message, parse_mode='Markdown')
    return ConversationHandler.END

async def recommend_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(RECOMMEND_CANCEL_MESSAGE)
    return ConversationHandler.END

# --- НОВАЯ ЛОГИКА ОБРАБОТКИ ВОПРОСОВ С ИСПОЛЬЗОВАНИЕМ LLM ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик всех текстовых сообщений."""
    user_question = update.message.text
    logger.info(f"Получен вопрос от пользователя {update.effective_user.first_name}: {user_question}")

    # Проверка наличия необходимых компонентов
    if not model or not index or not chunks:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Извините, бот еще не готов. Попробуйте позже.")
        return

    if not client:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=NO_LLM_MESSAGE)
        return

    try:
        # 1. Поиск релевантного контекста
        # Создание эмбеддинга вопроса
        question_embedding = model.encode([user_question])
        question_embedding = np.array(question_embedding).astype('float32')

        # Поиск похожих чанков
        k = 3
        distances, indices = index.search(question_embedding, k)
        
        # Фильтрация по релевантности
        relevance_threshold = 80.0 
        best_distance = distances[0][0] if len(distances[0]) > 0 else float('inf')

        # 2. Подготовка данных для LLM
        system_prompt = (
            "Вы являетесь полезным помощником для абитуриентов, выбирающих магистерские программы "
            "ИТМО 'Искусственный интеллект' и 'AI и ML в технических системах'. "
            "Ваша задача - отвечать на вопросы абитуриентов на основе предоставленной информации. "
            "Информация будет содержаться в разделе 'Контекст'. "
            "Если в 'Контексте' нет информации для ответа на вопрос, вежливо сообщите, что не знаете ответа. "
            "Всегда отвечайте на русском языке. "
            "Не придумывайте факты, которых нет в контексте. "
            "Если вопрос не по теме программ ИТМО, вежливо укажите на это."
            "Отвечай приветливо и вежливо"
        )

        if best_distance > relevance_threshold:
            # Контекст не найден - сообщаем LLM, что информации нет
            user_prompt = (
                f"Вопрос абитуриента: {user_question}\n\n"
                f"Контекст: Информация по данному вопросу в базе данных не найдена. "
                f"Пожалуйста, вежливо сообщите абитуриенту, что вы не можете ответить на этот вопрос, "
                f"так как он не относится к программам магистратуры ИТМО по ИИ. "
                f"Предложите задать вопросы о программах 'Искусственный интеллект' или 'AI и ML в технических системах'."
            )
        else:
            # Контекст найден - формируем его для LLM
            relevant_chunks = [chunks[idx] for i, idx in enumerate(indices[0]) if distances[0][i] <= relevance_threshold]
            
            # Формируем строку контекста
            context_parts = []
            for chunk in relevant_chunks:
                part = f"Источник: {chunk['source']}\nРаздел: {chunk['field']}\nИнформация: {chunk['text']}"
                context_parts.append(part)
            context_text = "\n\n---\n\n".join(context_parts)

            user_prompt = (
                f"Вопрос абитуриента: {user_question}\n\n"
                f"Контекст:\n{context_text}\n\n"
                f"Пожалуйста, ответьте на вопрос абитуриента, используя только информацию из контекста. "
                f"Если контекст не позволяет ответить, скажите, что информации недостаточно."
            )

        # 3. Вызов LLM
        logger.info("Отправка запроса к LLM...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="gpt-4o-mini", # Используем gpt-4o-mini
            max_tokens=1000,
            temperature=0.2 # Низкая температура для более точных и фактических ответов
        )
        logger.info("Ответ от LLM получен.")

        # 4. Отправка ответа пользователю
        answer = chat_completion.choices[0].message.content
        await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

    except AuthenticationError:
        logger.error("Ошибка аутентификации OpenAI API.")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=LLM_AUTH_ERROR_MESSAGE)
    except RateLimitError:
        logger.error("Превышен лимит запросов к OpenAI API.")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=LLM_RATE_LIMIT_MESSAGE)
    except APIError as e:
        logger.error(f"Ошибка API OpenAI: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=LLM_API_ERROR_MESSAGE)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке сообщения: {e}", exc_info=True)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Произошла непредвиденная ошибка. Попробуйте позже.")

# --- Обновленная функция post_init ---
async def post_init(application: ApplicationBuilder) -> None:
    """Функция, вызываемая при запуске бота для загрузки модели, индекса и клиента API."""
    global model, index, chunks, programs_data, client
    logger.info("Загрузка модели SentenceTransformer...")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Модель загружена.")

    logger.info("Загрузка FAISS индекса и чанков...")
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info("FAISS индекс и чанки загружены.")
    else:
        logger.error("Не найдены файлы векторной базы знаний. Пожалуйста, сначала запустите data_processor.py")
        return

    logger.info("Загрузка данных программ для рекомендаций...")
    if os.path.exists(DATA_FILE_PATH):
         with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
             programs_data = json.load(f)
         logger.info("Данные программ для рекомендаций загружены.")
    else:
         logger.warning("Файл данных программ не найден. Рекомендации будут ограничены.")

    # Инициализация клиента OpenAI
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            # Проверка подключения - ИСПРАВЛЕНО
            client.models.list() # Простой запрос для проверки
            logger.info("Клиент OpenAI API инициализирован и подключен.")
        except AuthenticationError:
            logger.error("Неверный API-ключ OpenAI. Проверьте файл .env.")
            client = None
        except Exception as e:
            logger.error(f"Ошибка при инициализации клиента OpenAI: {e}")
            client = None
    else:
        logger.warning("OPENAI_API_KEY не найден в .env. Функция генерации ответов будет недоступна.")
        client = None

    logger.info("✅ Бот готов к работе!")

def main():
    """Главная функция для запуска бота."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN не найден. Пожалуйста, создайте файл .env и укажите токен.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    # --- Обработчики ---
    app.add_handler(CommandHandler("start", start))
    
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

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("🚀 Бот запущен. Ожидание сообщений...")
    app.run_polling()

if __name__ == '__main__':
    main()
    