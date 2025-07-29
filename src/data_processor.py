# src/data_processor.py
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Убедимся, что папка models существует
os.makedirs('models', exist_ok=True)

def load_programs_data(filepath='data/programs_data.json'):
    """Загружает данные программ из JSON файла."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filepath} не найден. Сначала запустите парсер.")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_chunks(programs_data):
    """Разбивает данные на фрагменты (chunks) для индексации."""
    chunks = []
    for program in programs_data:
        program_name = program['name']
        program_url = program['url']
        
        # Создаем чанки для каждого поля
        for field, content in program.items():
            if isinstance(content, str) and len(content) > 50: # Игнорируем короткие строки и не-строки
                 # Простое разбиение по предложениям для примера. Можно улучшить.
                 sentences = content.split('. ')
                 current_chunk = ""
                 for sentence in sentences:
                     if len(current_chunk) + len(sentence) < 800: # Примерный размер чанка
                         current_chunk += sentence + ". "
                     else:
                         if current_chunk:
                             chunks.append({
                                 "text": current_chunk.strip(),
                                 "source": program_name,
                                 "field": field,
                                 "url": program_url
                             })
                         current_chunk = sentence + ". "
                 if current_chunk: # Добавляем последний оставшийся чанк
                     chunks.append({
                         "text": current_chunk.strip(),
                         "source": program_name,
                         "field": field,
                         "url": program_url
                     })
            elif isinstance(content, list): # Например, направления подготовки
                list_text = f"{field}: " + ", ".join([f"{item.get('code', '')} {item.get('name', '')}" for item in content if isinstance(item, dict)])
                chunks.append({
                    "text": list_text,
                    "source": program_name,
                    "field": field,
                    "url": program_url
                })
            # Можно добавить обработку других типов данных (например, companies)
    return chunks

def build_vector_store(chunks, model_name='intfloat/multilingual-e5-large'):
    """Создает векторную базу знаний с использованием SentenceTransformer и FAISS."""
    print("Загрузка модели SentenceTransformer...")
    model = SentenceTransformer(model_name)
    print("Модель загружена.")

    print("Создание эмбеддингов...")
    texts = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Создано {len(embeddings)} эмбеддингов.")

    # Создание FAISS индекса
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Используем L2 distance
    index.add(np.array(embeddings).astype('float32')) # FAISS требует float32
    
    # Сохранение индекса и чанков
    faiss.write_index(index, 'models/faiss_index.bin')
    with open('models/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    
    print("Векторная база знаний сохранена в models/")

def main():
    """Главная функция для обработки данных и создания векторной базы."""
    programs_data = load_programs_data()
    chunks = create_chunks(programs_data)
    build_vector_store(chunks)

if __name__ == "__main__":
    main()
    