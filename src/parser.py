# src/parser.py
import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os

# Убедимся, что папка data существует
os.makedirs('data', exist_ok=True)

def clean_text(text):
    """Очищает текст от лишних пробелов и символов."""
    if not isinstance(text, str):
        return ""
    # Заменяем множественные пробелы и переносы строк на один пробел
    cleaned = re.sub(r'\s+', ' ', text.strip())
    return cleaned

def parse_program(url, program_name):
    """
    Парсит информацию о магистерской программе с указанного URL.
    """
    print(f"Парсинг {program_name}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к {url}: {e}")
        return None

    program_data = {"name": program_name, "url": url}

    # --- Извлечение ключевой информации ---
    # О программе
    about_section = soup.find('h2', string='о программе')
    if about_section:
        content = []
        for sibling in about_section.find_next_siblings():
            if sibling.name == 'h2':
                break
            text = sibling.get_text(strip=False)
            if text:
                content.append(text)
        program_data['about'] = clean_text(' '.join(content))

    # Карьера
    career_section = soup.find('h2', string='Карьера')
    if career_section:
        content = []
        for sibling in career_section.find_next_siblings():
            if sibling.name == 'h2':
                break
            text = sibling.get_text(strip=False)
            if text:
                content.append(text)
        program_data['career'] = clean_text(' '.join(content))

    # Компании (ты сможешь работать в компаниях)
    companies_section = soup.find('h2', string='ты сможешь работать в компаниях')
    if companies_section:
        companies = []
        # Ищем все изображения после заголовка
        for img in companies_section.find_all_next('img'):
            # Проверяем, что мы все еще в секции компаний (не дошли до следующего h2)
            current_h2 = img.find_parent()
            while current_h2 and current_h2.name != 'h2':
                current_h2 = current_h2.find_parent()
            if current_h2 and current_h2.get_text() != 'ты сможешь работать в компаниях':
                break # Вышли за пределы секции
            
            alt_text = img.get('alt')
            if alt_text:
                companies.append(clean_text(alt_text))
            else:
                # Если нет alt, пробуем найти текст рядом
                parent_text = img.find_parent().get_text(strip=True)
                if parent_text and parent_text != 'ты сможешь работать в компаниях':
                     companies.append(clean_text(parent_text))
                     
        program_data['companies'] = list(set(filter(None, companies))) # Убираем дубликаты и пустые

    # Направления подготовки
    directions_section = soup.find('h2', string='направления подготовки')
    if directions_section:
        directions = []
        # Ищем все блоки с направлениями
        current_code = None
        for item in directions_section.find_next_siblings():
             if item.name == 'h2': # Останавливаемся на следующем h2
                 break
             if item.name in ['div', 'p']:
                 # Ищем код направления (обычно в h5 или strong)
                 code_elem = item.find(['h5', 'strong'])
                 if code_elem:
                     current_code = clean_text(code_elem.get_text(strip=True))
                     # Ищем название направления (обычно сразу после кода или в следующем элементе)
                     # Простая логика: берем текст после кода
                     full_text = clean_text(item.get_text(strip=True))
                     if current_code and full_text.startswith(current_code):
                         name = full_text[len(current_code):].strip()
                     else:
                         name = "Название не указано"
                     directions.append({"code": current_code, "name": name})
        program_data['directions'] = directions

    # Стипендии
    scholarships_section = soup.find('h2', string='Стипендии')
    if scholarships_section:
        content = []
        for sibling in scholarships_section.find_next_siblings():
            if sibling.name == 'h2':
                break
            text = sibling.get_text(strip=False)
            if text:
                content.append(text)
        program_data['scholarships'] = clean_text(' '.join(content))

    # Международные возможности
    international_section = soup.find('h2', string='международные возможности')
    if international_section:
        content = []
        for sibling in international_section.find_next_siblings():
            if sibling.name == 'h2':
                break
            text = sibling.get_text(strip=False)
            if text:
                content.append(text)
        program_data['international'] = clean_text(' '.join(content))

    print(f"Парсинг {program_name} завершен.")
    return program_data

def main():
    """
    Главная функция для парсинга всех программ и сохранения данных.
    """
    programs = [
        {
            "name": "Искусственный интеллект",
            "url": "https://abit.itmo.ru/program/master/ai"
        },
        {
            "name": "AI и ML в технических системах",
            "url": "https://abit.itmo.ru/program/master/ai_product"
        }
    ]

    all_programs_data = []

    for prog in programs:
        data = parse_program(prog["url"], prog["name"])
        if data:
            all_programs_data.append(data)
        # Вежливая задержка между запросами
        time.sleep(1)

    # Сохранение данных в JSON файл
    with open('data/programs_data.json', 'w', encoding='utf-8') as f:
        json.dump(all_programs_data, f, ensure_ascii=False, indent=4)
    print("Все данные успешно сохранены в data/programs_data.json")

if __name__ == "__main__":
    main()