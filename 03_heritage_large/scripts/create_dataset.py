import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "data-15-structure-3.csv"
OUTPUT_PATH = BASE_DIR / "data" / "culture_large_dataset_raw.csv"

# Словарь ключевых слов для определения категорий предметов
KEYWORDS_DICT = {
    'is_painting_hint': ['холст', 'масло', 'акварель', 'пейзаж', 'портрет', 'рама'],
    'is_icon_hint': ['дерево', 'доска', 'темпера', 'левкас', 'сусальн', 'святой', 'богородиц', 'спас', 'ковчег'],
    'is_coin_hint': ['монета', 'рубль', 'копейка', 'серебр', 'медь', 'золот', 'чекан', 'аверс', 'реверс'],
    'is_doc_hint': ['приказ', 'акт', 'дело', 'письмо', 'папка', 'протокол', 'рукопись', 'машинопись']
}

# Словарь переименования столбцов
COLUMN_MAPPING = {
    'Название культурной ценности': 'title',
    'Идентифицирующие признаки предмета (надписи, подписи, клейма, пометы, экслибрисы и т.п.)': 'marks',
    'Описание состояния сохранности': 'condition',
    'Классификация культурной ценности': 'category',
    'Высота': 'height',
    'Ширина': 'width',
    'Длина': 'length',
    'Вес': 'weight',
    'дата создания объекта': 'create_date'
}

# Маппинг категорий для приведения их к унифицированным группам
CATEGORY_MAPPING = {
    'ЖИВОПИСЬ': 'ART',
    'РИСУНОК': 'ART',
    'ОТТИСК': 'ART',
    'ХУДОЖЕСТВЕННЫЕ ЦЕННОСТИ': 'ART',
    
    'ИКОНА': 'ICON',
    'РЕЛИГИОЗНЫЙ ПРЕДМЕТ': 'ICON',

    'МОНЕТА / МЕДАЛЬ': 'NUMISMATICS',
    'ПРЕДМЕТЫ НУМИЗМАТИКИ И БОНИСТИКИ': 'NUMISMATICS',
    'ПРЕДМЕТЫ ФАЛЕРИСТИКИ И СФРАГИСТИКИ': 'NUMISMATICS',
    
    'АРХИВНЫЕ ДОКУМЕНТЫ': 'DOCUMENTS',
    'ПЕЧАТНЫЕ ИЗДАНИЯ': 'DOCUMENTS',
    'ДОКУМЕНТ / КНИГА': 'DOCUMENTS'
}

def extract_features_from_text(rows, keywords):
    """
    Создает бинарный признак на основе поиска ключевых слов в текстовых полях.

    Функция объединяет содержимое столбцов 'title', 'marks' и 'condition' в единую 
    строку, приводит ее к нижнему регистру и проверяет наличие любого из 
    заданных ключевых слов.

    Args:
        rows (dict): Строка датафрейма, содержащая текстовые поля.
        keywords (list of str): Список ключевых слов для поиска.

    Returns:
        int: 1, если хотя бы одно ключевое слово найдено в тексте, иначе 0.
             В случае ошибки обработки возвращает 0.
    """
    try:
        # Объединяем текстовые поля для поиска ключевых слов
        title = str(rows.get('title', ''))
        marks = str(rows.get('marks', ''))
        cond = str(rows.get('condition', ''))

        text = (title + " " + marks + " " + cond).lower()

        # Проверяем наличие ключевых слов в объединенном тексте
        for word in keywords:
            if word.lower() in text:
                return 1
        return 0
    
    except Exception as e:
        print(f"Ошибка при обработке строки: {e}")
        return 0

def load_and_filter_data(input_path, output_path):
    """
    Выполняет полный цикл загрузки, очистки и трансформации исходного датасета.

    Основные этапы обработки:
    1. Загрузка CSV с автоматическим определением разделителя.
    2. Очистка имен столбцов и выборка целевых колонок (по COLUMN_MAPPING).
    3. Переименование столбцов и маппинг категорий.
    4. Генерация новых признаков на основе поиска ключевых слов в тексте.
    5. Сохранение обработанного результата.

    Args:
        input_path (str или Path): Путь к исходному CSV-файлу.
        output_path (str или Path): Путь для сохранения обработанного Raw-датасета.

    Returns:
        None: Функция сохраняет файл на диск и выводит статус выполнения в консоль.
    """
    try:
        # Загрузка исходных данных
        df = pd.read_csv(input_path, sep=None, engine='python')

        # Удаляем лишние пробелы по краям названий
        df.columns = df.columns.str.strip()

        # Оставляем только нужные колонки
        needed_cols = list(COLUMN_MAPPING.keys())
        df = df[needed_cols]
    
    # Блок обработки ошибок
    except FileNotFoundError:
        print(f"Критическая ошибка: Файл {input_path} не найден!")
        return
    
    except ValueError as e:
        print(f"Ошибка в структуре файла: {e}")
        return
    
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при загрузке: {e}")
        return

    # Переименовываем колонки в короткие английские
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # Применяем маппинг категорий, чтобы привести их к унифицированным группам, и удаляем строки без заданной категории.
    df['target_group'] = df['category'].map(CATEGORY_MAPPING)
    df = df.dropna(subset=['target_group']).copy()
    
    # Создаем новые признаки на основе ключевых слов в текстовых полях
    for feature_name, words in KEYWORDS_DICT.items():
        feature_values = []

        for index, row in df.iterrows():
            val = extract_features_from_text(row, words)
            feature_values.append(val)

        df[feature_name] = feature_values

    # Сохраняем обработанный датасет в CSV файл
    try:
        df.to_csv(output_path, index=False, encoding='utf-8', sep=',', decimal='.', na_rep='NA')
        print(f"Готово! Raw-датасет сохранен в {output_path}")
    
    except PermissionError:
        print(f"Ошибка: Не могу сохранить файл {output_path}. Возможно, он открыт в Excel?")
    
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при сохранении: {e}")

if __name__ == "__main__":
    load_and_filter_data(INPUT_PATH, OUTPUT_PATH)