import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "data-15-structure-3.csv"
OUTPUT_PATH = BASE_DIR / "data" / "culture_large_dataset_raw.csv"

KEYWORDS_DICT = {
    'is_painting_hint': ['холст', 'масло', 'акварель', 'картон'],
    'is_coin_hint': ['монета', 'рубль', 'копейка'],
    'is_document_hint': ['приказ', 'акт', 'дело']
}

def extract_features_from_text(text, keywords):
    """
    Ищет слова в тексте.
    Возвращает 1, если хотя бы одно слово из списка есть в тексте, иначе 0.
    """
    pass

def load_and_filter_data(file_path):
    """
    Функция агрузки и фильтрации из исходного датасета.
    """
    pass

def main(input_file, output_file):
    """
    Основной управляющий скрипт для сборки ML-датасета из сырых данных.
    """
    pass

if __name__ == "__main__":
    main(INPUT_PATH, OUTPUT_PATH)