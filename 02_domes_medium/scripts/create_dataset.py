import pandas as pd
from pathlib import Path
from domes_extractor import DomesExtractor

COLUMNS = ["file_name", "aspect_ratio", "solidity", "extent", "top_width_ratio", "hu_moment", "century", "target_shape"]

# Настройка путей к данным и директориям проекта
BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "labels.csv"
IMAGES_DIR  = BASE_DIR / "images"
OUTPUT_PATH = BASE_DIR / "data" / "domes_medium_dataset.csv"

def main():
    # Загрузка исходной разметки (CSV-файл с именами файлов и словами)
    df_raw = pd.read_csv(LABELS_PATH)
    # ...
    all_data = []
    # ...
    if all_data:
        # Создаем DataFrame с жестко заданным порядком столбцов (COLUMNS)
        df = pd.DataFrame(all_data, columns=COLUMNS)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Датасет готов! Сохранен: {OUTPUT_PATH}, всего строк: {len(df)}")
    else:
        print("PNG файлы не найдены или произошла ошибка при обработке.")

if __name__ == "__main__":
    main()