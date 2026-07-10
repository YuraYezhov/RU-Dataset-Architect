import pandas as pd
import json
from pathlib import Path
from domes_extractor import DomesExtractor

COLUMNS = ["file_name", "aspect_ratio", "solidity", "extent", "top_width_ratio", "hu_moment_1", "century", "target_shape"]

# Настройка путей к директориям проекта
BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "intermediate_data" / "all_domes.json"
IMAGES_DIR = BASE_DIR / "cropped_domes"
OUTPUT_PATH = BASE_DIR / "data" / "domes_medium_dataset_raw.csv"

def get_empty_features():
    """
    Создает словарь признаков с пустыми значениями (None).

    Используется для сохранения структуры датасета в случаях, когда файл 
    изображения не найден или его невозможно обработать. Исключает технические 
    колонки, чтобы не перезаписать метаданные из JSON.

    Returns:
        dict: Словарь, где ключами являются названия признаков из COLUMNS, 
              а значениями — None.
    """
    features = {}
    excluded_columns = ["file_name", "century", "target_shape"]

    for col in COLUMNS:
        if col not in excluded_columns:
            features[col] = None

    return features

def main():
    """
    Основной управляющий скрипт для сборки ML-датасета из сырых данных.

    Выполняет следующие шаги:
    1. Загружает базу данных из JSON-файла.
    2. Итерируется по записям, формируя пути к трем ожидаемым изображениям (1-3).
    3. При наличии файла извлекает геометрические признаки через DomesExtractor.
    4. Обрабатывает ошибки (отсутствие файла, ошибки чтения изображения).
    5. Формирует итоговый список словарей для конвертации в Pandas DataFrame.
    6. Выполняет экспорт финального датасета в формат CSV (RAW версия) для последующего 
       обучения моделей машинного обучения.
    """
    if not JSON_PATH.exists():
        print(f"Файл базы данных не найден: {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Инициализация списка для сбора данных и счетчиков статистики обработки
    all_data = []
    total_processed = 0
    with_data_count = 0

    # Итерация по списку объектов и их вариациям (изображениям 1-3)
    # для извлечения характеристик и формирования датасета.
    for item in data:
        for n in range(1, 4):
            file_name = f"cropped_{item.get('id')}_{n}.jpg"
            img_path = IMAGES_DIR / file_name
            
            # Формирование словаря из базовой информации для записи в датасет
            base_info = {
                "id": file_name,
                "century": item.get("century"),
                "target_shape": item.get("target_shape")
            }


            features = {}
            # Если файл существует, то..
            if img_path.exists():
                try:
                    # Инициализация экстрактора признаков для текущего изображения
                    extractor = DomesExtractor(str(img_path))
                    # Извлечение геометрических признаков из изображения
                    features = extractor.extract_all()
                    with_data_count += 1

                # Блок обработки ошибок
                except ValueError as e:
                    print(f"Ошибка данных: {img_path.name} не является валидным изображением. {e}")
                    features = get_empty_features()

                except Exception as e:
                    print(f"Ошибка при обработке файла {img_path.name}: {e}")
                    features = get_empty_features()
            
            # Присвоение пустых значений, если файл отсутствует
            else:
                features = get_empty_features()
            
            # Объединение базовой метаинформации и извлеченных геометрических признаков
            all_data.append({
                "file_name": file_name, 
                **base_info, 
                **features
            })
            total_processed += 1

    if all_data:
        # Создаем DataFrame с жестко заданным порядком столбцов (COLUMNS)
        df = pd.DataFrame(all_data, columns=COLUMNS)
        df.to_csv(
            OUTPUT_PATH, 
            index=False,          # Без индексов
            encoding='utf-8',     # Кодировка UTF-8
            sep=',',              # Разделитель - запятая
            decimal='.',          # Десятичный разделитель - точка
            na_rep='NA'           # Пропущенные значения как NA
        )
        print(f"Датасет готов! Сохранен: {OUTPUT_PATH}, всего строк: {total_processed}")
        print(f"Строк с реальными данными: {with_data_count}")
    else:
        print("Произошла ошибка при обработке.")

if __name__ == "__main__":
    main()