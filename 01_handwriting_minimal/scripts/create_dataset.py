import pandas as pd
from pathlib import Path
from handwriting_features import HandwritingExtractor

COLUMNS = ["file_name", "aspect_ratio", "ink_density", "components", "peaks", "solidity", "target_letters"]

# Настройка путей к данным и директориям проекта
BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "labels.csv"
IMAGES_DIR  = BASE_DIR / "images"
OUTPUT_PATH = BASE_DIR / "data" / "handwriting_minimal_dataset.csv"

def main():
    """
    Основной управляющий скрипт для сборки ML-датасета из сырых данных.

    Функция координирует полный цикл подготовки данных:
    1. Загружает исходную разметку и вычисляет целевую переменную (длину слов).
    2. Сканирует директорию с изображениями и фильтрует их по наличию в разметке.
    3. Использует класс HandwritingExtractor для извлечения 5 геометрических 
       и плотностных признаков через алгоритмы OpenCV.
    4. Агрегирует полученную информацию в структурированную таблицу (DataFrame).
    5. Выполняет экспорт финального датасета в формат CSV для последующего 
       обучения моделей машинного обучения.
    """
    # Загрузка исходной разметки (CSV-файл с именами файлов и словами)
    df_raw = pd.read_csv(LABELS_PATH)
    
    # Формирование словаря {Имя_файла: Длина_слова} для мгновенного поиска
    label_map = {}
    for _, row in df_raw.iterrows():
        filename = str(row['Filename']).strip()
        word = str(row['Actual word']).strip()
        label_map[filename] = len(word)

    all_data = []

    # Итерация по изображениям в директории
    for img_path in IMAGES_DIR.glob("*.png"):
        # Проверка наличия файла в разметке и извлечение признаков
        if img_path.name in label_map:
            try:
                # Инициализация экстрактора признаков для текущего изображения
                extractor = HandwritingExtractor(img_path)

                # Извлечение признаков и добавление их в список данных
                all_data.append({
                    "file_name": img_path.name, 
                    "target_letters": label_map[img_path.name], 
                    **extractor.extract_all() 
                })
            
            # Блок обработки ошибок
            except ValueError as e:
                print(f"Ошибка данных: {img_path.name} не является валидным изображением. {e}")
        
            except KeyError as e:
                print(f"Ошибка разметки: Файл {img_path.name} отсутствует в CSV-таблице.")
            
            except Exception as e:
                print(f"Ошибка при обработке файла {img_path.name}: {e}")
    
    # Сохранение собранных данных в CSV-файл
    if all_data:
        # Создаем DataFrame с жестко заданным порядком столбцов (COLUMNS)
        df = pd.DataFrame(all_data, columns=COLUMNS)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Датасет готов! Сохранен: {OUTPUT_PATH}, всего строк: {len(df)}")
    else:
        print("PNG файлы не найдены или произошла ошибка при обработке.")

if __name__ == "__main__":
    main()