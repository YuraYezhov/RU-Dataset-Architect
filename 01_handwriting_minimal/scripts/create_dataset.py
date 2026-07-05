import pandas as pd
from pathlib import Path
from handwriting_features import HandwritingExtractor

COLUMNS = ["file_name", "aspect_ratio", "ink_density", "components", "peaks", "solidity", "target_letters"]

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "data" / "labels.csv"
IMAGES_DIR  = BASE_DIR / "images"
OUTPUT_PATH = BASE_DIR / "data" / "handwriting_minimal_dataset.csv"

def main():
    df_raw = pd.read_csv(LABELS_PATH)
    label_map = {}
    
    for _, row in df_raw.iterrows():
        filename = str(row['Filename']).strip()
        word = str(row['Actual word']).strip()
        label_map[filename] = len(word)

    all_data = []

    for img_path in IMAGES_DIR.glob("*.png"):
        if img_path.name in label_map:
            try:
                extractor = HandwritingExtractor(img_path)

                all_data.append({
                    "file_name": img_path.name, 
                    "target_letters": label_map[img_path.name], 
                    **extractor.extract_all() 
                })
            
            except ValueError as e:
                print(f"Ошибка данных: {img_path.name} не является валидным изображением. {e}")
        
            except KeyError as e:
                print(f"Ошибка разметки: Файл {img_path.name} отсутствует в CSV-таблице.")
            
            except Exception as e:
                print(f"Ошибка при обработке файла {img_path.name}: {e}")
    
    if all_data:
        df = pd.DataFrame(all_data, columns=COLUMNS)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Датасет готов! Сохранен: {OUTPUT_PATH}, всего строк: {len(df)}")
    else:
        print("PNG файлы не найдены или произошла ошибка при обработке.")

if __name__ == "__main__":
    main()

