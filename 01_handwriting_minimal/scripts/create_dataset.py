import pandas as pd
from pathlib import Path
from handwriting_features import HandwritingExtractor

COLUMNS = ["file_name", "aspect_ratio", "ink_density", "components", "peaks", "solidity", "target_letters"]

def main():
    df_raw = pd.read_csv("../data/labels.csv")
    label_map = {}
    
    for _, row in df_raw.iterrows():
        filename = str(row['Filename']).strip()
        word = str(row['Actual word']).strip()
        label_map[filename] = len(word)

    path_to_images = Path("../images")
    all_data = []

    for img_path in path_to_images.glob("*.png"):
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
        df.to_csv("hangwriting_minimal_dataset.csv", index=False)
        print("Датасет готов! Сохранено строк:", len(df))
    else:
        print("PNG файлы не найдены или произошла ошибка при обработке.")

if __name__ == "__main__":
    main()

