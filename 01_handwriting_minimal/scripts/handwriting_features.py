import cv2
import numpy as np
import pandas as pd
from pathlib import Path

class HandwritingExtractor:
    def __init__(self, image_path):
        """Загружает изображение и инициализирует базовые параметры."""
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.filename = self.filename = Path(image_path).name

    def get_aspect_ratio(self):
        """Вычисляет соотношение ширины к высоте слова."""
        pass

    def get_ink_density(self):
        """Вычисляет плотность - отношение закрашенных пикселей к общей площади."""
        pass

    def count_connected_components(self):
        """Подсчитывает количество связных фрагментов в слове."""
        pass

    def get_horizontal_peaks(self):
        """Определяет количество вертикальных штрихов через горизонтальную проекцию."""
        pass

    def get_solidity(self):
        """Вычисляет отношение площади контура к площади его выпуклой оболочки."""
        pass

    def extract_all(self) -> dict:
        """
        Запускает все методы извлечения признаков и возвращает результат в виде словаря.
        """
        features = {
            "aspect_ratio": self.get_aspect_ratio(),
            "ink_density": self.get_ink_density(),
            "connected_components": self.count_connected_components(),
            "horizontal_peaks": self.get_horizontal_peaks(),
            "solidity": self.get_solidity()
        }
        return features


columns = ["file_name", "aspect_ratio", "ink_density", "components", "peaks", "solidity", "target_letters"]

if __name__ == "__main__":
    path_to_images = Path("../images")
    all_data = []

    for file in path_to_images.glob("*.png"):
        try:
            extractor = HandwritingExtractor(file)
            features = extractor.extract_all()
            all_data.append(features)
        except Exception as e:
            print(f"Ошибка при обработке файла {file.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv("minimal_dataset.csv", index=False)
        print("Датасет готов! Сохранено строк:", len(df))
    else:
        print("PNG файлы не найдены или произошла ошибка при обработке.")