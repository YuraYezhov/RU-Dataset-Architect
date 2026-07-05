import cv2
import numpy as np
from pathlib import Path

class HandwritingExtractor:
    def __init__(self, image_path):
        """Загружает изображение и инициализирует базовые параметры."""
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.filename = Path(image_path).name
        
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    def get_aspect_ratio(self):
        """
        Вычисляет соотношение ширины к высоте слова.
        Использует бинаризацию и boundingRect для нахождения границ чернил, 
        игнорируя пустые поля вокруг слова.
        
        Returns:
            float: Коэффициент соотношения сторон, округленный до 2 знаков.
        """
        # Бинаризация изображения (инверсия для выделения текста)
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Поиск ненулевых пикселей (текста)
        coords = cv2.findNonZero(binary)
        
        if coords is not None:
            # Получение ограничивающего прямоугольника
            _, _, w, h = cv2.boundingRect(coords)
            
            # Вычисление и округление соотношения сторон
            if h > 0: return round(w / h, 2)
            
        return 0

    def get_ink_density(self):
        """
        Вычисляет плотность заполнения области слова чернилами.
        Определяется как отношение количества пикселей чернил к 
        площади описывающего прямоугольника (bounding box).
        
        Returns:
            float: Плотность (от 0 до 1), округленная до 4 знаков.
        
        """
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)

        if coords is not None:
            _, _, w, h = cv2.boundingRect(coords)

            # Подсчет количества пикселей текста
            ink_pixels = cv2.countNonZero(binary)

            # Вычисление площади ограничивающего прямоугольника
            area_rect = w * h

            # Расчет плотности (отношение чернил к площади)
            if area_rect > 0:
                return round(ink_pixels / area_rect, 4)

        return 0

    def count_connected_components(self):
        """
        Подсчитывает количество связных компонентов.
        В рукописном тексте это могут быть отдельные буквы, знаки препинания
        или не соединенные между собой штрихи.

        Returns:
            int: Количество найденных компонентов (чернильных фрагментов).
        """
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)

        # Поиск связных компонентов
        # retval — это целое число (количество найденных областей)
        # _ — это матрица того же размера, что и фото, где каждый пиксель 
        # имеет номер своей области (нам она здесь не нужна)
        retval, _ = cv2.connectedComponents(binary)

        return max(0, retval - 1)

    def get_horizontal_peaks(self):
        """Определяет количество вертикальных штрихов через горизонтальную проекцию."""
        pass

    def get_solidity(self):
        """
        Вычисляет плотность заполнения слова.
        Определяется как отношение площади чернильных пикселей к площади 
        выпуклой оболочки (Convex Hull), охватывающей всё слово.
        
        Помогает отличить компактные слова от слов с выступающими 
        элементами (хвостами букв 'у', 'д', 'б').

        Returns:
            float: Коэффициент плотности (от 0 до 1), округленный до 4 знаков.
        """
        _, binary = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(binary)

        if coords is not None:
            ink_area = cv2.countNonZero(binary)
            hull = cv2.convexHull(coords)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0:
                return round(ink_area / hull_area, 4)
        return 0

    def extract_all(self):
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