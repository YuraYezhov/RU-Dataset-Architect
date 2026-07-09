import cv2
import sys
import os
import numpy as np
from pathlib import Path

class DomesExtractor:
    def __init__(self, image_path):
        """
        Инициализирует объект класса для анализа изображений куполов.
        Загружает изображение в оттенках серого и проверяет его валидность.

        Args:
            image_path (Path/str): Полный путь к файлу изображения.

        Raises:
            ValueError: Если файл не удалось прочитать (неверный путь или поврежденный файл).
        """
        # Чтение изображения в оттенках серого
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Сохранение имени файла для идентификации
        self.filename = Path(image_path).name
        
        # Проверка корректности загрузки файла
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        self.gray = self.image
        self.contour = self._get_main_contour()

    def _get_main_contour(self):
        """
        Выделяет основной контур объекта на изображении с использованием бинаризации и аппроксимации.

        Логика работы:
        1. Применяет размытие Гаусса для удаления шумов.
        2. Выполняет бинаризацию по методу Оцу.
        3. Ищет все внешние контуры.
        4. Выбирает контур с максимальной площадью.
        5. Вычисляет выпуклую оболочку (Convex Hull) для сглаживания.
        6. Аппроксимирует форму полигоном (ApproxPolyDP) для упрощения геометрии.

        Returns:
            numpy.ndarray or None: Массив точек аппроксимированного контура, если контур найден, иначе None.
        """
        # Размытие
        img_blurred = cv2.GaussianBlur(self.gray, (5, 5), 0)
        
        # Бинаризация
        _, img_binary = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Поиск контуров
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Выбор самого крупного
        if contours:
            raw_contour = max(contours, key=cv2.contourArea)
            
            # Сглаживаем форму выпуклой оболочкой
            hull = cv2.convexHull(raw_contour)
            
            # Упрощаем форму
            # 0.01 - это точность, изменяемая
            epsilon = 0.01 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            return approx
        return None

    def get_aspect_ratio(self):
        """
        Вычисляет коэффициент формы для найденного контура.

        Коэффициент определяется как отношение ширины к высоте описывающего прямоугольника 
        (Bounding Rect). Позволяет судить о вытянутости объекта.

        Returns:
            float: Значение отношения Width / Height, округленное до 2 знаков.
                Возвращает 0, если контур не задан или высота равна нулю.
        """
        if self.contour is None:
            return 0
        
        # Получение ограничивающего прямоугольника
        _, _, w, h = cv2.boundingRect(self.contour)

        if h > 0:
            return round(w / h, 2)
        return 0

    def get_solidity(self):
        """
        Вычисляет плотность ().
        Вычисляет плотность (solidity) контура.

        Плотность — это отношение площади контура (Area) к площади его выпуклой оболочки (Convex Hull).
        Значение близкое к 1.0 указывает на выпуклый, плотный объект без глубоких выемок.

        Returns:
            float: Значение плотности, округленное до 2 знаков.
                Возвращает 0, если контур не задан или площадь оболочки равна нулю.
        """
        if self.contour is None:
            return 0
        
        # Вычисление площади контура
        area = cv2.contourArea(self.contour)

        # Вычисление площади выпуклой оболочки
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            return round(area / hull_area, 2)
        return 0

    def get_extent(self):
        """
        Вычисляет заполненность (Area / BoundingRectArea).
        """
        pass

    def get_top_width_ratio(self):
        """
        Показывает, насколько 'острая' у купола вершина.
        """
        pass
    
    def get_hu_moment(self):
        """
        Первый инвариантный момент Ху. 
        """
        pass
    
    def extract_all(self):
        """
        Запускает комплексный анализ изображения всеми доступными методами
        и агрегирует в единый словарь.

        Returns:
            dict: Набор вычисленных признаков.
        """
        features = {
            "aspect_ratio": self.get_aspect_ratio(),
            "solidity": self.get_solidity(),
            "extent": self.get_extent(),
            "top_width_ratio": self.get_top_width_ratio(),
            "hu_moment_1": self.get_hu_moment()
        }
        return features
