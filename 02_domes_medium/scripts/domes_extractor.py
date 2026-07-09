import cv2
import numpy as np
import math
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
        self.gray = self.image

        # Сохранение имени файла для идентификации
        self.filename = Path(image_path).name
        
        # Проверка корректности загрузки файла
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

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
        Вычисляет заполненность (extent).

        Заполненность — это отношение площади контура (Area) к площади его ограничивающего 
        прямоугольника (Bounding Box). Позволяет оценить, насколько объект 
        близок по форме к прямоугольнику.

        Returns:
            float: Значение заполненности (Area / BoundingRectArea), 
                округленное до 2 знаков. Возвращает 0, если контур не задан 
                или площадь прямоугольника равна нулю.
        """
        if self.contour is None:
            return 0
        
        area = cv2.contourArea(self.contour)

        # Вычисление площади ограничивающего прямоугольника
        x, y, w, h = cv2.boundingRect(self.contour)
        rect_area = w * h

        if rect_area > 0:
            return round(area / rect_area, 2)
        
        return 0

    def get_top_width_ratio(self):
        """
        Вычисляет коэффициент ширины вершины (Top Width Ratio).

        Показывает, насколько "острой" или "плоской" является верхушка купола. 
        Коэффициент определяется как отношение ширины купола на уровне 10% от 
        его верхней точки к его полной ширине.

        Логика работы:
        1. Определяет ограничивающий прямоугольник.
        2. Выбирает горизонтальную линию на уровне 10% высоты вниз от вершины.
        3. Создает временную маску и измеряет ширину контура в пикселях на этой линии.
        4. Нормирует полученную ширину, деля её на полную ширину (w) прямоугольника.

        Returns:
            float: Коэффициент "остроты" вершины (от 0 до 1), округленный до 2 знаков.
                Возвращает 0, если контур не задан или на данном уровне объект не обнаружен.
        """
        if self.contour is None:
            return 0

        _, y, w, h = cv2.boundingRect(self.contour)
        
        # Определяем "уровень" (10% от верхушки вниз)
        target_y = y + int(h * 0.1)
        
        #  Создаем черную маску размером с картинку
        mask = np.zeros(self.gray.shape, dtype=np.uint8)

        # Рисуем на маске купол белым цветом и заполненным (-1)
        cv2.drawContours(mask, [self.contour], -1, 255, -1)

        # Берем срез маски по строке target_y
        y_row = mask[target_y, :]

        # Находим индексы всех белых пикселей (255) в этой строке
        white_pixels = np.where(y_row == 255)[0]

        if len(white_pixels) > 0:
            # Вычисляем ширину контура как разницу между крайними точками
            top_width = np.max(white_pixels) - np.min(white_pixels)
            return round(top_width / w, 2)
        
        return 0

    def get_hu_moment_1(self):
        """
        Вычисляет первый инвариантный момент Ху с логарифмическим масштабированием.

        Логика работы:
        1. Рассчитывает центральные моменты контура.
        2. Извлекает первый инвариантный момент Ху.
        3. Применяет логарифмическое масштабирование: -1 * sign(h1) * log10(abs(h1)).
        Это необходимо, так как сырые значения моментов обычно очень малы (например, 10^-6), 
        а логарифм приводит их к удобному для анализа виду (например, в диапазон 1-10).

        Returns:
            float: Масштабированное значение первого момента Ху, округленное до 4 знаков.
                Возвращает 0, если контур не задан или значение момента равно нулю.
        """
        if self.contour is None:
            return 0
        
        # Считаем простые моменты
        M = cv2.moments(self.contour)
        
        # Считаем моменты Ху
        hu_moments = cv2.HuMoments(M)

        # Берем первый момент
        h1 = hu_moments[0][0]
        
        # Логарифмическое масштабирование значения момента для улучшения читаемости
        if h1 != 0:
            h1 = -1.0 * math.copysign(1.0, h1) * math.log10(abs(h1))
            return round(h1, 4)
        
        return 0
    
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
            "hu_moment_1": self.get_hu_moment_1()
        }
        return features
