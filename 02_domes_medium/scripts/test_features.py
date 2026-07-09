import cv2
from pathlib import Path
from domes_extractor import DomesExtractor

# Инициализация словаря с идентификаторами изображений, разбитыми по категориям
data = {
    "difficult_case": ["16725_3", "05167_1", "00648_1"],
    "helmet":  ["00090_1", "00187_2", "00188_3"],
    "onion":   ["00152_1", "00199_2", "00981_1"],
    "tent":    ["00601_1", "01845_3", "05919_1"]
}

def display_img(ext):
    """
    Визуализирует исходное изображение с наложенным на него контуром.

    Функция создает окно, отрисовывает контур (если он существует) черной линией 
    толщиной 3 пикселя и ожидает нажатия любой клавиши для закрытия окна.

    Args:
        ext (object): Объект (экземпляр класса), который должен содержать атрибуты:
                      - ext.image: Исходное изображение (numpy array).
                      - ext.contour: Контур объекта (может быть None).
    """
    vis_img = ext.image.copy()

    # Проверяем, был ли найден контур, и если да, рисуем его на копии изображения
    if ext.contour is not None:
        cv2.drawContours(vis_img, [ext.contour], -1, (0, 0, 0), 3)
        print("Контур отрисован.")

    cv2.imshow("cap", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

BASE_DIR = Path(__file__).resolve().parent.parent

# Созадние файла для записи результатов обработки изображений
with open("results.txt", "a", encoding="utf-8") as f:
    for category, ids in data.items():
        f.write(f"\nКАТЕГОРИЯ: {category}\n")
        
        # Итерация по списку идентификаторов изображений в текущей категории
        for file_id in ids:
            file_name = f"cropped_{file_id}.jpg"
            img_path = BASE_DIR / "cropped_domes" / file_name

            # Запись вычисленных геометрических характеристик контура            
            try:
                ext = DomesExtractor(str(img_path))
                print(f"\nФайл: {img_path.name}")

                f.write(
                    f"Файл: {file_name}\n"
                    f"Отношение высоты к ширине: {ext.get_aspect_ratio()}\n"
                    f"Коэффициент выпуклости: {ext.get_solidity()}\n"
                    f"Коэффициент заполненности: {ext.get_extent()}\n"
                    f"Коэффициент остроты вершины: {ext.get_top_width_ratio()}\n"
                    f"Первый инвариантный момент Ху: {ext.get_hu_moment_1()}\n\n"
                )
                print("Данные записаны.")
                display_img(ext)

            except Exception as e:
                print(f"Ошибка при тесте: {e}")
                f.write(f"Ошибка в файле {file_name}: {e}\n\n")

print("\nВсе данные успешно записаны в results.txt")