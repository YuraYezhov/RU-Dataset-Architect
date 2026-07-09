import cv2
from pathlib import Path
from domes_extractor import DomesExtractor

# Тестовые фото
test_files = [
    "cropped_19843_2.jpg",
    "cropped_18651_1.jpg",
    "cropped_16725_2.jpg",
    "cropped_05167_1.jpg",
    "cropped_00648_1.jpg"
]

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
        print("Контур отрисован!")

    cv2.imshow("cap", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

BASE_DIR = Path(__file__).resolve().parent.parent

# Итерация по списку тестовых файлов и запуск процесса обработки для каждого из них
for file_name in test_files:
    img_path = BASE_DIR / "cropped_domes" / file_name
    try:
        ext = DomesExtractor(str(img_path))
        print(f"\nФайл: {img_path.name}")
        print(f"Отношение высоты к ширине: {ext.get_aspect_ratio()}")
        print(f"Коэффициент выпуклости: {ext.get_solidity()}")
        print(f"Коэффициент заполнености: {ext.get_extent()}")
        print(f"Коэффициент остроты вершины: {ext.get_top_width_ratio()}")
        display_img(ext)

    except Exception as e:
        print(f"Ошибка при тесте: {e}")