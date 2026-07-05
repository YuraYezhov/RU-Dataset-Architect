import cv2
from pathlib import Path
from scripts.handwriting_features import HandwritingExtractor

BASE_DIR = Path(__file__).parent 
img_path = BASE_DIR / "images" / "Image 140.png"

try:
    # Создаем экстрактор
    ext = HandwritingExtractor(img_path)
    
    # Вызываем функции и печатаем результат
    print(f"Файл: {img_path.name}")
    print(f"Соотношение сторон: {ext.get_aspect_ratio()}")
    print(f"Плотность чернил: {ext.get_ink_density()}")
    
    # Вызов окна с инверсией
    _, thresh = cv2.threshold(ext.image, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("What OpenCV sees", thresh)
    print("Нажми любую клавишу на картинке, чтобы закрыть...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Ошибка при тесте: {e}")