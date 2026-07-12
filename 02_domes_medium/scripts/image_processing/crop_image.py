import cv2
from pathlib import Path

# Настройка путей к директориям проекта
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = BASE_DIR / "images"
SAVE_DIR = BASE_DIR / "cropped_domes" 

# Глобальные переменные
drawing = False
ix, iy = -1, -1
img_display = None
img_org = None
current_filename = ""

# Функция обработки событий мыши для выделения и обрезки области
def crop_mouse(event, x, y, flags, param):
    """
    Обработчик событий мыши для рисования рамки выделения и обрезки изображения.

    Функция отслеживает три состояния: нажатие левой кнопки (начало выделения), 
    движение мыши и отпускание кнопки (фиксация области 
    и автоматическое сохранение вырезанного фрагмента).

    Args:
        event (int): Тип события мыши OpenCV.
        x (int): Текущая координата мыши по горизонтали.
        y (int): Текущая координата мыши по вертикали.
        flags (int): Дополнительные флаги событий мыши.
        param (any): Дополнительные параметры, передаваемые в callback (не используются).

    Returns:
        None: Функция выполняет сохранение файла на диск и обновление глобальных переменных.
    """
    global ix, iy, drawing, img_display, img_org

    # Обработка событий нажатия, перемещения и отпускания кнопки мыши
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # Отрисовка прямоугольника при движении мыши
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Копируем чистый имидж, чтобы старые рамки стирались при движении мыши
            img_display = img_org.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)

    # Завершение рисования при отпускании кнопки мыши
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # Финальная отрисовка
        cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)

        # Координаты для обрезки
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        # Вырезаем из ОРИГИНАЛА (без зеленой рамки)
        roi = img_org[y1:y2, x1:x2]
        
        if roi.size > 0:
            # Формируем уникальное имя (добавляем таймштамп или индекс, если нужно много куполов с одного фото)
            save_name = f"cropped_{current_filename}"
            save_path = SAVE_DIR / save_name
            cv2.imwrite(str(save_path), roi)
            print(f"--- Сохранено в cropped_images: {save_name}")

def main():
    """
    Основной управляющий цикл программы для ручной обрезки куполов.

    Функция выполняет следующие шаги:
    1. Инициализирует директории для поиска исходных фото и сохранения результатов.
    2. Фильтрует файлы в исходной папке по расширениям (jpg, jpeg).
    3. Создает графическое окно OpenCV и привязывает к нему обработчик мыши.
    4. В цикле последовательно отображает каждое изображение и ожидает действий пользователя.
    5. Обрабатывает нажатия клавиш: ПРОБЕЛ для перехода к следующему фото, ESC для выхода.

    Returns:
        None: Завершает работу после обработки всех файлов в списке или принудительного выхода.
    """
    global img_display, img_org, current_filename
    
    # Создание директории для сохранения результатов, если она не существует
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Получение списка всех изображений в директории SOURCE_DIR
    files = [f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg'}]
    
    if not files:
        print(f"В папке {SOURCE_DIR} нет подходящих фото!")
        return

    cv2.namedWindow('Crop Window')
    cv2.setMouseCallback('Crop Window', crop_mouse)
    
    # Вывод инструкций по управлению в консоль
    print("\nИНСТРУКЦИЯ:")
    print("1. Зажмите ЛКМ и выделите купол.")
    print("2. Отпустите — купол сохранится автоматически.")
    print("3. ПРОБЕЛ — следующее фото.")
    print("4. ESC — выход из программы.")

    # Цикл обхода файлов изображений для ручной обработки
    for file_path in files:
        current_filename = file_path.name
        img_org = cv2.imread(str(file_path))
        
        if img_org is None:
            continue
        
        img_display = img_org.copy()
        
        # Отображение изображения в окне для взаимодействия
        while True:
            cv2.imshow('Crop Window', img_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:
                break
            if key == 27:
                cv2.destroyAllWindows()
                return
                
    cv2.destroyAllWindows()
    print("\nВсё! Теперь у тебя есть папка 'cropped_images' с чистыми куполами.")

if __name__ == "__main__":
    main()