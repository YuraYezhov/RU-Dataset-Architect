import json
import time
import requests
from pathlib import Path

# Конфигурация путей
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_FILE = PROJECT_ROOT / "02_domes_medium" / "data" / "intermediate_data" / "all_domes.json"
SAVE_DIR = PROJECT_ROOT / "02_domes_medium" / "images"

# Headers для имитации браузера
HEADERS = {
    'User-Agent': 'YOUR AGENT'
}

def download_images():
    """
    Считывает список объектов из JSON-файла, извлекает ссылки на изображения
    и сохраняет их в директорию SAVE_DIR с именами формата {id}_{index}.jpg.
    """
    # Создаем директорию для изображений, если она не существует
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Загружаем данные из JSON
    if not DATA_FILE.exists():
        print(f"Ошибка: Файл {DATA_FILE} не найден.")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Проходимся по каждому объекту
    for item in data:
        item_id = item.get("id")
        photos = item.get("photos", [])

    # Пропускаем объекты без ID или списка фотографий
        if not item_id or not photos:
            continue

        for index, photo_url in enumerate(photos):
            # Пропускаем null (нет ссылки на изображение)
            if not photo_url:
                continue

            # Формируем имя файла на основе ID объекта и порядкового номера фото
            file_name = f"{item_id}_{index + 1}.jpg"
            file_path = SAVE_DIR / file_name

            print(f"Загрузка {item_id} ({index + 1}/3)...")

            # Выполняем запрос к серверу и сохраняем изображение в файл
            try:
                response = requests.get(photo_url, headers=HEADERS, timeout=10)
                response.raise_for_status()

                with open(file_path, 'wb') as f:
                    f.write(response.content)

            except Exception as e:
                print(f"Ошибка при загрузке {photo_url}: {e}")
                continue
            
            # Пауза между загрузками
            time.sleep(2)

    print("Процесс загрузки завершен.")

if __name__ == "__main__":
    download_images()