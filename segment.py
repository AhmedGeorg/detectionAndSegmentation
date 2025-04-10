import cv2
import numpy as np
import pandas as pd
import json
import os
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Константы
FIRE_LABEL = 1
SMOKE_LABEL = 2
BACKGROUND_LABEL = 0


# =============================================
# 1. Подготовка данных с проверкой изображений
# =============================================

def load_and_validate_data(annotations_path, images_dir):
    """Загрузка аннотаций и проверка существования изображений"""
    # Загрузка JSON с аннотациями
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Проверка структуры данных
    if 'images' not in data or not data['images']:
        raise ValueError("Некорректный формат аннотаций или отсутствуют изображения")

    # Создание DataFrame с проверкой существования файлов
    records = []
    for img_info in data['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        records.append({
            'image_path': img_path,
            'file_name': img_info['file_name'],
            'exists': os.path.exists(img_path)
        })

    df = pd.DataFrame(records)

    # Статистика
    print(f"\nВсего изображений в аннотациях: {len(df)}")
    print(f"Существует на диске: {df['exists'].sum()}")
    print(f"Отсутствует: {len(df) - df['exists'].sum()}")

    # Фильтрация только существующих файлов
    df = df[df['exists']].drop(columns=['exists'])
    return df


# Пути к данным (уберите пробел после .json если есть)
annotations_path = r"C:\Users\User\PycharmProjects\detectionAndSegmentation\475_file_train\annotations\instances_default.json"
images_dir = r"C:\Users\User\PycharmProjects\detectionAndSegmentation\475_file_train\images"

# Загрузка и проверка данных
df = load_and_validate_data(annotations_path, images_dir)

# Разделение на train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\nРазделение данных: Train={len(train_df)}, Val={len(val_df)}")


# =============================================
# 2. Создание и применение Labeling Functions
# =============================================

def safe_imread(image_path):
    """Безопасная загрузка изображения с обработкой ошибок"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        return img
    except Exception as e:
        print(f"Ошибка при загрузке {image_path}: {str(e)}")
        return None


@labeling_function()
def lf_fire_presence(row):
    img = safe_imread(row['image_path'])
    if img is None:
        return -1

    # Детекция красных/оранжевых оттенков
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Диапазоны для красного (2 диапазона из-за особенностей HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    fire_mask = cv2.bitwise_or(mask1, mask2)

    return FIRE_LABEL if np.sum(fire_mask) > 5000 else BACKGROUND_LABEL


@labeling_function()
def lf_smoke_presence(row):
    img = safe_imread(row['image_path'])
    if img is None:
        return -1

    # Анализ текстур для дыма
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Дым обычно дает мало четких границ
    return SMOKE_LABEL if np.sum(edges) < 3000 else BACKGROUND_LABEL


@labeling_function()
def lf_high_contrast(row):
    img = safe_imread(row['image_path'])
    if img is None:
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return FIRE_LABEL if contrast > 50 else BACKGROUND_LABEL


# Применение labeling functions
print("\nПрименение labeling functions к тренировочным данным...")
lfs = [lf_fire_presence, lf_smoke_presence, lf_high_contrast]
applier = PandasLFApplier(lfs)
L_train = applier.apply(train_df)

# =============================================
# 3. Обучение LabelModel и разметка данных
# =============================================

print("\nОбучение LabelModel...")
label_model = LabelModel()
label_model.fit(L_train)

# Добавление меток к данным
train_df = train_df.copy()
train_df['label'] = label_model.predict(L_train)

# Анализ результатов разметки
print("\nРаспределение меток в тренировочных данных:")
print(train_df['label'].value_counts())


# =============================================
# 4. Визуализация результатов разметки
# =============================================

def plot_examples(df, label, title, n=3):
    """Визуализация примеров с заданной меткой"""
    samples = df[df['label'] == label].sample(min(n, len(df)))
    if len(samples) == 0:
        print(f"\nНет примеров с меткой {label} ({title})")
        return

    print(f"\nПримеры {title} (метка {label}):")
    plt.figure(figsize=(15, 5))
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        img = cv2.cvtColor(safe_imread(row['image_path']), cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i)
        plt.imshow(img)
        plt.title(f"{title}\n{row['file_name']}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_examples(train_df, FIRE_LABEL, "Fire")
plot_examples(train_df, SMOKE_LABEL, "Smoke")


# =============================================
# 5. Подготовка данных для YOLO
# =============================================

def prepare_yolo_structure(df, output_dir, dataset_type):
    """Подготовка структуры данных для YOLO"""
    # Создание директорий
    images_dir = os.path.join(output_dir, 'images', dataset_type)
    labels_dir = os.path.join(output_dir, 'labels', dataset_type)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Обработка каждого изображения
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preparing {dataset_type}"):
        try:
            img = safe_imread(row['image_path'])
            if img is None:
                continue

            # Копирование изображения
            img_name = row['file_name']
            cv2.imwrite(os.path.join(images_dir, img_name), img)

            # Создание YOLO аннотации
            label_file = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join(labels_dir, label_file), 'w') as f:
                # Используем весь изображение как bbox (можно заменить на реальные аннотации)
                height, width = img.shape[:2]
                x_center, y_center = 0.5, 0.5
                bbox_width, bbox_height = 0.9, 0.9  # 90% от размера изображения
                f.write(f"{row['label'] - 1} {x_center} {y_center} {bbox_width} {bbox_height}\n")
        except Exception as e:
            print(f"Ошибка при обработке {row['image_path']}: {str(e)}")


# Подготовка train/val данных для YOLO
print("\nПодготовка данных для YOLO...")
prepare_yolo_structure(train_df, "yolo_dataset", "train")
prepare_yolo_structure(val_df, "yolo_dataset", "val")

# Создание dataset.yaml
yolo_config = """
path: ./yolo_dataset
train: images/train
val: images/val

names:
  0: fire
  1: smoke
"""

with open("yolo_dataset/dataset.yaml", "w") as f:
    f.write(yolo_config.strip())

# =============================================
# 6. Обучение YOLO модели
# =============================================

print("\nЗапуск обучения YOLO...")
model = YOLO("yolov8n.pt")  # Загрузка предобученной модели

try:
    results = model.train(
        data="yolo_dataset/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="0",  # Использовать GPU если доступен
        name="fire_smoke_detection",
        patience=10,  # Ранняя остановка если нет улучшений
        save=True,
        save_period=5,
        workers=4
    )
except Exception as e:
    print(f"Ошибка при обучении YOLO: {str(e)}")

# =============================================
# 7. Валидация и тестирование модели
# =============================================

print("\nЗагрузка лучшей обученной модели...")
best_model = YOLO("runs/detect/fire_smoke_detection/weights/best.pt")

# Валидация на тестовых данных
print("\nОценка модели на валидационных данных...")
metrics = best_model.val()
print(metrics)

# Тестирование на примерах
test_images = ["test1.jpg", "test2.jpg"]  # Замените на реальные пути
for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"\nТестовое изображение не найдено: {img_path}")
        continue

    print(f"\nОбработка тестового изображения: {img_path}")
    results = best_model(img_path)

    # Сохранение результатов
    for result in results:
        result.save(filename=f"results/{os.path.basename(img_path)}")

    # Визуализация
    for result in results:
        plt.figure(figsize=(10, 6))
        plt.imshow(result.plot()[:, :, ::-1])
        plt.title(f"Результаты детекции: {img_path}")
        plt.axis('off')
        plt.show()

print("\nОбучение и тестирование завершены!")