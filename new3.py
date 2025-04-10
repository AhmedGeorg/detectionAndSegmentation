import os
import json
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# Конфигурация путей
TRAIN_ANNOTATIONS = r"C:\Users\User\PycharmProjects" \
                    r"\detectionAndSegmentation\475_file_train\annotations\instances_default.json"
TRAIN_IMAGES_DIR = r"C:\Users\User\PycharmProjects\detectionAndSegmentation\475_file_train\images"
VAL_ANNOTATIONS = r"C:\Users\User\PycharmProjects" \
                  r"\detectionAndSegmentation\474_fire_val\annotations\instances_default.json"
VAL_IMAGES_DIR = r"C:\Users\User\PycharmProjects\detectionAndSegmentation\474_fire_val\images"

# Создаем директории для YOLO
os.makedirs("yolo_dataset/images/train", exist_ok=True)
os.makedirs("yolo_dataset/labels/train", exist_ok=True)
os.makedirs("yolo_dataset/images/val", exist_ok=True)
os.makedirs("yolo_dataset/labels/val", exist_ok=True)


# =============================================
# 1. Функции для обработки аннотаций COCO
# =============================================

def load_coco_annotations(annotation_path, images_dir):
    """Загрузка COCO аннотаций с проверкой структуры данных"""
    with open(annotation_path) as f:
        data = json.load(f)

    # Проверяем наличие обязательных ключей
    if 'annotations' not in data or 'images' not in data or 'categories' not in data:
        raise ValueError("Invalid COCO format: missing required fields")

    # Создаем маппинги
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    annotations = []
    for ann in data['annotations']:
        # Проверяем наличие обязательных полей
        if 'image_id' not in ann or 'bbox' not in ann or 'category_id' not in ann:
            print(f"Skipping invalid annotation: {ann}")
            continue

        img_info = images.get(ann['image_id'])
        if not img_info:
            print(f"Image ID {ann['image_id']} not found in images")
            continue

        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # COCO bbox: [x, y, width, height]
        bbox = ann['bbox']
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            print(f"Invalid bbox in annotation: {ann}")
            continue

        annotations.append({
            'image_id': ann['image_id'],
            'image_path': img_path,
            'file_name': img_info['file_name'],
            'category_id': ann['category_id'],
            'category_name': categories.get(ann['category_id'], 'unknown'),
            'bbox': bbox,
            'segmentation': ann.get('segmentation', []),
            'area': ann.get('area', 0),
            'iscrowd': ann.get('iscrowd', 0)
        })

    if not annotations:
        raise ValueError("No valid annotations found!")

    return pd.DataFrame(annotations)


print("\nЗагрузка train данных...")
train_df = load_coco_annotations(TRAIN_ANNOTATIONS, TRAIN_IMAGES_DIR)
print(f"Загружено {len(train_df)} train аннотаций")
print("Столбцы в train_df:", train_df.columns.tolist())  # Проверка столбцов

print("\nЗагрузка val данных...")
val_df = load_coco_annotations(VAL_ANNOTATIONS, VAL_IMAGES_DIR)
print(f"Загружено {len(val_df)} val аннотаций")
print("Столбцы в val_df:", val_df.columns.tolist())  # Проверка столбцов


# =============================================
# 2. Подготовка данных в формате YOLO
# =============================================

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """Конвертация COCO bbox в YOLO формат"""
    x_center = (bbox[0] + bbox[2] / 2) / img_width
    y_center = (bbox[1] + bbox[3] / 2) / img_height
    width = bbox[2] / img_width
    height = bbox[3] / img_height
    return [x_center, y_center, width, height]


def prepare_yolo_dataset(df, output_dir, dataset_type):
    """Подготовка данных для YOLO с корректной обработкой путей"""
    images_dir = os.path.join(output_dir, 'images', dataset_type)
    labels_dir = os.path.join(output_dir, 'labels', dataset_type)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Группируем аннотации по изображениям
    grouped = df.groupby('image_id')

    for image_id, group in tqdm(grouped, desc=f"Preparing {dataset_type}"):
        first_row = group.iloc[0]
        img_path = first_row['image_path']

        # Загружаем изображение
        img = cv2.imread(img_path)
        if img is None:
            print(f"Image not found or corrupted: {img_path}")
            continue

        # Извлекаем имя файла без пути (на случай, если в file_name есть поддиректории)
        img_name = os.path.basename(first_row['file_name'])

        # Копируем изображение
        output_img_path = os.path.join(images_dir, img_name)
        cv2.imwrite(output_img_path, img)

        # Создаем файл с аннотациями (убедимся, что путь корректен)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, 'w') as f:
            height, width = img.shape[:2]
            for _, row in group.iterrows():
                # Конвертируем bbox в YOLO формат
                yolo_bbox = coco_to_yolo_bbox(row['bbox'], width, height)
                # YOLO использует индексы классов с 0
                line = f"{row['category_id'] - 1} {' '.join(map(str, yolo_bbox))}\n"
                f.write(line)


# Подготовка train и val данных
print("\nПодготовка train данных для YOLO...")
prepare_yolo_dataset(train_df, "yolo_dataset", "train")

print("\nПодготовка val данных для YOLO...")
prepare_yolo_dataset(val_df, "yolo_dataset", "val")

# Создаем dataset.yaml
yolo_config = f"""
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
# 3. Обучение YOLO модели для детекции и сегментации
# =============================================

print("\nИнициализация YOLO модели...")
model = YOLO("yolov8n-seg.pt")  # Модель для сегментации

print("\nНачало обучения...")
results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cpu",  # Используем GPU
    name="fire_smoke_segmentation",
    patience=15,  # Ранняя остановка
    save=True,
    save_period=10,
    workers=4,
    optimizer='AdamW',
    lr0=0.001,
    augment=True  # Аугментация данных
)

# =============================================
# 4. Валидация и визуализация результатов
# =============================================

print("\nЗагрузка лучшей обученной модели...")
best_model = YOLO("runs/segment/fire_smoke_segmentation/weights/best.pt")

# Валидация на val данных
print("\nОценка модели на val данных...")
metrics = best_model.val()
print(metrics)


# Визуализация результатов на val изображениях
def visualize_predictions(model, val_df, num_samples=5):
    """Визуализация предсказаний на val данных"""
    sample_df = val_df.sample(min(num_samples, len(val_df)))

    for _, row in sample_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue

        # Предсказание
        results = model(img_path)

        # Визуализация
        plt.figure(figsize=(15, 8))

        # Оригинальное изображение с аннотациями
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Рисуем ground truth
        gt_img = img.copy()
        bbox = row['bbox']
        cv2.rectangle(gt_img,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                      (0, 255, 0), 2)
        plt.subplot(1, 2, 1)
        plt.imshow(gt_img)
        plt.title(f"Ground Truth\n{row['file_name']}\nClass: {row['category_name']}")
        plt.axis('off')

        # Рисуем предсказания
        plt.subplot(1, 2, 2)
        plotted_img = results[0].plot()
        plt.imshow(plotted_img[:, :, ::-1])
        plt.title("Model Predictions")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


print("\nВизуализация результатов на val данных...")
visualize_predictions(best_model, val_df, num_samples=5)

# Сохранение модели в ONNX формате
print("\nЭкспорт модели в ONNX формат...")
best_model.export(format="onnx")
print("Модель успешно экспортирована в ONNX формат")

print("\nОбучение и валидация завершены!")
