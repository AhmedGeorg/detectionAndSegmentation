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

        allowed_classes = {1, 2}  # Какие классы оставляем (исходные COCO классы)

        with open(label_path, 'w') as f:
            height, width = img.shape[:2]
            for _, row in group.iterrows():
                original_class = row['category_id']

                # Пропускаем классы не из allowed_classes
                if original_class not in allowed_classes:
                    continue

                # Преобразуем классы 1 → 0, 2 → 1
                new_class = 0 if original_class == 1 else 1

                yolo_bbox = coco_to_yolo_bbox(row['bbox'], width, height)
                line = f"{new_class} {' '.join(map(str, yolo_bbox))}\n"
                f.write(line)


# Подготовка train и val данных
print("\nПодготовка train данных для YOLO...")
prepare_yolo_dataset(train_df, "yolo_dataset", "train")

print("\nПодготовка val данных для YOLO...")
prepare_yolo_dataset(val_df, "yolo_dataset", "val")

# Создаем dataset.yaml
yolo_config = f"""
train: C:/Users/User/PycharmProjects/detectionAndSegmentation/yolo_dataset/images/train
val: C:/Users/User/PycharmProjects/detectionAndSegmentation/yolo_dataset/images/val

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
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8n-seg.pt")

print("\nНачало обучения...")
results = [model1.train(
    data="yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cpu",  # Используем CPU
    name="fire_smoke_detection",
    patience=15,  # Ранняя остановка
    conf=0.1,
    save=True,
    save_period=10,
    workers=4,
    optimizer='AdamW',
    lr0=1e-40,
    augment=True,  # Аугментация данных
    mixup=0.2,  # Добавляем mixup аугментацию
    dropout=0.1,  # Регуляризация
    hsv_h=0.015,  # Цветовая аугментация
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,  # Вертикальное отражение
    fliplr=0.5   # Горизонтальное отражение
), model2.train(
    data="yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cpu",
    name="fire_smoke_segmentation",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mixup=0.1,
)]

# =============================================
# 4. Валидация и визуализация результатов
# =============================================

print("\nЗагрузка лучших обученных моделей...")
best_model1 = YOLO("runs/segment/fire_smoke_segmentation/weights/best.pt")
best_model2 = YOLO("runs/detect/fire_smoke_detection/weights/best.pt")

# Валидация на val данных
print("\nОценка модели на val данных...")
metrics = best_model1.val()
print(f"Precision: {metrics.box.mp}, Recall: {metrics.box.mr}, mAP50: {metrics.box.map75}")
metrics = best_model2.val()
print(f"Precision: {metrics.box.mp}, Recall: {metrics.box.mr}, mAP50: {metrics.box.map75}")


# Визуализация результатов на val изображениях
def visualize_predictions(model1, model2, val_df, num_samples=5):
    """Визуализация предсказаний на val данных"""
    sample_df = val_df.sample(min(num_samples, len(val_df)))

    for _, row in sample_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue

        # Предсказание
        results = [model1(img_path), model2(img_path)]

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
visualize_predictions(best_model1, best_model2, val_df, num_samples=5)

# Сохранение модели в ONNX формате
print("\nЭкспорт моделей в ONNX формат...")
best_model1.export(format="onnx")
best_model2.export(format="onnx")
print("Модель успешно экспортирована в ONNX формат")

print("\nОбучение и валидация завершены!")
