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
import sklearn.metrics.precision_score
import sklearn.metrics.recall_score


# =============================================
# 1. Подготовка данных
# =============================================

# Загрузка путей к изображениям
def load_image_paths(annotations_path, images_dir):
    with open(annotations_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        raise ValueError("No images found in annotations")

    return [os.path.join(images_dir, img["file_name"]) for img in images]


# Пример использования:
annotations_path = r"C:\Users\User\PycharmProjects\detectionAndSegmentation\475_file_train\annotations" \
                   r"\instances_default.json "
images_dir = r"C:\Users\User\PycharmProjects\detectionAndSegmentation\475_file_train\images"
image_paths = load_image_paths(annotations_path, images_dir)

# Создаем DataFrame
df = pd.DataFrame({"image_path": image_paths})

# 1. После создания df добавим проверку существования файлов
df["exists"] = df["image_path"].apply(lambda x: os.path.exists(x))
print(f"Всего изображений: {len(df)}, Существует: {df['exists'].sum()}")

# Удаляем несуществующие файлы
df = df[df["exists"]].drop(columns=["exists"])

# 2. Разделяем данные ДО применения LF
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 6. Для val_df можно использовать другую стратегию (например, ручную разметку)
val_df["label"] = 0  # По умолчанию

# 7. Объединяем обратно для YOLO (если нужно)
df_labeled = pd.concat([train_df, val_df])


# =============================================
# 2. Создание Labeling Functions для fire/smoke
# =============================================

@labeling_function()
def lf_fire_presence(row):
    img = cv2.imread(row["image_path"])
    if img is None:
        return -1

    # Анализ цветов (красный/оранжевый)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    return 1 if np.sum(mask) > 5000 else 0  # 1 = fire, 0 = нет


@labeling_function()
def lf_smoke_presence(row):
    img = cv2.imread(row["image_path"])
    if img is None:
        return -1

    # Анализ текстур (размытость, низкий контраст)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blur, 30, 100)

    return 2 if np.sum(edges) < 3000 else 0  # 2 = smoke, 0 = нет


@labeling_function()
def lf_high_contrast(row):
    img = cv2.imread(row["image_path"])
    if img is None:
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return 1 if contrast > 50 else 0


# =============================================
# 3. Применение Labeling Functions
# =============================================

lfs = [lf_fire_presence, lf_smoke_presence, lf_high_contrast]
applier = PandasLFApplier(lfs)
L_train = applier.apply(df)

# 3. Применяем LF только к train_df
lfs = [lf_fire_presence, lf_smoke_presence, lf_high_contrast]
applier = PandasLFApplier(lfs)
L_train = applier.apply(train_df)

# 4. Обучаем LabelModel
label_model = LabelModel()
label_model.fit(L_train)

# 5. Добавляем метки только к train_df
train_df = train_df.copy()
train_df["label"] = label_model.predict(L_train)

# =============================================
# 4. Обучение LabelModel
# =============================================

label_model = LabelModel()
label_model.fit(L_train)
df["label"] = label_model.predict(L_train)


# Визуализация результатов
def show_examples(df, label, title):
    sample = df[df["label"] == label].sample(3)
    for _, row in sample.iterrows():
        img = cv2.imread(row["image_path"])
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{title} (Label: {label})")
        plt.axis("off")
        plt.show()


show_examples(df, 1, "Fire")
show_examples(df, 2, "Smoke")

# =============================================
# 5. Подготовка данных для YOLO
# =============================================

# Создаем директории для YOLO
os.makedirs("yolo_dataset/images/train", exist_ok=True)
os.makedirs("yolo_dataset/labels/train", exist_ok=True)

# Классы для YOLO
classes = {1: "fire", 2: "smoke"}

# Создаем аннотации в формате YOLO
for idx, row in df.iterrows():
    if row["label"] in classes:
        # Копируем изображение
        img = cv2.imread(row["image_path"])
        img_name = os.path.basename(row["image_path"])
        cv2.imwrite(f"yolo_dataset/images/train/{img_name}", img)

        # Создаем файл с аннотацией
        label_file = os.path.splitext(img_name)[0] + ".txt"
        with open(f"yolo_dataset/labels/train/{label_file}", "w") as f:
            # Формат YOLO: class_id x_center y_center width height
            # Здесь используем всю область изображения как bounding box
            f.write(f"{row['label'] - 1} 0.5 0.5 1.0 1.0\n")

# Создаем конфиг для YOLO
with open("yolo_dataset/dataset.yaml", "w") as f:
    f.write(f"""
path: ../yolo_dataset
train: images/train
val: images/train  # Используем те же данные для примера

names:
  0: fire
  1: smoke
""")

# =============================================
# 6. Обучение YOLO
# =============================================

model = YOLO("yolov8n.pt")  # Загружаем предобученную модель
results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="0"  # Используем GPU, если доступен
)

# =============================================
# 7. Тестирование модели
# =============================================

# Загружаем лучшую обученную модель
best_model = YOLO("runs/detect/train/weights/best.pt")

# Тестируем на новых изображениях
test_images = ["test1.jpg", "test2.jpg"]
for img_path in test_images:
    results = best_model(img_path)
    for result in results:
        result.show()  # Показываем результат с bounding boxes
        result.save(filename=f"result_{img_path}")  # Сохраняем результат

# Выводим примеры с fire/smoke
for idx, row in df[df["label"] == 1].iterrows():  # label=1 → fire
    img = cv2.imread(row["image_path"])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Fire detected")
    plt.show()

for idx, row in df[df["label"] == 2].iterrows():  # label=2 → smoke
    img = cv2.imread(row["image_path"])
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Smoke detected")
    plt.show()
