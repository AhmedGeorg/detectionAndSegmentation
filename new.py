import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
from joblib import dump

# Загрузка аннотаций
with open(
        'C:\\Users\\User\\PycharmProjects\\detectionAndSegmentation'
        '\\475_file_train\\annotations\\instances_default.json',
        "r") as f:
    data = json.load(f)

# Подготовка путей к изображениям
images = data["images"]
img_paths = [
    f"C:\\Users\\User\\PycharmProjects\\detectionAndSegmentation\\475_file_train\\images\\{img['file_name']}"
    for img in images
]

# Загрузка изображений и создание DataFrame
image_data = []
for path in img_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Не удалось загрузить изображение по пути: {path}")
        continue  # Пропускаем проблемные изображения

    img = np.array(img, dtype=np.uint8)
    image_data.append({"image": img, "path": path})

# Создаем единый DataFrame для всех изображений
if not image_data:
    raise ValueError("Не удалось загрузить ни одного изображения")

df = pd.DataFrame(image_data)


# Функции разметки
@labeling_function()
def lf_red_pixels(row):
    image = row["image"]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_pixels = np.sum((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170))
    return 1 if red_pixels > 1000 else 0


@labeling_function()
def lf_brightness(row):
    image = row["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return 1 if np.mean(gray) > 200 else 0


@labeling_function()
def lf_high_contrast(row):
    image = row["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return 1 if contrast > 50 else 0


# Применяем правила разметки
lfs = [lf_red_pixels, lf_brightness, lf_high_contrast]
applier = PandasLFApplier(lfs)
L_train = applier.apply(df)

# Обучение и сохранение модели только если есть данные
if len(L_train) > 0 and L_train.size > 0:
    try:
        label_model = LabelModel()
        label_model.fit(L_train)
        df["label"] = label_model.predict(L_train)

        # Сохраняем модель и результаты
        dump(label_model, 'model.joblib')
        df.to_csv('labeled_results.csv', index=False)
        print("Модель успешно обучена и сохранена")
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
else:
    print("Нет данных для обучения модели")

# for idx, row2 in df[df["label"] == 1].iterrows():  # label=1 → fire
#     img = cv2.imread(row2["image_path"])
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("Fire detected")
#     plt.show()
#
# for idx, row2 in df[df["label"] == 2].iterrows():  # label=2 → smoke
#     img = cv2.imread(row2["image_path"])
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("Smoke detected")
#     plt.show()
