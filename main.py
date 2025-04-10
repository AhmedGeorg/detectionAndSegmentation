import cv2
import numpy as np
import pandas as pd
import json
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import sklearn.metrics.precision_score
import sklearn.metrics.recall_score

with open('C:\\Users\\User\\PycharmProjects\\detectionAndSegmentation\\475_file_train\\annotations'
          '\\instances_default.json', "r") as f:
    data = json.load(f)

images = data["images"]
img_path = []

for i in images:
    img_path.append(f"C:\\Users\\User\\PycharmProjects"
                    f"\\detectionAndSegmentation\\475_file_train\\images\\{i['file_name']}")

for i in img_path:
    img = cv2.imread(i)
    if img is None:
        print(f"Не удалось загрузить изображение по пути: {i}")
        exit()
        img = np.array(img, dtype=np.uint8)
        # Создаем DataFrame с изображениями (а не с метками)
        df = pd.DataFrame({"image": [img]})

# # Загрузка изображения
# img_path = "C:\\Users\\User\\PycharmProjects\\detectionAndSegmentation\\001640.jpg"
# img = cv2.imread(img_path)
#
# if img is None:
#     print(f"Не удалось загрузить изображение по пути: {img_path}")
#     exit()
#
# # Преобразование в numpy array (не обязательно, так как cv2.imread уже возвращает numpy array)
# img = np.array(img, dtype=np.uint8)
#
# # Создаем DataFrame с изображениями (а не с метками)
# df = pd.DataFrame({"image": [img]})  # Помещаем изображение в DataFrame
#
#
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


# Правило 3: Контраст (новое)
@labeling_function()
def lf_high_contrast(row):
    image = row["image"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return 1 if contrast > 50 else 0


# Применяем правила
lfs = [lf_red_pixels, lf_brightness, lf_high_contrast]
applier = PandasLFApplier(lfs)
L_train = applier.apply(df)

# Объединяем метки
label_model = LabelModel()
label_model.fit(L_train)
df["label"] = label_model.predict(L_train)

precision = precision_score(L_val, label_model.predict)
recall = recall_score(L_val, label_model.predict)
