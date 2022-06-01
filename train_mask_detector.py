# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# 20 эпох
EPOCHS = 20
# коэффициент скорости обучения
# отвечает за величину коррекции весов (с каждой новой эпохой задаем разные веса)
INIT_LR = 1e-4
# размер одного батча
# батч - часть датасета, что бы прогонять в одной эпохе весь датасет по частям
BS = 32

DIRECTORY = r"D:\dev\python\face-mask-detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# собираю картинки из датасета
print("[INFO] loading images...")
data = []
labels = []
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)

# перевожу категории в бинарный вид
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# из существующего датасета генерирую новые картинки изменяя старые
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# загружю MobileNetV2 (сеть для компьютерного зрения)
# убедившись, что набор верхних уровней отключен
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

# строю вершину сети (та часть, через которую поступают данные)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# размещаю вершину модели над базовой моделью
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

# собираю модель
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# обучаем верхушку сети
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# делаем предположения на тестовом датасете
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# для каждого изображения нужно найти индекс категории 
# с соответствующей наибольшей прогнозируемой вероятностью
predIdxs = np.argmax(predIdxs, axis=1)

# красиво выводим отчет о классификации
print(classification_report(testY.argmax(axis=1), predIdxs,
    target_names=lb.classes_))

# сохраняем модель на диск
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# вывожу график обучения
# строим график ощибки и точности обучения
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
# ошибки в обучении
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# точность обучения
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")