# defining-emotions-using-CNN
Development of a convolutional neural network designed to detect human emotions in real time and from a video file
# Библиотеки, которые должны быть установлены:
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow
# Загрузите набор данных по ссылке ниже и поместите в папку данных в каталоге вашего проекта
https://www.kaggle.com/msambare/fer2013
# Обучение нейронной сети по всем изображениям лиц в наборе данных FER2013:
command --> python TranEmotionDetector.py
Это займет несколько часов, зависит от вашего процессора. После обучения вы обнаружите, что структура обученной модели и веса хранятся в вашем каталоге проекта.
emotion_model.json
emotion_model.h5
Скопируйте эти два файла в папку create model в каталоге вашего проекта и вставьте ее.
# Запустите свой тестовый файл обнаружения эмоций
TestEmotionDetector.py
