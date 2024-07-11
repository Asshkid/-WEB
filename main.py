import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('bg.jpg')

# тайтл
st.title('Определение пневмонии')

# устанавливаем header
st.header('Пожалуйста загрузите изображение ')

# загрузка файла
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# загружаем модель классификации
model = load_model('./model/pneumonia_classifier.h5')

# загружаем классы
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()



# # выводим изображение
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # классифицируем
    class_name, conf_score = classify(image, model, class_names)

    # пишем классификатор
    st.write("## {}".format(class_name))
    st.write("### Вероятность: {}%".format(int(conf_score * 1000) / 10))