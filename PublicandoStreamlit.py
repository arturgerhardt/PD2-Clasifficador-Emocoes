import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image

# Carregar o modelo previamente treinado
model = tf.keras.models.load_model('./models/ceMod2.h5')


classes = ['Raiva', 'Alegria', 'Neutro', 'Triste', 'Surpresa']

# Função para realizar a previsão 
def predict_image(file_path):
    img = image.load_img(file_path, target_size=(200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)

    emocao = classes[np.argmax(prediction)]
    return prediction, emocao

# Configuração do Streamlit
st.title('Classificador de Emoções em Imagens')
st.write('Carregue uma imagem e receba a classificação de emoção entre Raiva, Alegria, Neutro, Triste, Surpresa.')

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagem carregada.', use_column_width=True)
    st.write("")
    st.write("Classificando...")

    predictions, emocao = predict_image(image)

    st.write(f'A emoção classificada é: {emocao}')
    st.write('Resultados:')
    st.write(f'Raiva: {round(predictions[0]*100,2)}%')
    st.write(f'Alegria: {round(predictions[1]*100,2)}%')
    st.write(f'Neutro: {round(predictions[2]*100,2)}%')
    st.write(f'Triste: {round(predictions[3]*100,2)}%')
    st.write(f'Surpresa: {round(predictions[4]*100,2)}%')