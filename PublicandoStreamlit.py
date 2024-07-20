import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image

# Carregar o modelo previamente treinado
model = tf.keras.models.load_model('./models/ceMod2.h5')


classes = ['Raiva', 'Alegria', 'Neutro', 'Triste', 'Surpresa']

# Função para realizar a previsão 
def predict_image(imagem):
    #img = image.load_img(imagem, target_size=(200, 200))
    img = image.img_to_array(imagem)
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
    imagem = Image.open(uploaded_file)
    st.image(imagem, caption='Imagem carregada.', use_column_width=True)
    st.write("")

    predictions, emocao = predict_image(imagem)

    st.write(f'A emoção classificada é: {emocao}')
    st.write('Resultados:')
    st.write(f'Raiva: {predictions[0][0]*100:.2f}%')
    st.write(f'Alegria: {predictions[0][1]*100:.2f}%')
    st.write(f'Neutro: {predictions[0][2]*100:.2f}%')
    st.write(f'Triste: {predictions[0][3]*100:.2f}%')
    st.write(f'Surpresa: {predictions[0][4]*100:.2f}%')