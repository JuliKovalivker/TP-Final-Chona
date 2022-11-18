
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import cv2

def load(filename):
    np_image = Image.open(filename)
    plt.imshow(np_image, cmap = 'gray')
    plt.xticks([])
    plt.yticks([])
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def selectCharacter(prediction):
    i = 0
    predicted_character = 0
    character_prediction = 0
    for pred in prediction[0]:
        if pred > character_prediction:
            character_prediction = pred
            predicted_character = i
        i += 1
    return predicted_character, character_prediction

tab1, tab2, tab3 = st.tabs(["Proba el modelo!", "Como lo desarrollamos?", "Que personaje sos vos?"])

with tab1:
    st.title('TP Final - Rick & Morty character classification ')
    st.subheader('Subi una imagen de un personaje y te decimos si es Rick, Meeseeks, Morty, Poopybutthole o Summer!')
    img = st.file_uploader("")
    if img:
        model = tf.keras.models.load_model("Rick-Morty-Model-BEST.h5")
        st.image(img)
        img = load(img)
        prediction = model.predict(img)
        predicted_character, character_prediction = selectCharacter(prediction)
        characters = ["Mr Meeseeks", "Morty", "Mr PoopyButtHole", "Rick", "Summer"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label = "El personaje es:" , value = (characters[predicted_character] ))
        with col2:
            st.metric(label= "Con una probabilidad de:", value= str(round(character_prediction*100  , 2)) + "%")


with tab2:
    st.subheader('Nosotros decidimos clasificar a 5 de los personajes de Rick & Morty. Para comenzar, lo primero que hicimos fue buscar un dataset. Terminamos utilizando uno de Kaggle.')
    st.image("Screenshot 2022-11-18 083930.png")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image('00000000.png')
    with col2:
        st.image('00000003.jpg')
    with col3:
        st.image('00000009.png')
    with col4:
        st.image('00000010.jpg')
    with col5:
        st.image('00000013.jpg')
    st.subheader('Despues le hicimos data augmentation al dataset para mejorar la acurracy y reducir el overfitting del modelo.')
    st.code('''train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
    )''')
    st.subheader("Ahora si, creamos el modelo. Para decidir las capas del modelo empezamos con una base que nos parecia bien a nosotros y luego a base de prueba y error las fuimos cambiando")
    st.code('''cnn=tf.keras.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=16,padding='same',strides=2,kernel_size=3,activation='relu',input_shape=(224,224,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',strides=2,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Conv2D(filters=64,padding='same',strides=2,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Conv2D(filters=128,padding='same',strides=2,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(5,activation='sigmoid'))''')
    st.image('.\download.png')
    st.subheader('Después de haber creado y compilado el modelo lo entrenamos.')
    st.code('model = cnn.fit(train_datagen, epochs = 30, validation_data = test_datagen')
    st.subheader('En el entrenamiento el modelo pudo llegar a %80 de accuracy!!')


with tab3:
    st.title("Sacate una foto para ver que personaje sos vos!")
    img = st.camera_input("")
    if img:
        model = tf.keras.models.load_model("Rick-Morty-Model-BEST.h5")
        img = load(img)
        prediction = model.predict(img)
        predicted_character, character_prediction = selectCharacter(prediction)
        characters = ["Mr Meeseeks", "Morty", "Mr PoopyButtHole", "Rick", "Summer"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label = "El personaje es:" , value = (characters[predicted_character] ))
        with col2:
            st.metric(label= "Con una probabilidad de:", value= str(round(character_prediction*100  , 2)) + "%")
        
