import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from classes import AnimalClassifierS, AnimalClassifierM, pytorch_data

def load_model(model_choice):
    if model_choice == "AnimalClassifierS":
        model = AnimalClassifierS()
        try:
            model_load = model
            model_load.load_state_dict(torch.load('models/cheetah_hyena_modelM.pth'))
        except Exception as e:
            print(e)
    else:
        model = AnimalClassifierM()
        try:
            model_load = model
            model_load.load_state_dict(torch.load('models/cheetah_hyena_modelM.pth'))
        except Exception as e:
            print(e)

    transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    #print(type(transform))

    return model_load, transform 

def predict_image(uploaded_file, model, transforms, class_names=['cheetah', 'hyena']):
    image = Image.open(uploaded_file)
    image = transforms(image).unsqueeze(0)  # Add batch dimension
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

        # Map predicted class index to class name
        predicted_class_name = class_names[predicted_class.item()]

    return predicted_class_name, confidence.item()


def mostrar_inferencia():
    st.title("Inferencia")
    model_choice = st.selectbox("Seleccione el modelo:", ("AnimalClassifierS", "AnimalClassifierM"))
    st.markdown("Sube una imagen para realizar la clasificación.")
    model, transform = load_model(model_choice)

    # Subida de imagen
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Abrir y mostrar la imagen cargada
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', use_column_width=True)
        st.write("")

        # Obtener la clase predicha y la confianza
        predicted_class_name, confidence = predict_image(uploaded_file, model, transform)
        print(f"predicted class: {predicted_class_name}, confidence: {confidence}")

        st.write(f"Clase predicha: {predicted_class_name}")
        st.write(f"Confianza: {confidence:.2f}")

        # Clases y probabilidades
        classes = ['cheetah','hyena']
        if predicted_class_name == 'cheetah':
            probabilities = [confidence, 1 - confidence]
        else:
            probabilities = [1 - confidence, confidence]

        # Mostrar probabilidades en gráfico de barras
        plot_placeholder = st.empty()
        with plot_placeholder.container():
            fig, ax = plt.subplots()
            ax.bar(classes, probabilities, color=['skyblue', 'salmon'])
            ax.set_ylabel("Probabilidad")
            ax.set_title("Distribución de Probabilidades por Clase")
            st.pyplot(fig)
