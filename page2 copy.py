import time

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from classes import AnimalClassifierS, AnimalClassifierM, pytorch_data

# Shared variable for elapsed time
time_elapsed = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to run the timer in a separate thread
def start_timer():
    global time_elapsed
    while True:
        time.sleep(1)  # Increment every second
        time_elapsed += 1  # Increment the time counter by 1 second

def cargar_dataloaders(ruta_datos, train_size=0.8, batch_size=16):
    # Cargar el dataset
    dataset = pytorch_data(ruta_datos)
    dataset.classes
    st.write(dataset.classes)
    
    # Crear los dataloaders de entrenamiento y validación
    train_loader, val_loader = dataset.create_dataloaders(train_size=train_size, batch_size=batch_size)
    
    return train_loader, val_loader

def evaluate_model(model, test_loader):
    model.eval()  # Modo evaluación
    all_preds = []  # Lista para almacenar las predicciones
    all_labels = []  # Lista para almacenar las etiquetas verdaderas

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def confusion_matrix(model, test_loader):
    labels, preds = evaluate_model(model, test_loader)
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bear', 'Panda'])

    print("Matriz de confusión:")
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def train_model(model, model_file_path, train, val, criterion, optimizer, device, epochs):

    start_time = time.time()

    estado = st.empty()
    estado.write("Entrenando modelo...")
    progress_bar = st.progress(0)
    time_placeholder = st.empty()  # Barra de progreso de Streamlit
    loss_placeholder = st.empty()  # Espacio reservado para mostrar la pérdida
    epoch_placeholder = st.empty()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Cálculo de la pérdida promedio de la época
        avg_loss = running_loss / len(train)
        
        # Actualización de la barra de progreso y de los mensajes de estado
        progress_bar.progress((epoch + 1) / epochs)
        epoch_placeholder.write(f"Época {epoch + 1}/{epochs}")
        loss_placeholder.write(f"Pérdida: {avg_loss:.4f}")
        elapsed_time = time.time() - start_time
        time_placeholder.write(f"Tiempo transcurrido: {elapsed_time:.2f} segundos")

    confusion_matrix(model, val)
    estado.write("Entrenamiento completado.")
    torch.save(model.state_dict(), model_file_path)

    with open(model_file_path, "rb") as f:
        st.download_button(
            label="Descargar Modelo",
            data=f,
            file_name=model_file_path,
            mime="application/octet-stream"
        )


def page_load_2():
    st.title("Entrenamiento de Modelos CNN")
  
    modelos = {
        "AnimalClassifierS": AnimalClassifierS(),
        "AnimalClassifierM": AnimalClassifierM()
    }   

    optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,               
        "RMSprop": optim.RMSprop      
    }
    
    # Selección del modelo
    modelo_seleccionado = st.selectbox("Modelo", list(modelos.keys()))

    if modelo_seleccionado == 'AnimalClassifierS':
        model_file_path = 'cheetah_hyena_modelS.pth'
    elif modelo_seleccionado == 'AnimalClassifierM':
        model_file_path = 'cheetah_hyena_modelM.pth'


    # Hiperparámetros
    params = {
        "epochs": st.slider("Épocas", 1, 10, 5, key="epochs_slider"),
        "batch_size": st.slider("Tamaño de Batch", 16, 64, 32, key="batch_slider"),
        "learning_rate": st.selectbox("Learning Rate", [0.01, 0.001, 0.0001], key="learning_slider"),
        "optimizer_name": st.selectbox("Optimizador", list(optimizers.keys()), key="optimizer_selector")
    }

    # Cargar datos
    train, val = cargar_dataloaders('archive', batch_size=params['batch_size'])

    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if st.button("Entrenar Modelo"):
        try:
            # Obtener y mover el modelo al dispositivo
            model = modelos[modelo_seleccionado]
            model = model.to(device)

            # Configurar criterion
            criterion = nn.CrossEntropyLoss()

            # Configurar optimizer
            optimizer_class = optimizers[params["optimizer_name"]]
            optimizer = optimizer_class(model.parameters(), lr=params["learning_rate"])
            
            epochs = params["epochs"]
            train_model(model, model_file_path, train, val, criterion, optimizer, device, epochs)

        except Exception as e:
            st.error(f"Error durante la configuración del entrenamiento: {str(e)}")
    

