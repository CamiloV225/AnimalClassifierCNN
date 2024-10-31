import streamlit as st  # Librería para construir la app web
import pandas as pd  # Para manipulación de datos
import matplotlib.pyplot as plt  # Para gráficos
import seaborn as sns  # Para visualización avanzada
import torch
import os
from torchvision import transforms
import random
from PIL import Image

def page_load():
    # Definir rutas de carpetas para train y test de cada clase
    train_hyena_dir = "C:/Users/camil/OneDrive/Documents/ANALITICA/Streamlit/archive/train/hyena"
    train_cheetah_dir = "C:/Users/camil/OneDrive/Documents/ANALITICA/Streamlit/archive/train/cheetah"
    test_hyena_dir = "C:/Users/camil/OneDrive/Documents/ANALITICA/Streamlit/archive/validation/hyena"
    test_cheetah_dir = "C:/Users/camil/OneDrive/Documents/ANALITICA/Streamlit/archive/validation/cheetah"

    # Crear lista de rutas de imágenes, categorías y conjuntos (train/test)
    image_paths = []
    datasets = [
        (train_hyena_dir, "Hyena", "Train"),
        (train_cheetah_dir, "Cheetah", "Train"),
        (test_hyena_dir, "Hyena", "Test"),
        (test_cheetah_dir, "Cheetah", "Test")
    ]
    
    for dir_path, label, dataset_type in datasets:
        image_list = os.listdir(dir_path)
        image_paths.extend([(os.path.join(dir_path, img), label, dataset_type) for img in image_list])

    # Crear DataFrame con nombres, categorías y tipo de conjunto
    df_animals = pd.DataFrame(image_paths, columns=["File_Path", "Category", "Dataset_Type"])

    # Título y descripción
    st.title("Dashboard de Cheeta vs Hiena")
    st.markdown("Este dashboard interactivo utiliza el **Cheeta vs Hiena** para la clasificación binaria de imágenes.")

    # Exploración del Dataset
    st.header("Exploración del Dataset")
    st.write("Vista preliminar del dataset:")
    st.dataframe(df_animals.head(10)) 

    # Mostrar imágenes aleatorias
    mostrar_ejemplos_en_cuadricula(image_paths)

    # Cálculo de Media y Desviación de canales RGB
    transform = transforms.ToTensor()
    all_images = [transform(Image.open(img_path).convert("RGB")) for img_path, _, _ in image_paths]
    all_images_tensor = torch.stack(all_images)

    # Calcular medias y desviaciones estándar para cada canal (RGB)
    mean_rgb = torch.mean(all_images_tensor, dim=[0, 2, 3])
    std_rgb = torch.std(all_images_tensor, dim=[0, 2, 3])

    # Visualización de estadísticas RGB en Streamlit
    st.title("Estadísticas de los canales RGB")
    st.write(f"Media de los canales RGB: {mean_rgb}")
    st.write(f"Desviación estándar de los canales RGB: {std_rgb}")

    # Gráfico de media y desviación estándar
    fig, ax = plt.subplots()
    channels = ['Red', 'Green', 'Blue']
    ax.bar(channels, mean_rgb, yerr=std_rgb, color=['red', 'green', 'blue'], capsize=5)
    ax.set_ylabel('Valor')
    ax.set_title('Media y Desviación Estándar de los Canales RGB')
    st.pyplot(fig)

    # Histograma de balanceo de clases por conjunto (train/test)
    class_counts = df_animals.groupby(['Category', 'Dataset_Type']).size().unstack(fill_value=0)
    fig, ax = plt.subplots()
    class_counts.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'salmon'])
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Distribución de Clases en el Dataset por Conjunto')
    st.title("Distribución de Clases en el Dataset")
    st.pyplot(fig)

def mostrar_ejemplos_en_cuadricula(image_paths):
    st.title("Ejemplos aleatorios de imágenes de animales")
    
    # Seleccionar aleatoriamente 9 imágenes
    random_images = random.sample(image_paths, k=9)
    
    # Mostrar imágenes en una cuadrícula de 3x3
    rows = 3
    cols = 3
    for i in range(rows):
        columns = st.columns(cols)
        for j in range(cols):
            img_index = i * cols + j
            img_path, label, dataset_type = random_images[img_index]
            with columns[j]:
                st.image(Image.open(img_path), caption=f"{label} - {dataset_type}", use_column_width=True)