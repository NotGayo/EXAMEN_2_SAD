# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
"""

import random
import re
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
from nltk import NaiveBayesClassifier
# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar los recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from tqdm import tqdm

# Funciones auxiliares


#####################
##NOMBRES VAIRABLES##
#####################

nom_modelo = "dt_2.pkl"
nom_modelo_csv = "dt_2.csv"
nom_pred_csv = "pred_test_dt_2.csv"
nom_carpeta_model = "OUTPUT_dt_2"




def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para calcular métricas

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    from sklearn.metrics import f1_score
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_precision(y_test, y_pred):
    """
    Función para calcular la precision
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: precision (micro), precision (macro)
    """
    from sklearn.metrics import precision_score
    precision_micro = precision_score(y_test, y_pred, average='micro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    return precision_micro, precision_macro

def calculate_accuracy(y_test, y_pred):
    """
    Función para calcular el accuracy
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    """
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def calculate_recall(y_test, y_pred):
    """
    Función para calcular el recall
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: recall (micro), recall (macro)
    """
    from sklearn.metrics import recall_score
    recall_micro = recall_score(y_test, y_pred, average='micro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    return recall_micro, recall_macro

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return cm


# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
        numerical_feature.columns = numerical_feature.columns.astype(str)
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature.columns = categorical_feature.columns.astype(str)
        categorical_feature = categorical_feature.loc[:,
                              categorical_feature.nunique() <= args.preprocessing[
                                  "unique_category_threshold"]]  # Si en el JSON no existe este apartado, añadirlo manualmente aqui
        # Por defecto usar mejor 10 proque significa que la columna tiene 10 valores o menos

        # Text features


        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)
        text_feature.columns = text_feature.columns.astype(str)
        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + "> Columnas numéricas:\n" + Fore.RESET, numerical_feature.columns)
            print(numerical_feature)

            print(Fore.MAGENTA + "> Columnas de texto:\n" + Fore.RESET, text_feature.columns)
            print(text_feature)

            print(Fore.MAGENTA + "> Columnas categóricas:\n" + Fore.RESET, categorical_feature.columns)
            print(categorical_feature)
        print(numerical_feature)
        print(text_feature)
        print(categorical_feature)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada.

    Para las características numéricas, se usa la media o mediana para imputar los valores faltantes.
    Para las características categóricas, se usa la moda para imputar los valores faltantes.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None
    """

    # Reemplazar valores vacíos ("") por NaN para que pandas los reconozca
    data.replace("", np.nan, inplace=True)

    # Eliminar filas que contengan al menos un NaN en cualquier columna
    data.dropna(inplace=True)

    # Separar columnas numéricas y categóricas
    numerical_feature = data.select_dtypes(include=['int64', 'float64'])
    categorical_feature = data.select_dtypes(include=['object'])

    # Imputar valores faltantes en características numéricas (Media)
    numerical_imputer = SimpleImputer(strategy='mean')
    if not numerical_feature.empty:
        numerical_feature[:] = numerical_imputer.fit_transform(numerical_feature)

    # Imputar valores faltantes en características categóricas (Moda)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    if not categorical_feature.empty:
        categorical_feature[:] = categorical_imputer.fit_transform(categorical_feature)

    # Reconstruir el DataFrame original después de la imputación
    processed_data = pd.concat([numerical_feature, categorical_feature], axis=1)

    ''' 
    # Imputar valores faltantes en características numéricas
    numerical_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')  # O usa 'median' si prefieres la mediana
    numerical_feature[:] = numerical_imputer.fit_transform(numerical_feature)

    # Imputar valores faltantes en características categóricas
    categorical_imputer = SimpleImputer(strategy='most_frequent')  # Imputación con la moda
    categorical_feature[:] = categorical_imputer.fit_transform(categorical_feature)
    '''


    # Eliminar filas que contengan NaN después del reemplazo
    #numerical_feature.dropna(inplace=True)
    #categorical_feature.dropna(inplace=True)

    print("Valores faltantes procesado exitosamente.")


def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.
    """
    try:
        # MinMaxScaler: Escala las características al rango [0, 1]
        min_max_scaler = MinMaxScaler()
        numerical_feature[:] = min_max_scaler.fit_transform(numerical_feature)

        # Otras opciones que se pueden habilitar según tus necesidades:
        # StandardScaler: Estandariza las características para que tengan media 0 y desviación 1
        # standard_scaler = StandardScaler()
        # numerical_feature[:] = standard_scaler.fit_transform(numerical_feature)

        # MaxAbsScaler: Escala sin cambiar los signos, manteniendo la relación original de los valores
        # max_abs_scaler = MaxAbsScaler()
        # numerical_feature[:] = max_abs_scaler.fit_transform(numerical_feature)

        # Normalizer: Normaliza las características para que la norma sea 1
        # normalizer = Normalizer()
        # numerical_feature[:] = normalizer.fit_transform(numerical_feature)

        print("Características numéricas reescaladas exitosamente.")

    except Exception as e:
        raise Exception(f"Error al reescalar los datos: {e}")


def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
    label_encoder = LabelEncoder()

    for col in categorical_feature:
        data[col] = label_encoder.fit_transform(data[col])


def clean_text(text, stop_words, stemmer):
    """
    Limpia un texto aplicando normalización, eliminación de stopwords y stemming.

    Parámetros:
    - text (str): Texto a procesar.
    - stop_words (set): Conjunto de stopwords.
    - stemmer (PorterStemmer): Stemmer para reducir palabras a su raíz.

    Retorna:
    - str: Texto procesado.
    """
    if not isinstance(text, str):  # Evita errores con valores nulos o no textuales
        return ""

    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\W+', ' ', text)  # Eliminar caracteres especiales y puntuación
    tokens = word_tokenize(text)  # Tokenizar el texto
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # Stemming y eliminación de stopwords

    return " ".join(tokens)  # Reconstruir el texto limpio


def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....

    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.

    Retorna:
    None
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    for col in text_feature:
        data[col] = data[col].apply(lambda x: clean_text(x, stop_words, stemmer))


def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":
                tfidf_vectorizer = TfidfVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(tfidf_matrix.toarray(),
                                                columns=tfidf_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN + "Texto tratado con éxito usando BOW" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].

    Args:
        None

    Returns:
        None

    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    try:
        X = data.drop(args.prediction, axis=1)
        y = data[args.prediction]

        sampling_strategy = args.preprocessing["sampling"]

        if sampling_strategy == "oversampling":
            sampler = SMOTE(random_state=42)
        elif sampling_strategy == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X, y  # Devuelve los datos sin cambios si no se especifica un método válido

        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    except Exception as e:
        raise Exception(f"Error al realizar el {sampling_strategy}: {e}")


def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.
    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)
        print(e)
        sys.exit(1)


def preprocesar_datos(data):
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """

    if args.algorithm == "kNN":
        # Separamos los datos por tipos
        numerical_feature, text_feature, categorical_feature = select_features()

        # Simplificamos el texto
        simplify_text(text_feature)

        # Pasar los datos a categoriales a numéricos
        cat2num(categorical_feature)

        # Tratamos missing values


        # Reescalamos los datos numéricos
        reescaler(numerical_feature)

        # Tratamos el texto
        if not text_feature.empty:
            process_text(text_feature)

        # Realizamos Oversampling o Undersampling
        over_under_sampling()

        if args.preprocessing["drop_features"] != " ":
            drop_features()
        process_missing_values(numerical_feature, categorical_feature)
    else:
        numerical_feature, text_feature, categorical_feature = select_features()
        cat2num(categorical_feature)
        process_missing_values(numerical_feature, categorical_feature)

    return data


# Funciones para entrenar un modelo



def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - x_test: DataFrame con las características de prueba.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    - y_test: Serie con las etiquetas de prueba.
    """

    try:
        # Extraer la columna objetivo (target)
        target_column = args.prediction
        if target_column not in data.columns:
            raise ValueError(f"La columna objetivo '{target_column}' no está en el archivo {args.file}.")

        # Separar características (X) y variable objetivo (y)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Primero dividimos en entrenamiento (70%) y un conjunto temporal (30%)
        x_train, x_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

        # Luego dividimos x_temp en validación (15%) y prueba (15%)
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

        print(Fore.GREEN + "Datos divididos en 70% entrenamiento, 15% validación y 15% prueba." + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + f"> Tamaño del conjunto de entrenamiento: {x_train.shape}" + Fore.RESET)
            print(Fore.MAGENTA + f"> Tamaño del conjunto de validación: {x_val.shape}" + Fore.RESET)
            print(Fore.MAGENTA + f"> Tamaño del conjunto de prueba: {x_test.shape}" + Fore.RESET)

        # Guardar los .csv del test para luego usarlos en el predict
        x_test.to_csv(nom_carpeta_model+"/X_test.csv", index=False)
        y_test.to_csv(nom_carpeta_model+"/y_test.csv", index=False)

        return x_train, x_val, y_train, y_val

    except Exception as e:
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        ##CAMBIAR AL MODELO QUE SEA EN EL EXAMEN
        with open(nom_carpeta_model+'/'+nom_modelo, 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
        with open(nom_carpeta_model+'/'+nom_modelo_csv, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev  = divide_data()
    
    # Hacemos un barrido de hiperparametros

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        #TODO Llamar al decision trees
        gs = GridSearchCV(DecisionTreeClassifier(),args.decission_tree, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()

    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(RandomForestClassifier(), args.random_forest, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)



def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open(nom_carpeta_model+'/'+nom_modelo, 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    X_test = pd.read_csv(nom_carpeta_model+"/X_test.csv")
    y_test = pd.read_csv(nom_carpeta_model+"/y_test.csv")


    model = load_model()

    # Asegurar que la columna objetivo no está en los datos de entrada
    if args.prediction in X_test.columns:
        X_test = X_test.drop(columns=[args.prediction])

    # Realizar predicción
    prediction = model.predict(X_test)
    print(args.prediction+"_pred")
    # Guardar las predicciones en un CSV
    X_test[args.prediction + "_pred"] = prediction
    print(X_test.columns)
    X_test.to_csv(nom_carpeta_model+"/"+nom_pred_csv, index=False)

    print(Fore.GREEN + "[INFO] Predicción completada y guardada en "+nom_carpeta_model+"/"+nom_pred_csv+""+ Fore.RESET)

    # Evaluar si existe y_test
    try:
        y_test = pd.read_csv(nom_carpeta_model+"/y_test.csv")
        accuracy = accuracy_score(y_test, prediction)
        print(Fore.CYAN + f"[INFO] Precisión en test: {accuracy:.4f}" + Fore.RESET)
        print(Fore.CYAN + "[INFO] Reporte de clasificación:\n" + Fore.RESET, classification_report(y_test, prediction, zero_division= 1))
    except FileNotFoundError:
        print(Fore.YELLOW + "[WARNING] No se encontró 'y_test.csv'. Solo se guardaron las predicciones." + Fore.RESET)
    
# Función principal
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta "+nom_carpeta_model+"...")
    try:
        os.makedirs(nom_carpeta_model)
        print(Fore.GREEN+"Carpeta"+nom_carpeta_model+"creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta "+nom_carpeta_model+" ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta "+nom_carpeta_model+" "+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos(data)
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv(nom_carpeta_model+'/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decission_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)

            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
