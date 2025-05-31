# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement du modèle et des noms de features
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

model, feature_names = load_model()


# Titre et description
st.title("Prédiction de la Consommation Électrique (kW/h)")
st.markdown("""
Cette application vous permet de prédire la consommation électrique en fonction des paramètres environnementaux et du bâtiment.
""")

# Création des colonnes pour l'interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Paramètres environnementaux")
    temperature = st.slider("Température (°C)", -10.0, 40.0, 20.0, 0.1)
    humidite = st.slider("Humidité (%)", 0.0, 100.0, 50.0, 0.1)
    vitesse_vent = st.slider("Vitesse du vent (km/h)", 0.0, 50.0, 10.0, 0.1)
    
    jour_semaine = st.selectbox("Jour de la semaine", 
                               ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])
    heure = st.slider("Heure de la journée", 0, 23, 12)

with col2:
    st.subheader("Caractéristiques du bâtiment")
    type_habitation = st.selectbox("Type d'habitation", ["Maison", "Appartement", "Bureau"])
    nombre_personnes = st.slider("Nombre de personnes", 1, 10, 2)
    
    # Bouton pour effectuer la prédiction
    predict_button = st.button("Prédire la consommation")

# Créer un dataframe avec les valeurs des inputs
if predict_button:
    input_data = pd.DataFrame({
        'temperature (°C)': [temperature],
        'humidite (%)': [humidite],
        'vitesse_vent (km/h)': [vitesse_vent],
        'jour_semaine': [jour_semaine],
        'heure': [heure],
        'type_habitation': [type_habitation],
        'nombre_personnes': [nombre_personnes]
    })
    
    # Effectuer la prédiction
    prediction = model.predict(input_data)[0]
    
    # Afficher la prédiction
    st.subheader("Résultat de la prédiction")
    st.metric(label="Consommation prédite (kW/h)", value=f"{prediction:.2f} kW/h")
    
    # Ajouter des visualisations et explications supplémentaires
    st.subheader("Facteurs influençant la consommation")
    
    # Créer des données de comparaison en faisant varier la température
    temp_range = np.linspace(temperature - 10, temperature + 10, 11)
    consumption_by_temp = []
    
    for temp in temp_range:
        temp_data = input_data.copy()
        temp_data['temperature (°C)'] = temp
        consumption_by_temp.append(model.predict(temp_data)[0])
    
    # Graphique de l'influence de la température
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(temp_range, consumption_by_temp, marker='o')
    ax.axvline(x=temperature, color='r', linestyle='--', label='Votre température')
    ax.set_xlabel('Température (°C)')
    ax.set_ylabel('Consommation prédite (kW/h)')
    ax.set_title('Influence de la température sur la consommation')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    # Créer des données de comparaison pour le nombre de personnes
    person_range = range(1, 11)
    consumption_by_person = []
    
    for persons in person_range:
        person_data = input_data.copy()
        person_data['nombre_personnes'] = persons
        consumption_by_person.append(model.predict(person_data)[0])
    
    # Graphique de l'influence du nombre de personnes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(person_range, consumption_by_person)
    ax.axvline(x=nombre_personnes, color='r', linestyle='--', label='Votre sélection')
    ax.set_xlabel('Nombre de personnes')
    ax.set_ylabel('Consommation prédite (kW/h)')
    ax.set_title('Influence du nombre de personnes sur la consommation')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Ajouter une section d'explication du modèle
st.markdown("---")
st.subheader("À propos du modèle")
st.write("""
Ce modèle a été entraîné sur un jeu de données contenant des informations sur la consommation électrique 
en fonction de divers facteurs environnementaux et caractéristiques des bâtiments. Les principales variables 
utilisées sont la température, l'humidité, la vitesse du vent, le jour de la semaine, l'heure, 
le type d'habitation et le nombre de personnes.

Deux algorithmes ont été testés:
- Support Vector Machine (SVM)
- Decision Tree Regressor

Le modèle avec la meilleure performance a été sélectionné pour cette application.
""")

# Footer
st.markdown("---")
st.markdown("Projet de Machine Learning Avancé | 2023")