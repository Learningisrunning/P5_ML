import streamlit as st
import requests

#Front of the API
API_URL = "https://tags-prediction-2a1cfec639e1.herokuapp.com/predict"

st.title("Prédiction des Tags - Interface API")

st.header("Entrez les données pour prédire les tags")
title = st.text_input("Titre :", placeholder="Entrez le titre ici...")
body = st.text_area("Message :", placeholder="Entrez le contenu ici...")

if st.button("Prédire"):
    if title and body:
        input_data = {"Title": title, "Body": body}
        with st.spinner("Envoi des données à l'API..."):
            try:
                response = requests.post(API_URL, json=input_data)
                if response.status_code == 200:
                    predictions = response.json().get("predictions", [])
                    st.success("Prédictions obtenues avec succès !")
                    if predictions:
                        st.table(predictions)
                    else:
                        st.info("Aucune prédiction retournée.")
                else:
                    st.error(f"Erreur {response.status_code} : {response.json().get('detail', 'Erreur inconnue')}")
            except Exception as e:
                st.error(f"Erreur lors de la connexion à l'API : {str(e)}")
    else:
        st.warning("Veuillez remplir tous les champs avant de prédire.")
