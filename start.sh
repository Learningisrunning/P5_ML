#!/bin/bash

# Démarrer FastAPI avec Uvicorn
echo "Lancement de FastAPI..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Attendre un peu pour que FastAPI soit lancé
sleep 5

# Lancer Streamlit
echo "Lancement de Streamlit..."
streamlit run stream_lite/app_streamlit.py --server.port 8501
