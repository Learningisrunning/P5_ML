#!/bin/bash
# Lancer FastAPI en arrière-plan
nohup uvicorn API.app:app --host=0.0.0.0 --port=8000 &

# Lancer Streamlit
streamlit run stream_lite.app_streamlit.py --server.port=${PORT:-8501} --server.enableCORS=false
