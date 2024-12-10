#!/bin/bash
# Lancer FastAPI en arrière-plan
uvicorn fastapi_app.main:app --host=0.0.0.0 --port=8000 &

# Lancer Streamlit
streamlit run streamlit_app/app.py --server.port=${PORT} --server.enableCORS=false
