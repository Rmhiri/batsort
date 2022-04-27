FROM python:3.8
WORKDIR /app_parasi
COPY requirements.txt ./requirements.txt
#RUN conda activate rahma_env
RUN pip install -r requirements.txt
EXPOSE 8501
#COPY ./app_parasi
COPY . .
ENTRYPOINT ["streamlit", "run"]

CMD ["predict_streamlit.py"]
#CMD streamlit run --server.port 8501 --server.enableCORS false app.py
