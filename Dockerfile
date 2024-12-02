FROM python:latest

WORKDIR /app

RUN git clone https://github.com/kykazabra/file_rag_bot.git

WORKDIR /app/file_rag_bot

RUN pip install -r requirements.txt

CMD ["python", "bot.py"]