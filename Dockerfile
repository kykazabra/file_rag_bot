FROM python:latest

RUN git clone https://github.com/kykazabra/file_rag_bot.git

RUN cd file_rag_bot

RUN pip install -r requirements.txt

CMD ["python", "bot.py"]