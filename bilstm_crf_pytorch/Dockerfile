FROM python:3.9

WORKDIR ./docker_demo
 
ADD . .
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r requirements.txt

CMD ["python", "./run_lstm_crf.py"]