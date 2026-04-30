FROM python:3.13-slim
RUN pip install --no-cache-dir aiohttp==3.10.10
COPY hl_proxy.py /hl_proxy.py
EXPOSE 8888
CMD ["python", "-u", "/hl_proxy.py"]
