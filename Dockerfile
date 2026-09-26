FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Системные зависимости для сборки dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dlib собирается из исходников с ЯВНО отключёнными AVX/AVX2/SSE4, а не через
# генерический wheel (как dlib-bin). CMake dlib автоматически включает AVX,
# если это поддерживает КОМПИЛЯТОР сборочной машины — а не если это
# поддерживает CPU, на котором контейнер будет реально запущен. Именно
# рассинхронизация "собрано на новом CPU / выполняется на старом" — причина
# SIGILL, из-за которой распознавание лиц в интеграции intersvyaz падает и
# откатывается на portable-движок. Собирая с baseline SSE2 здесь, образ
# гарантированно запускается на любом x86_64, независимо от того, где он
# был собран.
ARG DLIB_VERSION=19.24.6
RUN git clone --depth 1 --branch v${DLIB_VERSION} https://github.com/davisking/dlib.git /tmp/dlib \
 && cd /tmp/dlib \
 && python setup.py install \
    --no USE_AVX_INSTRUCTIONS \
    --no USE_SSE4_INSTRUCTIONS \
    --no DLIB_USE_CUDA \
 && rm -rf /tmp/dlib

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

# Keep-alive дольше, чем у клиента (общая aiohttp-сессия Home Assistant держит
# соединение 15 с). С дефолтными 5 с uvicorn закрывал соединение, в которое
# HA тут же слал запрос, и интеграция получала ServerDisconnectedError.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
