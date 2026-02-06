# -------------------------------
# Python 3.11 slim для arm64
# -------------------------------
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------------
# Системные зависимости
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        cmake \
        git \
        wget \
        curl \
        unzip \
        pkg-config \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libfreetype6-dev \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        ca-certificates \
        vim \
        ninja-build \
        git-lfs \
    && rm -rf /var/lib/apt/lists/*

# (libgl1-mesa-glx отсутствует в arm64 slim, используем другие X-библиотеки)

# -------------------------------
# Обновляем pip/setuptools/wheel
# -------------------------------
RUN python -m pip install --upgrade pip setuptools wheel cmake --no-cache-dir

# -------------------------------
# Проверяющий шаг: Версия pip и site-packages
# -------------------------------
RUN python -m pip --version && python -c "import sys; print(sys.path)"

# -------------------------------
# Устанавливаем PyTorch (CPU only для linux/arm64)
# -------------------------------
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# -------------------------------
# Проверяем torch: импорт и версия
# -------------------------------
RUN python -c "import torch; print('torch OK:', torch.__version__)"

# -------------------------------
# Data science / Jupyter зависимости
# -------------------------------
RUN python -m pip install --no-cache-dir \
        matplotlib \
        opencv-python-headless \
        pycocotools \
        pillow \
        tqdm \
        pandas \
        seaborn \
        notebook \
        jupyterlab \
        ipywidgets \
        scikit-learn \
        tensorboard \
        albumentations

# -------------------------------
# Переменные для сборки Detectron2 (без CUDA)
# -------------------------------
ENV FORCE_CUDA=0
ENV D2_BUILD_TEST=0

# -------------------------------
# Устанавливаем Detectron2 c "pyproject.toml", строго не в editable и с проверкой на torch
# -------------------------------

RUN git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2 \
    && cd /tmp/detectron2 \
    && python -c "import torch; print('Torch available for detectron2 setup:', torch.__version__)" \
    && python -m pip install --no-cache-dir . \
    && cd / && rm -rf /tmp/detectron2

# -------------------------------
# Проверяем detectron2: импорт и версия
# -------------------------------
RUN python -c "import detectron2; print('detectron2 OK, version:', getattr(detectron2, '__version__', 'unknown'))"

# -------------------------------
# Рабочая директория
# -------------------------------
WORKDIR /workspace

# expose Jupyter
EXPOSE 8888

# команда запуска по умолчанию
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]