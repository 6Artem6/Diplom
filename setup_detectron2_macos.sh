#!/usr/bin/env bash
set -euo pipefail

echo "1) Проверка Xcode Command Line Tools"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Устанавливаем Xcode Command Line Tools..."
  xcode-select --install || true
  echo "Если установка GUI открылась — завершите её и перезапустите скрипт."
  exit 1
else
  echo "Xcode CLI установлен."
fi

echo "2) Устанавливаем Homebrew зависимости (cmake, pkg-config, libomp, protobuf)"
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew не найден. Установите Homebrew вручную и перезапустите скрипт:"
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi

brew update
brew install cmake pkg-config libomp protobuf

echo "3) Создаём виртуальное окружение"
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

echo "4) Устанавливаем PyTorch, torchvision, torchaudio"
# Рекомендуется свериться с https://pytorch.org/get-started/locally
# Попробуем сначала стабильные колеса (обычно содержат MPS на macOS arm64)
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio

echo "Проверяем доступность MPS/CPU"
python - <<'PY'
import torch
print("torch", torch.__version__)
try:
    mps = torch.backends.mps.is_available()
except Exception as e:
    mps = False
print("MPS available:", mps)
PY

echo "5) Устанавливаем вспомогательные пакеты Python"
# Устанавливаем вспомогательные пакеты ПОСЛЕДОВАТЕЛЬНО, чтобы numpy гарантированно был до pycocotools/cython
# Предполагается, что venv уже активирован: source .venv/bin/activate
echo "Убедитесь, что виртуальное окружение активировано."

echo "Устанавливаем numpy и cython в окружение (нужно для сборки pycocotools)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade numpy cython

echo "Устанавливаем остальные зависимости Python"
python -m pip install --upgrade opencv-python-headless pillow tqdm pandas seaborn albumentations jupyterlab

echo "6) Клонируем cocoapi (если ещё не клонирован)"
if [ ! -d cocoapi ]; then
  git clone https://github.com/cocodataset/cocoapi.git
fi

echo "Переходим в PythonAPI и ставим pycocotools используя текущее окружение для сборки"
cd cocoapi/PythonAPI

# Используем --no-build-isolation чтобы pip не создавал чистое временное окружение без numpy
python -m pip install --no-build-isolation -e .

cd -
echo "pycocotools установлен."

echo "7) Клонируем detectron2 и ставим из исходников (без wheel)"
# Замените URL на ваш рабочий форк при необходимости
DETECTRON2_REPO="https://github.com/facebookresearch/detectron2.git"
git clone --depth 1 "${DETECTRON2_REPO}" /tmp/detectron2
cd /tmp/detectron2

# Принудительно без CUDA
export FORCE_CUDA=0
export D2_BUILD_TEST=0

python -m pip install -e .

echo "8) Проверяем импорт detectron2"
python - <<'PY'
import detectron2
print("detectron2 imported OK, version:", getattr(detectron2, "__version__", "unknown"))
PY

echo "9) Очистка временных файлов"
cd -
rm -rf /tmp/detectron2

echo "Готово. Активируйте виртуальное окружение: source .venv/bin/activate"
echo "Запуск тренировки: python train_ui.py"