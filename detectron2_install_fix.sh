#!/usr/bin/env bash
set -euo pipefail

# Предполагается: виртуальное окружение уже активировано (source .venv/bin/activate)

source .venv/bin/activate

echo "1) Убедимся, что pip/setuptools/wheel обновлены"
python -m pip install --upgrade pip setuptools wheel

echo "2) Проверим, что torch установлен и импортируется в текущем окружении"
python - <<'PY'
import sys
try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("Ошибка импорта torch:", e)
    sys.exit(2)
PY

echo "3) Клонируем detectron2 (если ещё не клонирован)"
DETECTRON2_REPO="https://github.com/facebookresearch/detectron2.git"
TMPDIR="/tmp/detectron2"
if [ -d "${TMPDIR}" ]; then
  rm -rf "${TMPDIR}"
fi
git clone --depth 1 "${DETECTRON2_REPO}" "${TMPDIR}"
cd "${TMPDIR}"

echo "4) Отключаем CUDA и тестовую сборку"
export FORCE_CUDA=0
export D2_BUILD_TEST=0

# На macOS иногда нужно указать флаги для libomp (если brew установлен)
if command -v brew >/dev/null 2>&1 && brew list libomp >/dev/null 2>&1; then
  export CFLAGS="-I$(brew --prefix libomp)/include ${CFLAGS:-}"
  export LDFLAGS="-L$(brew --prefix libomp)/lib ${LDFLAGS:-}"
  echo "Set CFLAGS=${CFLAGS}"
  echo "Set LDFLAGS=${LDFLAGS}"
fi

echo "5) Пытаемся установить detectron2 без build isolation (pip будет использовать текущее venv)"
python -m pip install --no-build-isolation -e .

# Если предыдущая команда прошла успешно, проверим импорт
if python - <<'PY'
try:
    import detectron2
    print("detectron2 import OK")
    raise SystemExit(0)
except Exception as e:
    print("detectron2 import failed:", e)
    raise SystemExit(1)
PY
then
  echo "detectron2 успешно установлен (editable)."
  cd -
  exit 0
fi

echo "6) Резервный путь: собрать wheel в текущем окружении и установить его"
# Соберём wheel в /tmp/detectron2_wheel
WHEEL_DIR="/tmp/detectron2_wheel"
rm -rf "${WHEEL_DIR}"
mkdir -p "${WHEEL_DIR}"
python -m pip wheel -w "${WHEEL_DIR}" .
# Установим первый собранный wheel
python -m pip install "${WHEEL_DIR}"/detectron2-*.whl

echo "Проверка импорта detectron2 (после установки wheel)"
python - <<'PY'
import detectron2
print("detectron2 import OK, version:", getattr(detectron2, "__version__", "unknown"))
PY

cd -
echo "Готово."