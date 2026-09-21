# Document-QA runtime image.
# Prefer incremental rebuild from an existing local tag to avoid re-pulling torch.
# GPU OCR: uninstall CPU onnxruntime, install CUDA-matched onnxruntime-gpu.
ARG BASE_IMAGE=document-qa-assistant-worker:latest
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple \
    PIP_TRUSTED_HOST=mirrors.aliyun.com \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib

COPY src ./src
COPY requirements.txt constraints-torch.txt ./

# Docling pipeline option hashing needs pydantic>=2.12 (2.10.6 circular-ref crash).
RUN python -m pip install --upgrade-strategy only-if-needed "pydantic==2.13.5"

# Keep torch as-is on BASE_IMAGE. Only ensure GPU ORT matches that CUDA major.
# For torch cu124 use onnxruntime-gpu==1.20.2; for cu130 use >=1.27 (e.g. 1.29.0).
# Runtime CUDA EP also needs cuDNN 9 shared libs in the image (libcudnn_adv.so.9).
ARG ORT_GPU_VERSION=1.20.2
RUN python -m pip uninstall -y onnxruntime || true \
 && python -m pip install --upgrade-strategy only-if-needed "onnxruntime-gpu==${ORT_GPU_VERSION}" \
 && python -c "import torch, onnxruntime as ort, pydantic; print(torch.__version__, ort.__version__, pydantic.__version__, ort.get_available_providers())"

# D49: the host HF cache is bind-mounted here (compose: ${HF_CACHE_DIR:-$HOME/.cache/huggingface}).
ENV HF_HOME=/root/.cache/huggingface

# D50 (2026-09-21): the image prepares a non-root user, but it is NOT activated --
# measured on this deployment, `USER appuser` breaks the app.  On Docker Desktop for
# Windows the bind-mounted ./data is presented as root:root, so the existing database
# is mode 644 and uid 1000 cannot write it:
#     /app/data            drwxrwxrwx root root   -> uid 1000 CAN create files
#     /app/data/app.db     -rw-r--r-- root root   -> uid 1000 CANNOT write it
#     -> sqlite3.OperationalError: attempt to write a readonly database
# The remedy this block used to recommend ("chown ./data on the host once") is not
# available on Windows: the drive has no POSIX ownership to change.  Enabling non-root
# therefore needs an entrypoint that fixes ownership as root and then drops privileges
# (gosu/setpriv); until that exists, running as root is the working configuration.
# See docs/问题台账.md D50b for the evidence and the options.
RUN groupadd -g 1000 appuser \
 && useradd -m -u 1000 -g 1000 -s /bin/bash appuser \
 && mkdir -p /app/data \
 && chown -R 1000:1000 /app/data /root/.cache/huggingface

# USER appuser   # deliberately NOT enabled -- see the block above

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
