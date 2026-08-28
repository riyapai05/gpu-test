FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

CMD ["python", "train.py"]


# FROM
# ↓
# Take a PyTorch + CUDA environment

# WORKDIR
# ↓
# Create /app inside container

# COPY requirements.txt
# ↓
# Copy dependencies

# RUN pip install
# ↓
# Install dependencies

# COPY train.py
# ↓
# Copy our Python program

# CMD
# ↓
# Run train.py