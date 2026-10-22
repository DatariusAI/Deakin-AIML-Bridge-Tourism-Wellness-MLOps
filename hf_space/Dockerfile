# Hugging Face Space (Docker SDK) for the Wellness Tourism Package predictor.
# The image pulls the public GitHub repo, so the Space always serves the model
# the GitHub Actions pipeline last committed to main.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user
RUN git clone --depth 1 https://github.com/DatariusAI/Deakin-AIML-Bridge-Tourism-Wellness-MLOps.git app
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r tourism_project/deployment/requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "tourism_project/deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
