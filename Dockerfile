# Use an official Python runtime as a parent image
FROM python:3.10.10-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy specific files
COPY ./app.py ./app.py
COPY ./functions.py ./functions.py
COPY ./requirements.txt ./requirements.txt
COPY ./best_model.pkl ./best_model.pkl
COPY ./pipeline.pkl ./pipeline.pkl

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uses this port
EXPOSE 8501 

# Run streamlit app
CMD ["streamlit", "run", "app.py"]
