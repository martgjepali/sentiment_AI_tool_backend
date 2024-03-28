# Start from the official Python 3.12 image
FROM python:3.12-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file to the working directory
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the content of the local app directory to the working directory
COPY ./app .

CMD uvicorn main:app --port 8000 --host 0.0.0.0