# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN apt update
RUN apt install -y awscli ffmpeg 

# Copy the rest of the application code into the container
COPY . .

# Define the command to run the application
CMD ["python", "app.py"]
