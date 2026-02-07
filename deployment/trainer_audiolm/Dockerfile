# AWS Deep Learning Container
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-cpu-py310-ubuntu20.04-ec2

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
