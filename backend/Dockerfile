# Using a python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the requirements.txt file
COPY backend/requirements.txt .

# Print a message to acknowledge the presence of requirements.txt
RUN echo "Requirements.txt file detected: $(cat requirements.txt)"

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py from the root directory to the current working directory
COPY backend/app.py .

# Make port 5000 available to the container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
