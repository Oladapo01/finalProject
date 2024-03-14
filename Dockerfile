# Using a python file
FROM python:3.10-slim

# Se the working directory
WORKDIR /usr/src/app


# Copy the requirements.txt file from the context directory to the current WORKDIR
COPY requirements.txt .
COPY app .

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt


# Make port 8888 available to the container
EXPOSE 5000

# Define envirnoment variable
ENV NAME World

# Run app.py  when the container launches
CMD ["python", "app.py"]
