FROM python:3.10-slim AS python-base

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src /app/src
COPY model /app/model

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r /app/requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run /app/src/app.py