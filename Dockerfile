FROM python:3.10-slim AS pbase

WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

COPY src ./src
COPY model ./model
COPY data/samples ./data/samples



# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD streamlit run /app/src/app.py