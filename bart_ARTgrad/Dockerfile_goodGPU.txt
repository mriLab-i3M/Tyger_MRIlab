# Use a CUDA development image that includes the CUDA toolkit and headers
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables for CUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Update the system and install Python and required tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    && apt-get clean

# Install CuPy with the appropriate CUDA version
RUN pip3 install cupy-cuda11x
RUN pip install numpy==1.26.4 scipy==1.10.1 matplotlib==3.7.1 mrd-python==2.0.0 

# Copy the application code into the container
WORKDIR /app
COPY scripts/ /app

# Set the default command to run the ART script
CMD ["python3"]