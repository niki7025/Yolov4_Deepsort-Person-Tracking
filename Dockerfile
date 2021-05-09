FROM nvcr.io/nvidia/l4t-tensorflow:r32.5.0-tf2.3-py3
# tensorflow deleted from requirements
RUN echo "Build our Container based on L4T Tensorflow"
RUN nvcc --version

WORKDIR /app

COPY . ./

# Needed for accessing Jetpack 4.4
COPY  /docker-requirements/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
COPY  /docker-requirements/jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

RUN apt-get update && \ 
    apt-get install -y libopencv-python libboost-python-dev libboost-thread-dev && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    zlib1g-dev \
    zip \
    libjpeg8-dev && \ 
    rm -rf /var/lib/apt/lists/*

RUN pip3 install -U \
    pip \
    setuptools \
    wheel && \
    pip3 install \
    -r requirements.txt \
    && \
    rm -rf ~/.cache/pip


# ============================================
# END

# INSTALL OPENCV
# RUN echo "deb https://repo.download.nvidia.com/jetson/common r32.4 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends \
#     libopencv-python \
#     && rm /etc/apt/sources.list.d/nvidia-l4t-apt-source.list \
#     && rm -rf /var/lib/apt/lists/*   


# --------------------------------------
# Test code 
# -------------------------------------

# CMD ["python3", "object_tracker.py"]

# COPY  nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# COPY jetson-ota-public.asc /etc/apt/trusted.gpg.d/jetson-ota-public.asc

# -----------------
# INSTALL TENSORFLOW
# -----------------

# ARG HDF5_DIR="/usr/lib/aarch64-linux-gnu/hdf5/serial/"
# ARG MAKEFLAGS=-j6
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     gfortran \
#     build-essential \
#     liblapack-dev \ 
#     libblas-dev \
#     libhdf5-serial-dev \
#     hdf5-tools \
#     libhdf5-dev \
#     zlib1g-dev \
#     zip \
#     libjpeg8-dev \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip3 install setuptools Cython wheel
# RUN pip3 install h5py==2.10.0 --verbose
# RUN pip3 install future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11 --verbose
# RUN pip3 install numpy --verbose

# ARG TENSORFLOW_URL=https://nvidia.box.com/shared/static/rummpy6q1km1wivomalpkwt2jy28mndf.whl 
# ARG TENSORFLOW_WHL=tensorflow-1.15.2+nv-cp36-cp36m-linux_aarch64.whl

# RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${TENSORFLOW_URL} -O ${TENSORFLOW_WHL} && \
#     pip3 install ${TENSORFLOW_WHL} --verbose && \
#     rm ${TENSORFLOW_WHL}

# FROM continuumio/miniconda3


# FROM continuumio/anaconda3

# ADD /src/environment.yml /src/environment.yml

# RUN conda env create -f /src/environment.yml

# ENV PATH /opt/conda/envs/mro_env/bin:$PATH
# RUN /bin/bash -c "source activate mro_env \
#     && conda config --add channels conda-forge \
#     && conda install Jupiter \
#     && conda env list"

# # Create the environment:
# COPY environment.yml .
# RUN conda env create -f environment.yml

# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# # Make sure the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"

# # The code to run when container is started:
# COPY run.py .
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "run.py"]


# docker pull continuumio/anaconda3
# docker run -i -t continuumio/anaconda3 /bin/bash


# RUN conda env create -f conda-gpu.yml
# RUN conda activate yolov4-gpu


# RUN pip3 install -r requirements.txt

# RUN pip3 install -U \
#     pip \
#     setuptools \
#     wheel && \
#     pip3 install \
#     -r requirements.txt \
#     && \
#     rm -rf ~/.cache/pip

# COPY . .

# CMD ["python3", "object_tracker.py"]
