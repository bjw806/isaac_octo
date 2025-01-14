FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=0
ENV TERM=xterm

RUN apt-get update --fix-missing && \
    apt-get upgrade -y
RUN apt-get install -y \
    git curl cmake sudo build-essential python3-pip nvtop neovim \
    libgl1-mesa-glx libglx-mesa0 libxext6 libx11-6 libxrandr2 libxinerama1 libxcursor1 libgmp-dev libgmpxx4ldbl \
    libvulkan1 libvulkan-dev vulkan-tools \
    xvfb xorg zenity 

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /home/sites

RUN pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install "envlogger[tfds]"
RUN pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

RUN git clone https://github.com/isaac-sim/IsaacLab.git
RUN IsaacLab/isaaclab.sh -i

RUN git clone https://github.com/octo-models/octo.git
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -v -e octo/. && pip install -r octo/requirements.txt

RUN pip install jaxlib==0.4.20
RUN pip install scipy==1.9.0

COPY octo/modifier/dataset.py ./octo/octo/data/.
COPY octo/modifier/gym_wrappers.py ./octo/octo/utils/.
COPY octo/modifier/isaac_lab_wrapper.py ./octo/octo/utils/.

COPY docker_entrypoint.sh .
RUN chmod +x docker_entrypoint.sh

RUN useradd -u 1000 -m -s /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER user

CMD ["bash"]