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
# RUN pip install isaacsim==4.2.0.2 isaacsim-extscache-physics==4.2.0.2 isaacsim-extscache-kit==4.2.0.2 isaacsim-extscache-kit-sdk==4.2.0.2 --extra-index-url https://pypi.nvidia.com
RUN pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit-sdk isaacsim-extscache-kit isaacsim-app --extra-index-url https://pypi.nvidia.com

RUN git clone https://github.com/isaac-sim/IsaacLab.git

# RUN pip install scipy==1.9.0
WORKDIR /home/sites/IsaacLab
RUN ./isaaclab.sh -i
# RUN pip uninstall skrl -y && pip install git+https://github.com/Toni-SM/skrl.git@develop
# RUN cd source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks && \
#     mkdir direct/agv && mkdir manager_based/agv
# RUN chmod +x isaaclab.sh
# RUN sh isaaclab.sh -i
# COPY run_finetune.sh ./octo/.
# COPY modifier/dataset.py ./octo/octo/data/dataset.py
# RUN chmod +x ./octo/run_finetune.sh

CMD ["bash"]