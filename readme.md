# NVIDIA Isaac Lab(with Octo Imitation Learning) for Digital Twin

![image1](https://github.com/user-attachments/assets/e0336b0b-7ed9-4735-958a-add1b9d8f89e)

## pre-install
ubuntu: 22.04
nvidia-driver: 550.127.08

### 1. install docker
```
$ sudo apt-get update
$ sudo apt-get install ca-certificates curl
$ sudo install -m 0755 -d /etc/apt/keyrings
$ sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
$ sudo chmod a+r /etc/apt/keyrings/docker.asc

$ echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
### 2. nvidia-docker2
```
$ sudo apt-get install -y nvidia-docker2
$ sudo systemctl restart docker
```

### 3. nvidia-container-toolkit
```
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \\n  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\n    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\n    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-container-toolkit
$ sudo nvidia-ctk runtime configure --runtime=docker
$ sudo systemctl restart docker
```

### 4. xhost
run `$ xhost + local:docker`

### 5. octo/local.env
```
WANDB_API_KEY={YOUR_API_KEY}
```

## Launch
run `$ docker compose -f docker-compose-{name}.yml up --build`

### exec
run `sudo docker exec -it {name} bash`
```
container> $ any command
ex)
isaac_lab> $ python mnt/script/collect_demonstrations_agv.py --num_demos 10 --enable_cameras
octo> $ python mnt/script/finetune_new_observation_action.py
```
