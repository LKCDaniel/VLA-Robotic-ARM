# Bend Stick Demo

Within an industrial context, a subtask on a production line requires the robot arm to bend an elastic stick on the phone. There's a chance that the stick escapes during the motion of pushing it to the side.

In this project, I trained a simple multi-task VLA model that is capable of end-to-end robot-arm control. It takes in pure visual input, and output the robot arm's movement, as well as a task-completion classification.

## Simulation Scene Description

The first phase of this task is to pretrain the model in a virtual 3D environment. The stick has a random chance of escaping from the robot's rod.

## Setup Environment

```bash
# Python 3.11.0
python --version

# create python virtual environment
# actually newer python versions would also work
py -3.11 -m venv venv

# activate python virtual environment on windows
source venv/Script/activate

# on ubuntu
# source venv/bin/activate

# install required packages, here we install blender as a python module.
# See https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html
# 
bash install.sh
```

## Generate Synthesis Data
We choose blender rather than packages like pyvista just because blender use more modern 
graphic API OpenGL, not that modern, but still better than VTK used by pyvista.
```bash
python record_episode.py --episode_to 1000 --resolution 224
```

Then prepare the train and validate dataset.
```bash
python prepare_dataset.py --dataset_type "train" --episode_from 0 --episode_to 400
python prepare_dataset.py --dataset_type "validate" --episode_from 400 --episode_to 500
```

## Training
```bash
python train.py
```

## Test Online
Here "online" means the deep learning model interacts dynamically with the synthesis scene.
You need to adjust the parameters. (very simple)
```bash
python test_online.py -h
```

## Turn Images into Video
For better visualization
```bash
python generate_video.py
```

## Sample of online test data, from three cameras of different view
![camera_1]()
![camera_2]()
![camera_3]()
