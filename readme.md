# Overview

The project is to show how to train a deep learning model to complete a two_stage task. The first stage is to reach and pick a cubic object on the ground. The second task is then carry the object to the target place.

We simulate the task in a virtual 3D environment. The robotic arm is simulated by a red square column. The object to be caught is simulated as a blue cube. The target place is green-colored square area on the ground.

## Setup Environment

```bash
# Python 3.11.0
python --version

# create python virtual environment
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
We choose blender rather than packages like pyvista just because blender use more modern graphic API OpenGL, not that modern, but still better than VTK used by pyvista.
```bash
python record_episode.py --num_episodes 500 --resolution 128
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
```bash
python test_online.py
```

## Turn Images into Video
For better visualization
```bash
python generate_video.py
```

## Sample of online test data, from three cameras of different view
