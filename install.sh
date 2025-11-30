pip install numpy==1.26.3
pip install opencv-python==4.9.0.80
pip install matplotlib 3.10.7
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# find and download .whl from https://download.blender.org/pypi/bpy/
pip install bpy-4.5.1-cp311-cp311-win_amd64.whl
# or directly install as follows
# pip install bpy==4.5.1 --extra-index-url https://download.blender.org/pypi/

ffmpeg -i real_time_test/camera_1.mp4 -vf "fps=30,scale=512:-1:flags=lanczos" -loop 0 real_time_test/camera_1.gif
ffmpeg -i real_time_test/camera_2.mp4 -vf "fps=30,scale=512:-1:flags=lanczos" -loop 0 real_time_test/camera_2.gif
ffmpeg -i real_time_test/camera_3.mp4 -vf "fps=30,scale=512:-1:flags=lanczos" -loop 0 real_time_test/camera_3.gif