import bpy
from mathutils import Vector
from math import radians, degrees
import math
import random
from macro import *
from enum import Enum
import os
import imageio


def set_camera_lookat(camera, target: Vector):
    direction = target - camera.location
    quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = quat.to_euler()


class SimScene:
    # stick angle: 
    # initialized at 120 degrees, Touched to move to 180 degrees
    # if stick is Free, it escapes at a random angle between stick_free_range (percentage%)

    def __init__(self, resolution, board_init_x, board_init_y,
                 robot_arm_init_x, robot_arm_init_y, robot_arm_init_z,
                 will_stick_free:bool, free_angle_percentage,
                 sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density):
        assert free_angle_percentage < 1 and free_angle_percentage >= 0.4, "free_angle_percentage must be within 1 and 0.4"
        
        if DEBUG_MODE:
            print(f'parameters: will_stick_free={will_stick_free}, free_angle_percentage={free_angle_percentage}')
        
        self.clear_scene()
        self.scene = bpy.context.scene
        self.board_init_x = board_init_x
        self.board_init_y = board_init_y
        self.robot_arm_init_x = robot_arm_init_x
        self.robot_arm_init_y = robot_arm_init_y
        self.robot_arm_init_z = robot_arm_init_z
        self.stick = self._create_lying_stick(board_init_x+4, board_init_y, 0)
        self.robot_arm = self._create_robot_arm(robot_arm_init_x, robot_arm_init_y, robot_arm_init_z)
        self._create_floor(board_init_x, board_init_y)
        self._create_lights(sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density)
        # how about 30? currently it is a little too far
        self.camera_1 = self._create_camera(camera_name="Camera1", camera_position=(30, 0, 20), camera_lookat=(0, 0, 5))
        self.camera_2 = self._create_camera(camera_name="Camera2", camera_position=(0, 30, 20), camera_lookat=(0, 0, 5))
        self.camera_3 = self._create_camera(camera_name="Camera3", camera_position=(0, 0, 30), camera_lookat=(0, 0, 0))
        # self.scene.camera = self.camera_1 # debug
        self._set_rendering(resolution)

        self.stick_state = self.StickState.Stationary
        self.will_stick_free = will_stick_free
        self.free_angle_degrees = free_angle_percentage * 60 + 120
        self.stick_angularV = 0
        self.stick_angularA = 0
        self.stick_touch_d = 5
        self.thickness_angle = math.asin(2*0.15/self.stick_touch_d) # adjust for stick radius & touch distance
        # 0.06 rad or 3.44 degrees

        self.robot_arm_state = self.RobotArmState.Horrizontal
    
    @staticmethod
    def clear_scene():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        # 清理残余数据
        for block in bpy.data.meshes:
            bpy.data.meshes.remove(block, do_unlink=True)
        for block in bpy.data.materials:
            bpy.data.materials.remove(block, do_unlink=True)
        for block in bpy.data.images:
            if block.name != "Render Result":
                bpy.data.images.remove(block, do_unlink=True)

    class StickState(Enum):
        Stationary = 0
        Touched = 1
        Free = 2

    class RobotArmState(Enum):
        Upward = 0
        Horrizontal = 1
        Downward = 2
        Touching = 3
        # Waiting = 4

    def _create_floor(self, init_x, init_y):
        bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(init_x, init_y, 0))
        floor = bpy.context.active_object
        floor.name = "GrayFloor"
        floor.scale = (0.5, 1, 0)

        mat_floor = bpy.data.materials.new(name="GrayMaterial")
        mat_floor.use_nodes = True
        bsdf_floor = mat_floor.node_tree.nodes["Principled BSDF"]
        bsdf_floor.inputs['Base Color'].default_value = (0.8, 0.8, 0.5, 1)
        bsdf_floor.inputs['Roughness'].default_value = 0.8
        bsdf_floor.inputs['Metallic'].default_value = 0.1
        floor.data.materials.append(mat_floor)

    def _create_lying_stick(self, init_x, init_y, init_z):
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=0.15, depth=6, location=(0,0,0))
        stick = bpy.context.active_object
        stick.name = "Stick"

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')    
        bpy.ops.object.mode_set(mode='OBJECT') # Switch to Object mode to read vertex selection/coordinates
        bpy.context.scene.cursor.location = (0,0,-3)
            
        # Set Origin to Cursor
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

        stick.rotation_euler = (0, math.radians(80), math.radians(120))
        stick.location = (init_x, init_y, init_z)

        # Material
        mat = bpy.data.materials.new(name="StickDark")
        mat.use_nodes = True
        bsdf_stick = mat.node_tree.nodes["Principled BSDF"]
        bsdf_stick.inputs['Base Color'].default_value = (0.2, 0.2, 0.2, 1)
        bsdf_stick.inputs['Roughness'].default_value = 0.8
        bsdf_stick.inputs['Metallic'].default_value = 0.4
        stick.data.materials.append(mat)
        
        return stick

    def _create_robot_arm(self, init_x=0, init_y=0, init_z=3):
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
        arm_base = bpy.context.active_object
        arm_base.name = "RobotArm"
        
        bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=4, location=(0, 0, -2.5))
        stick = bpy.context.active_object
        mat_robot_stick = bpy.data.materials.new(name="RedMaterial")
        mat_robot_stick.use_nodes = True
        bsdf_robot_arm = mat_robot_stick.node_tree.nodes["Principled BSDF"]
        bsdf_robot_arm.inputs['Base Color'].default_value = (1.0, 0.2, 0.2, 1)
        bsdf_robot_arm.inputs['Roughness'].default_value = 0.4
        bsdf_robot_arm.inputs['Metallic'].default_value = 1.0
        stick.data.materials.append(mat_robot_stick)
        arm_base.data.materials.append(mat_robot_stick)
        
        # join both objects
        bpy.ops.object.select_all(action='DESELECT')
        stick.select_set(True)
        arm_base.select_set(True)
        bpy.context.view_layer.objects.active = arm_base
        bpy.ops.object.join()

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT') # Switch to Object mode to riably read vertex selection/coordinates
        bpy.context.scene.cursor.location = (0, 0, -4.5)
    
        # Set Origin to Cursor
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        
        arm_base.location = (init_x, init_y, init_z)
        
        return arm_base


    def _create_lights(self, sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density):
        # ---------------------- 添加光源 ----------------------
        # 1. 强阳光（关键光）
        bpy.ops.object.light_add(type='SUN', location=(1000, -1000, 1000))
        sun = bpy.context.object
        sun.data.energy = sun_density
        sun.data.angle = 0
        sun.rotation_euler = (sun_rx_radian, sun_ry_radian, 0)

        # 2. 环境光（世界背景）
        world = self.scene.world
        world.use_nodes = True
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value = (bg_r, bg_g, bg_b, 1)
        bg.inputs[1].default_value = bg_density  # 强度降低，避免过曝

        # 3. 开启 Cycles 自带的接触阴影（让立方体贴地部分更黑，超级真实）
        self.scene.cycles.use_fast_gi = True
        self.scene.render.film_transparent = False

    def _set_rendering(self, resolution):
        self.scene.render.engine = 'CYCLES'
        self.scene.render.use_persistent_data = True

        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        cprefs.refresh_devices()
        cprefs.compute_device_type = 'OPTIX'
        self.scene.cycles.device = 'GPU'

        self.scene.cycles.samples = 64
        self.scene.cycles.denoiser = 'OPTIX'
        self.scene.cycles.use_denoising = True
        self.scene.view_layers["ViewLayer"].cycles.use_denoising = True
        self.scene.cycles.denoiser = 'OPTIX'
        self.scene.cycles.denoising_use_gpu = True

        self.scene.render.resolution_x = resolution
        self.scene.render.resolution_y = resolution
        self.scene.render.image_settings.file_format = 'PNG'

    def _create_camera(self, camera_name, camera_position, camera_lookat):
        # ---------------------- 创建相机并看向原点 ----------------------
        bpy.ops.object.camera_add(location=camera_position)
        camera = bpy.context.active_object
        camera.name = camera_name
        set_camera_lookat(camera, Vector(camera_lookat))  # 稍微看向立方体中心
        return camera

    def _shot(self, output_path):
        self.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True, animation=False)

    def shot_1(self, output_path):
        self.scene.camera = self.camera_1
        self._shot(output_path)
        self.scene.camera = None

    def shot_2(self, output_path):
        self.scene.camera = self.camera_2
        self._shot(output_path)
        self.scene.camera = None

    def shot_3(self, output_path):
        self.scene.camera = self.camera_3
        self._shot(output_path)
        self.scene.camera = None

    def _move_robot_arm(self, dx, dy, dz):
        self.robot_arm.location[0] += dx
        self.robot_arm.location[1] += dy
        self.robot_arm.location[2] += dz        

    def _stick_touch_position(self, bias_angle=0):
        angle = self.stick.rotation_euler[2] + bias_angle - self.thickness_angle
        x = self.stick.location[0] + self.stick_touch_d * math.cos(angle)
        y = self.stick.location[1] + self.stick_touch_d * math.sin(angle)
        return Vector((x, y, 0))
    
    def _local_arm_angle(self):
        arm_local_pos = (self.robot_arm.location - self.stick.location)
        return math.atan2(arm_local_pos[1], arm_local_pos[0])
        
    def is_stick_inplace(self) -> bool:
        return degrees(self.stick.rotation_euler[2]) % 360 >= 180
    
    def touch_score(self) -> float:
        # return how close the stick is being touched, 0~1
        if self.robot_arm_state == self.RobotArmState.Touching:
            return max(0, 1 - (self.stick.rotation_euler[2] - self._local_arm_angle() - self.thickness_angle) / radians(30))
        return 0.

    # return true if the stick is in place, else false. Also make movements of course
    def update_frame(self, dx, dy, dz) -> bool:
        self._move_robot_arm(dx, dy, dz)
        
        if DEBUG_MODE:
            print(f'will be free: {self.will_stick_free}, stick state: {self.stick_state}, robot arm state: {self.robot_arm_state}, robot arm pos: {self.robot_arm.location}, stick angle: {degrees(self.stick.rotation_euler[2])}')
        
        if self.stick_state == self.StickState.Stationary:
            pass
        
        elif self.stick_state == self.StickState.Touched:
            self.stick.rotation_euler[2] = self._local_arm_angle() + self.thickness_angle
            if self.will_stick_free and degrees(self.stick.rotation_euler[2]) >= self.free_angle_degrees:
                self.stick_state = self.StickState.Free
                self.will_stick_free = False

        else: # Free state
            self.robot_arm_state = self.RobotArmState.Upward
            self.stick_angularA = (radians(120) - self.stick.rotation_euler[2]) / (math.pi*2/6) * 0.25 # elastic force
            self.stick_angularV += self.stick_angularA
            # damping
            self.stick_angularV *= 0.9
            if self.stick_angularV > 0:
                self.stick_angularV = max(0, self.stick_angularV - 0.02)
            else:
                self.stick_angularV = min(0, self.stick_angularV + 0.02)
            
            self.stick.rotation_euler[2] += self.stick_angularV

            if self.stick_angularV == 0 and not self.is_stick_inplace():
                self.stick_state = self.StickState.Stationary

        return self.is_stick_inplace() 
    
    def next_robot_arm_movement(self) -> (Vector):
        # if the height is not enough, move Upward first
        if self.robot_arm_state == self.RobotArmState.Upward:
            if self.robot_arm.location[2] >= 2:
                self.robot_arm_state = self.RobotArmState.Horrizontal
            else:
                return Vector((0, 0, 1))
        
        # high enough, move horrizontally to the stick touch position, with some gap space
        if self.robot_arm_state == self.RobotArmState.Horrizontal:
            distance = self._stick_touch_position(bias_angle = radians(-15)) - self.robot_arm.location
            distance.z = 0
            if distance.length < ROBOT_ARM_SPEED * 1.25:
                self.robot_arm_state = self.RobotArmState.Downward
            else:
                return distance.normalized()

        # reached the position, move Downward to level with the stick
        if self.robot_arm_state == self.RobotArmState.Downward:
            if self.robot_arm.location[2] <= 0.5:
                self.robot_arm_state = self.RobotArmState.Touching
            else:
                return Vector((0, 0, -1))
        
        # Touching the stick now
        if self.is_stick_inplace():
            return Vector((0, 0, 0))
        
        # push
        local_angle = self._local_arm_angle()
        if (local_angle + self.thickness_angle)%(2*math.pi) > self.stick.rotation_euler[2]%(2*math.pi):
            self.stick.rotation_euler[2] = local_angle + self.thickness_angle
            self.stick_state = self.StickState.Touched

        action = self._stick_touch_position(bias_angle=radians(5)) - self.robot_arm.location
        action.z = 0
        return action.normalized()
        

        # # Touching the stick now
        # if self.robot_arm_state == self.RobotArmState.Touching:
        #     if self.is_stick_inplace():
        #         self.robot_arm_state = self.RobotArmState.Waiting
        #     else:
        #         local_angle = self._local_arm_angle()
        #         if (local_angle + self.thickness_angle)%(2*math.pi) > self.stick.rotation_euler[2]%(2*math.pi):
        #             self.stick.rotation_euler[2] = local_angle + self.thickness_angle
        #             self.stick_state = self.StickState.Touched
            
        #         return (self._stick_touch_position(bias_angle=radians(5)) - self.robot_arm.location).normalized()
        
        # # Waiting for stick to be in place
        # if self.stick_state == self.StickState.Stationary:
        #     self.robot_arm_state = self.RobotArmState.Upward
        # return Vector((0, 0, 0))




# for debug purposes

if __name__ == "__main__":
    # Define your required macro variables for this example to run
    FLOOR_SIZE = 20

    def clear_scene():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)
        
    clear_scene()

    output_dir = "./blender_sim_output/2/" # debug
    os.makedirs(output_dir, exist_ok=True)

    sun_rx_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
    sun_ry_radian = random.uniform(math.pi / 8, 7 * math.pi / 8)
    sun_density = 6.0
    bg_r = 1.0
    bg_g = 1.0
    bg_b = 1.0
    bg_density = 0.2
    
    # 1. Initialize the scene
    scene = SimScene(
        resolution=512, board_init_x=0, board_init_y=0,
        robot_arm_init_x=-1, robot_arm_init_y=0, robot_arm_init_z=3,
        will_stick_free=True, free_angle_degrees=random.uniform(130, 170),
        sun_rx_radian=radians(60), sun_ry_radian=radians(30), sun_density=3.0,
        bg_r=bg_r, bg_g=bg_g, bg_b=bg_b, bg_density=bg_density
    )
    
    # 2. Run the simulation for up to n frames
    
    total_frames = 300
    
    # 1. Setup Animation Parameters
    scene.set_animation_params(start_frame=1, end_frame=total_frames, fps=30)

    # 2. Keyframe initial positions
    scene._keyframe_object(scene.stick, 1)
    scene._keyframe_object(scene.robot_arm, 1)

    # 3. Simulation Loop
    final_frame = total_frames
    for frame in range(2, total_frames + 1):
        # Set the current frame
        scene.scene.frame_current = frame
        
        # Calculate the robot arm's next movement vector
        # Assuming the movement vector from next_robot_arm_movement is a "per-frame" step
        movement_vector = ROBOT_ARM_SPEED * scene.next_robot_arm_movement()
        dx, dy, dz = movement_vector[0], movement_vector[1], movement_vector[2]
        
        # Update the scene state (move arm, rotate stick, check state)
        stick_in_place = scene.update_frame(dx, dy, dz)
        
        # Insert keyframes for the objects' new positions/rotations
        scene._keyframe_object(scene.stick, frame)
        scene._keyframe_object(scene.robot_arm, frame)
        
        print(f'frame {frame}: Robot Arm pos: {scene.robot_arm.location}, Stick angle: {degrees(scene.stick.rotation_euler[2])}')

        if stick_in_place:
            print(f"Stick is in place at frame {frame}. Stopping simulation.")
            scene.scene.frame_end = frame # Optionally stop the animation here
            final_frame = frame
            break

    # 4. Set Output for Rendering (Sequence of images)
    # The output path should be set to save an image for every frame.
    scene.scene.render.filepath = os.path.join(output_dir, "frame_") 
    
    # Optional: Render the animation (Sequence of PNGs)
    bpy.ops.render.render(animation=True)
    
    # If you want a video file directly:
    video_path = os.path.join(output_dir, "output_video.mp4")
    images = []
    for i in range(1, final_frame + 1):
        fname = f"frame_{i:04d}.png"
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            images.append(imageio.imread(fpath))
    if images:
        imageio.mimsave(video_path, images, fps=fps, codec='libx264', format='FFMPEG')
        print("Video stitching successful!")
    
    print(f"Animation keyframes set from frame 1 to {scene.scene.frame_end}.")
    print(f"Output path set to: {scene.scene.render.filepath}####.png")
    
    print("Simulation setup complete.")
    
    
    
    
    
    
    
    
    
# import bpy
# from mathutils import Vector
# from macro import *


# def set_camera_lookat(camera, target: Vector):
#     direction = target - camera.location
#     quat = direction.to_track_quat('-Z', 'Y')
#     camera.rotation_euler = quat.to_euler()


# class SimScene:
#     def __init__(self, resolution, object_init_x, object_init_y,
#                  robot_arm_init_x, robot_arm_init_y, robot_arm_init_z,
#                  sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density):
#         self.clear_scene()
#         self.scene = bpy.context.scene
#         self.object_init_x = object_init_x
#         self.object_init_y = object_init_y
#         self.robot_arm_init_x = robot_arm_init_x
#         self.robot_arm_init_y = robot_arm_init_y
#         self.robot_arm_init_z = robot_arm_init_z
#         self.object = self.create_object(object_init_x, object_init_y)
#         self.robot_arm = self.create_robot_arm(robot_arm_init_x, robot_arm_init_y, robot_arm_init_z)
#         self.create_floor()
#         self.bin_place = self.create_bin_place()
#         self.create_lights(sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density)
#         self.camera_1 = self.create_camera(camera_name="Camera1", camera_position=(35, 0, 20), camera_lookat=(0, 0, 5))
#         self.camera_2 = self.create_camera(camera_name="Camera2", camera_position=(0, 35, 20), camera_lookat=(0, 0, 5))
#         self.camera_3 = self.create_camera(camera_name="Camera2", camera_position=(0, 0, 35), camera_lookat=(0, 0, 0))
#         self.set_rendering(resolution)

#     @staticmethod
#     def clear_scene():
#         bpy.ops.object.select_all(action='SELECT')
#         bpy.ops.object.delete(use_global=False)
#         # 清理残余数据
#         for block in bpy.data.meshes:
#             bpy.data.meshes.remove(block, do_unlink=True)
#         for block in bpy.data.materials:
#             bpy.data.materials.remove(block, do_unlink=True)
#         for block in bpy.data.images:
#             if block.name != "Render Result":
#                 bpy.data.images.remove(block, do_unlink=True)

#     @staticmethod
#     def create_object(init_x, init_y):
#         # ---------------------- 创建蓝色立方体 ----------------------
#         bpy.ops.mesh.primitive_cube_add(size=OBJECT_SIZE, location=(init_x, init_y, OBJECT_INIT_Z))
#         cubic_object = bpy.context.active_object
#         cubic_object.name = "BlueCube"

#         mat_cubic_object = bpy.data.materials.new(name="BlueMaterial")
#         mat_cubic_object.use_nodes = True
#         bsdf_cubic_object = mat_cubic_object.node_tree.nodes["Principled BSDF"]
#         bsdf_cubic_object.inputs['Base Color'].default_value = (0.0, 0.4, 1.0, 1)
#         bsdf_cubic_object.inputs['Roughness'].default_value = 0.4
#         bsdf_cubic_object.inputs['Metallic'].default_value = 1.0
#         cubic_object.data.materials.append(mat_cubic_object)
#         return cubic_object

#     @staticmethod
#     def create_robot_arm(init_x, init_y, init_z):
#         # ---------------------- 创建红色夹爪 ----------------------
#         bpy.ops.mesh.primitive_cube_add(size=OBJECT_SIZE, location=(init_x, init_y, init_z))
#         robot_arm = bpy.context.active_object
#         robot_arm.name = "RobotArm"
#         robot_arm.scale = (1, 1, ROBOT_ARM_HEIGHT / OBJECT_SIZE)

#         mat_robot_arm = bpy.data.materials.new(name="RedMaterial")
#         mat_robot_arm.use_nodes = True
#         bsdf_robot_arm = mat_robot_arm.node_tree.nodes["Principled BSDF"]
#         bsdf_robot_arm.inputs['Base Color'].default_value = (1.0, 0.2, 0.2, 1)
#         bsdf_robot_arm.inputs['Roughness'].default_value = 0.4
#         bsdf_robot_arm.inputs['Metallic'].default_value = 1.0
#         robot_arm.data.materials.append(mat_robot_arm)
#         return robot_arm

#     @staticmethod
#     def create_floor():
#         # ---------------------- 创建灰色地面 ----------------------
#         bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(0, 0, 0))
#         floor = bpy.context.active_object
#         floor.name = "GrayFloor"
#         floor.scale = (1, 1, 1)

#         mat_floor = bpy.data.materials.new(name="GrayMaterial")
#         mat_floor.use_nodes = True
#         bsdf_floor = mat_floor.node_tree.nodes["Principled BSDF"]
#         bsdf_floor.inputs['Base Color'].default_value = (0.8, 0.8, 0.5, 1)
#         bsdf_floor.inputs['Roughness'].default_value = 0.8
#         bsdf_floor.inputs['Metallic'].default_value = 0.1
#         floor.data.materials.append(mat_floor)
#         return floor

#     @staticmethod
#     def create_bin_place():
#         # ---------------------- 创建物体放置区域 ----------------------
#         bpy.ops.mesh.primitive_plane_add(size=1, location=BIN_LOCATION)
#         bin_place = bpy.context.active_object
#         bin_place.name = "BinPlace"
#         bin_place.scale = (BIN_SIZE, BIN_SIZE, 1)

#         mat_bin_place = bpy.data.materials.new(name="GreenMaterial")
#         mat_bin_place.use_nodes = True
#         bsdf_bin_place = mat_bin_place.node_tree.nodes["Principled BSDF"]
#         bsdf_bin_place.inputs['Base Color'].default_value = (0.0, 1.0, 0.0, 1)
#         bsdf_bin_place.inputs['Roughness'].default_value = 1.0
#         bsdf_bin_place.inputs['Metallic'].default_value = 0.0
#         bin_place.data.materials.append(mat_bin_place)
#         return bin_place

#     def create_lights(self, sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density):
#         # ---------------------- 添加光源 ----------------------
#         # 1. 强阳光（关键光）
#         bpy.ops.object.light_add(type='SUN', location=(1000, -1000, 1000))
#         sun = bpy.context.object
#         sun.data.energy = sun_density
#         sun.data.angle = 0
#         sun.rotation_euler = (sun_rx_radian, sun_ry_radian, 0)

#         # 2. 环境光（世界背景）
#         world = self.scene.world
#         world.use_nodes = True
#         bg = world.node_tree.nodes['Background']
#         bg.inputs[0].default_value = (bg_r, bg_g, bg_b, 1)
#         bg.inputs[1].default_value = bg_density  # 强度降低，避免过曝

#         # 3. 开启 Cycles 自带的接触阴影（让立方体贴地部分更黑，超级真实）
#         self.scene.cycles.use_fast_gi = True
#         self.scene.render.film_transparent = False

#     def set_rendering(self, resolution):
#         self.scene.render.engine = 'CYCLES'

#         prefs = bpy.context.preferences
#         cprefs = prefs.addons['cycles'].preferences
#         cprefs.refresh_devices()
#         cprefs.compute_device_type = 'OPTIX'
#         self.scene.cycles.device = 'GPU'

#         self.scene.cycles.samples = 64
#         self.scene.cycles.denoiser = 'OPTIX'
#         self.scene.cycles.use_denoising = True
#         self.scene.view_layers["ViewLayer"].cycles.use_denoising = True
#         self.scene.cycles.denoiser = 'OPTIX'
#         self.scene.cycles.denoising_use_gpu = True

#         self.scene.render.resolution_x = resolution
#         self.scene.render.resolution_y = resolution
#         self.scene.render.image_settings.file_format = 'PNG'

#     @staticmethod
#     def create_camera(camera_name, camera_position, camera_lookat):
#         # ---------------------- 创建相机并看向原点 ----------------------
#         bpy.ops.object.camera_add(location=camera_position)
#         camera = bpy.context.active_object
#         camera.name = camera_name
#         set_camera_lookat(camera, Vector(camera_lookat))  # 稍微看向立方体中心
#         return camera

#     def shot(self, output_path):
#         self.scene.render.filepath = output_path
#         bpy.ops.render.render(write_still=True, animation=False)

#     def shot_1(self, output_path):
#         self.scene.camera = self.camera_1
#         self.shot(output_path)
#         self.scene.camera = None

#     def shot_2(self, output_path):
#         self.scene.camera = self.camera_2
#         self.shot(output_path)
#         self.scene.camera = None

#     def shot_3(self, output_path):
#         self.scene.camera = self.camera_3
#         self.shot(output_path)
#         self.scene.camera = None

#     def move_robot_arm(self, dx, dy, dz):
#         self.robot_arm.location[0] += dx
#         self.robot_arm.location[1] += dy
#         self.robot_arm.location[2] += dz

#     def set_robot_arm_location(self, x, y, z):
#         self.robot_arm.location[0] = x
#         self.robot_arm.location[1] = y
#         self.robot_arm.location[2] = z

#     def move_object(self, dx, dy, dz):
#         self.object.location[0] += dx
#         self.object.location[1] += dy
#         self.object.location[2] += dz

#     def robot_arm_to_pick_object(self):
#         current_position = self.robot_arm.location
#         target_position = self.object.location.copy()
#         target_position[2] = OBJECT_SIZE + 0.5 * ROBOT_ARM_HEIGHT
#         return target_position - current_position

#     def robot_arm_to_place_object(self):
#         current_position = self.robot_arm.location
#         target_position = self.bin_place.location.copy()
#         target_position[2] = OBJECT_SIZE + 0.5 * ROBOT_ARM_HEIGHT
#         return target_position - current_position

