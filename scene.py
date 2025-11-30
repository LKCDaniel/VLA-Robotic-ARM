import bpy
from mathutils import Vector
from macro import *


def set_camera_lookat(camera, target: Vector):
    direction = target - camera.location
    quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = quat.to_euler()


class SimScene:
    def __init__(self, resolution, object_init_x, object_init_y,
                 robot_arm_init_x, robot_arm_init_y, robot_arm_init_z,
                 sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density):
        self.clear_scene()
        self.scene = bpy.context.scene
        self.object_init_x = object_init_x
        self.object_init_y = object_init_y
        self.robot_arm_init_x = robot_arm_init_x
        self.robot_arm_init_y = robot_arm_init_y
        self.robot_arm_init_z = robot_arm_init_z
        self.object = self.create_object(object_init_x, object_init_y)
        self.robot_arm = self.create_robot_arm(robot_arm_init_x, robot_arm_init_y, robot_arm_init_z)
        self.create_floor()
        self.bin_place = self.create_bin_place()
        self.create_lights(sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density)
        self.camera_1 = self.create_camera(camera_name="Camera1", camera_position=(35, 0, 20), camera_lookat=(0, 0, 5))
        self.camera_2 = self.create_camera(camera_name="Camera2", camera_position=(0, 35, 20), camera_lookat=(0, 0, 5))
        self.camera_3 = self.create_camera(camera_name="Camera2", camera_position=(0, 0, 35), camera_lookat=(0, 0, 0))
        self.set_rendering(resolution)

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

    @staticmethod
    def create_object(init_x, init_y):
        # ---------------------- 创建蓝色立方体 ----------------------
        bpy.ops.mesh.primitive_cube_add(size=OBJECT_SIZE, location=(init_x, init_y, OBJECT_INIT_Z))
        cubic_object = bpy.context.active_object
        cubic_object.name = "BlueCube"

        mat_cubic_object = bpy.data.materials.new(name="BlueMaterial")
        mat_cubic_object.use_nodes = True
        bsdf_cubic_object = mat_cubic_object.node_tree.nodes["Principled BSDF"]
        bsdf_cubic_object.inputs['Base Color'].default_value = (0.0, 0.4, 1.0, 1)
        bsdf_cubic_object.inputs['Roughness'].default_value = 0.4
        bsdf_cubic_object.inputs['Metallic'].default_value = 1.0
        cubic_object.data.materials.append(mat_cubic_object)
        return cubic_object

    @staticmethod
    def create_robot_arm(init_x, init_y, init_z):
        # ---------------------- 创建红色夹爪 ----------------------
        bpy.ops.mesh.primitive_cube_add(size=OBJECT_SIZE, location=(init_x, init_y, init_z))
        robot_arm = bpy.context.active_object
        robot_arm.name = "RobotArm"
        robot_arm.scale = (1, 1, ROBOT_ARM_HEIGHT / OBJECT_SIZE)

        mat_robot_arm = bpy.data.materials.new(name="RedMaterial")
        mat_robot_arm.use_nodes = True
        bsdf_robot_arm = mat_robot_arm.node_tree.nodes["Principled BSDF"]
        bsdf_robot_arm.inputs['Base Color'].default_value = (1.0, 0.2, 0.2, 1)
        bsdf_robot_arm.inputs['Roughness'].default_value = 0.4
        bsdf_robot_arm.inputs['Metallic'].default_value = 1.0
        robot_arm.data.materials.append(mat_robot_arm)
        return robot_arm

    @staticmethod
    def create_floor():
        # ---------------------- 创建灰色地面 ----------------------
        bpy.ops.mesh.primitive_plane_add(size=FLOOR_SIZE, location=(0, 0, 0))
        floor = bpy.context.active_object
        floor.name = "GrayFloor"
        floor.scale = (1, 1, 1)

        mat_floor = bpy.data.materials.new(name="GrayMaterial")
        mat_floor.use_nodes = True
        bsdf_floor = mat_floor.node_tree.nodes["Principled BSDF"]
        bsdf_floor.inputs['Base Color'].default_value = (0.8, 0.8, 0.5, 1)
        bsdf_floor.inputs['Roughness'].default_value = 0.8
        bsdf_floor.inputs['Metallic'].default_value = 0.1
        floor.data.materials.append(mat_floor)
        return floor

    @staticmethod
    def create_bin_place():
        # ---------------------- 创建物体放置区域 ----------------------
        bpy.ops.mesh.primitive_plane_add(size=1, location=BIN_LOCATION)
        bin_place = bpy.context.active_object
        bin_place.name = "BinPlace"
        bin_place.scale = (BIN_SIZE, BIN_SIZE, 1)

        mat_bin_place = bpy.data.materials.new(name="GreenMaterial")
        mat_bin_place.use_nodes = True
        bsdf_bin_place = mat_bin_place.node_tree.nodes["Principled BSDF"]
        bsdf_bin_place.inputs['Base Color'].default_value = (0.0, 1.0, 0.0, 1)
        bsdf_bin_place.inputs['Roughness'].default_value = 1.0
        bsdf_bin_place.inputs['Metallic'].default_value = 0.0
        bin_place.data.materials.append(mat_bin_place)
        return bin_place

    def create_lights(self, sun_rx_radian, sun_ry_radian, sun_density, bg_r, bg_g, bg_b, bg_density):
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

    def set_rendering(self, resolution):
        self.scene.render.engine = 'CYCLES'

        self.scene.cycles.samples = 64
        self.scene.cycles.denoiser = 'OPTIX'
        self.scene.view_layers["ViewLayer"].cycles.use_denoising = True

        self.scene.render.resolution_x = resolution
        self.scene.render.resolution_y = resolution
        self.scene.render.image_settings.file_format = 'PNG'

    @staticmethod
    def create_camera(camera_name, camera_position, camera_lookat):
        # ---------------------- 创建相机并看向原点 ----------------------
        bpy.ops.object.camera_add(location=camera_position)
        camera = bpy.context.active_object
        camera.name = camera_name
        set_camera_lookat(camera, Vector(camera_lookat))  # 稍微看向立方体中心
        return camera

    def shot(self, output_path):
        self.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True, animation=False)

    def shot_1(self, output_path):
        self.scene.camera = self.camera_1
        self.shot(output_path)
        self.scene.camera = None

    def shot_2(self, output_path):
        self.scene.camera = self.camera_2
        self.shot(output_path)
        self.scene.camera = None

    def shot_3(self, output_path):
        self.scene.camera = self.camera_3
        self.shot(output_path)
        self.scene.camera = None

    def move_robot_arm(self, dx, dy, dz):
        self.robot_arm.location[0] += dx
        self.robot_arm.location[1] += dy
        self.robot_arm.location[2] += dz

    def set_robot_arm_location(self, x, y, z):
        self.robot_arm.location[0] = x
        self.robot_arm.location[1] = y
        self.robot_arm.location[2] = z

    def move_object(self, dx, dy, dz):
        self.object.location[0] += dx
        self.object.location[1] += dy
        self.object.location[2] += dz

    def robot_arm_to_pick_object(self):
        current_position = self.robot_arm.location
        target_position = self.object.location.copy()
        target_position[2] = OBJECT_SIZE + 0.5 * ROBOT_ARM_HEIGHT
        return target_position - current_position

    def robot_arm_to_place_object(self):
        current_position = self.robot_arm.location
        target_position = self.bin_place.location.copy()
        target_position[2] = OBJECT_SIZE + 0.5 * ROBOT_ARM_HEIGHT
        return target_position - current_position
