## Import all relevant libraries
import bpy
import numpy as np
import math as m
import random
import os
import numpy as np
from typing import Tuple
import glob
import argparse
import bmesh
import sys
import json
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"

fbx_path = argv[0]
output_path = argv[1]
if not os.path.exists(output_path): os.makedirs(output_path)
blink = "BL" if "BL" in output_path else "NB"
head_rotate = "HR" if "HR" in output_path else "NR"
background = "IMB" if "IMB" in output_path else "HDB" if "HDB" in output_path else "TEB"
light = "LL" if "LL" in output_path else "ML" if "ML" in output_path else "HL"

# ---------------------------------------------------------------------------------------------
    
def Vertex2ImageCoornidate(obj, vertices, cam):
    # get the boiding box if some vertices in obj under cam
    scene = bpy.context.scene
    # needed to rescale 2d coordinates
    render = scene.render
    render_scale = scene.render.resolution_percentage / 100
    res_x = render.resolution_x *render_scale
    res_y = render.resolution_y *render_scale
    # use generator expressions () or list comprehensions []
    mat = obj.matrix_world
    res = []
    rnd = lambda i: round(i)
    for v in vertices:
        vert = mat @ v.co
        coords_2d = world_to_camera_view(scene, cam, vert)
        x, y, distance_to_lens = coords_2d
        res.append(tuple((rnd(res_x*x), rnd(res_y-res_y*y))))
    return res
# ---------------------------------------------------------------------------------------------

working_dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir_path)

# import fbx model without pose
bpy.ops.import_scene.fbx(filepath = fbx_path, use_anim=False)

scene = bpy.context.scene
# clear animation
scene.animation_data_clear()
for o in scene.objects:
    o.animation_data_clear()

# elements
scene = bpy.data.scenes['Scene']

# resulution
bpy.context.scene.render.resolution_x = 350
bpy.context.scene.render.resolution_y = 350

# frames
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 13

# shape key & keyframe
# bpy.ops.action.interpolation_type(type='BEZIER')
scene.tool_settings.use_keyframe_insert_auto = True

# if head_rotate == "HR":

obj_name = "Armature" # "Kevin"

# head rotate
headbone = bpy.data.objects[obj_name].pose.bones["CC_Base_NeckTwist02"] # This is for CC1 model, in CC4 is Kevin
headbone.rotation_mode = 'XYZ'

headbone.rotation_euler = (random.uniform(-2, 2)*m.pi/180, random.uniform(-2, 2)*m.pi/180, random.uniform(-2, 2)*m.pi/180)
headbone.keyframe_insert(data_path="rotation_euler", frame = 1)

headbone.rotation_euler = (random.uniform(-10, 10)*m.pi/180, random.uniform(-2, 2)*m.pi/180, random.uniform(-2, 2)*m.pi/180)
headbone.keyframe_insert(data_path="rotation_euler", frame = 13)

# eye size
me = bpy.data.objects["CC_Base_Body"].data
initial_blink_degree = random.uniform(0, 0.35)
final_blink_degree = random.uniform(0, 0.35)

me.shape_keys.key_blocks["Eye_Blink_L"].value = initial_blink_degree
me.shape_keys.key_blocks["Eye_Blink_L"].keyframe_insert(data_path="value", frame = 1)
me.shape_keys.key_blocks["Eye_Blink_L"].value = final_blink_degree
me.shape_keys.key_blocks["Eye_Blink_L"].keyframe_insert(data_path="value", frame = 12)

me.shape_keys.key_blocks["Eye_Blink_R"].value = initial_blink_degree
me.shape_keys.key_blocks["Eye_Blink_R"].keyframe_insert(data_path="value", frame = 1)

me.shape_keys.key_blocks["Eye_Blink_R"].value = final_blink_degree
me.shape_keys.key_blocks["Eye_Blink_R"].keyframe_insert(data_path="value", frame = 12)

# eye blink
if blink == "BL":
    close_frame = random.randint(4, 8)

    me.shape_keys.key_blocks["Eye_Blink_L"].value = 1
    me.shape_keys.key_blocks["Eye_Blink_L"].keyframe_insert(data_path="value", frame = close_frame)

    me.shape_keys.key_blocks["Eye_Blink_R"].value = 1
    me.shape_keys.key_blocks["Eye_Blink_R"].keyframe_insert(data_path="value", frame = close_frame)

# camera
cam = bpy.data.objects["Camera"]

# change camera
cam.location = (random.uniform(-0.8, 0.8), -1.2, random.uniform(1.2, 2.1))

# random background
if background == "IMB" or background == "TEB":
    scene.render.film_transparent = True
    cam.data.show_background_images = True

    if background == "IMB":
        folder = r"**\Background\COCO-train2017" # Replace ** with your path
        image_paths = glob.glob(os.path.join(folder, "*.jpg"))
    else:
        image_paths = glob.glob(r"**\Background\Describable Textures Dataset (DTD)\images\*\*.jpg") # Replace ** with your path
    
    filepath = random.choice(image_paths)
    img = bpy.data.images.load(filepath)
    # create light
    # Create light datablock
    light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
    if light == "LL":
        light_data.energy = random.randint(90, 100)
    elif light == "ML":
        light_data.energy = random.randint(100, 200)
    elif light == "HL":
        light_data.energy = random.randint(200, 300)
    # Create new object, pass the light data 
    light_object = bpy.data.objects.new(name="my-light", object_data=light_data)
    # Link object to collection in context
    bpy.context.collection.objects.link(light_object)
    # Change light position
    light_object.location = (random.uniform(-1.5, 1.5), random.uniform(-1, -2), random.uniform(0, 4))
    
    tree = bpy.context.scene.node_tree
    image_node = tree.nodes.new(type='CompositorNodeImage')
    image_node.image = img

    links = tree.links
    link = links.new(image_node.outputs[0], tree.nodes['缩放'].inputs[0])

elif background == "HDB": # HDRI Background
    C = bpy.context
    scn = C.scene
    C.scene.render.film_transparent = False
    # Get the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes
    # Clear all nodes
    tree_nodes.clear()
    # Add Background node
    node_background = tree_nodes.new(type='ShaderNodeBackground')
    # Add Environment Texture node
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    # Load and assign the image to the node property
    hdr_paths = glob.glob(r"D:\01-Research\2022-07-05-Eye-State-and-Blink-Detection\Blender模拟数据\3-数字人生成眨眼\Background\HDRI\*.exr")
    node_environment.image = bpy.data.images.load(random.choice(hdr_paths))
    node_environment.location = -300,0
    # Add Output node
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200,0

    if light == "LL":
        node_background.inputs[1].default_value = random.uniform(0, 3) * 0.1
    elif light == "ML":
        node_background.inputs[1].default_value = random.uniform(3, 6) * 0.1
    elif light == "HL":
        node_background.inputs[1].default_value = random.uniform(6, 10) * 0.1

    # Link all nodes
    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

# ---------------------------------------------------------------------------------------------

# annnotation
obj_body = bpy.data.objects['CC_Base_Body']
obj_eye = bpy.data.objects['CC_Base_Eye']

depgraph = bpy.context.evaluated_depsgraph_get()
annos = {}
for frame in range(scene.frame_start, scene.frame_end + 1):
    bpy.context.scene.frame_set(frame)
    blink_strength = bpy.data.shape_keys["Key"].key_blocks["Eye_Blink_L"].value

    # define new bmesh object:
    bm = bmesh.new()
    # read the evaluated mesh data into the bmesh   object:
    bm.from_object( obj_body, depgraph )
    bm.verts.ensure_lookup_table()
    L_Eye_Left, L_Eye_Right, R_Eye_Left, R_Eye_Right = bm.verts[5078], bm.verts[830], bm.verts[6125], bm.verts[6131]
    L_Eye_Left, L_Eye_Right, R_Eye_Left, R_Eye_Right = Vertex2ImageCoornidate(obj_body, [L_Eye_Left, L_Eye_Right, R_Eye_Left, R_Eye_Right], cam)

    bm = bmesh.new()
    bm.from_object( obj_eye, depgraph )
    bm.verts.ensure_lookup_table()
    L_Eye_Center, R_Eye_Center = bm.verts[475], bm.verts[338]
    L_Eye_Center, R_Eye_Center = Vertex2ImageCoornidate(obj_eye, [L_Eye_Center, R_Eye_Center], cam)

    anno = {
            "L_Eye_Left": L_Eye_Left,
            "L_Eye_Right": L_Eye_Right,
            "L_Eye_Center": L_Eye_Center,
            "R_Eye_Left": R_Eye_Left,
            "R_Eye_Right": R_Eye_Right,
            "R_Eye_Center": R_Eye_Center,
            "Blink_Strength": blink_strength
        }
    
    annos[str(frame).zfill(2)] = anno
    scene.render.filepath = output_path + str(frame).zfill(2)
    bpy.ops.render.render(write_still=True)

with open('{}/annotations.json'.format(output_path), 'w') as file:
    file.write(json.dumps(annos, indent=4)) # use `json.loads` to do the reverse

