"""Shared MuJoCo scene: Panda + table + policy cameras, with game XML merged in.

Geom groups:
  0  physical game objects (visible to policy and player)
  2  robot visuals (Panda default)
  3  robot collision (hidden everywhere)
  4  game-only decoration, player view only, must be collision-free
  5  neutral lab backdrop, policy views only, collision-free
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

GAME1016 = Path(__file__).resolve().parents[2]
PANDA = GAME1016 / "assets" / "downloaded" / "panda"

TABLE_TOP_Z = 0.74
ROBOT_BASE = (-0.15, 0.0, TABLE_TOP_Z)
POLICY_GROUPS = (0, 1, 2, 5)
PLAYER_GROUPS = (0, 1, 2, 4)
POLICY_IMAGE = (256, 256)
PLAYER_IMAGE = (640, 400)


def look_at(position, target, up=(0, 0, 1)):
    """MuJoCo camera xyaxes string for a camera at `position` facing `target`."""
    position = np.asarray(position, float)
    forward = np.asarray(target, float) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    return " ".join(f"{v:.5f}" for v in (*right, *camera_up))


SCENE_CAMERA = {"pos": (1.10, -0.62, 1.38), "target": (0.42, 0.0, 0.76), "fovy": 52}
DEFAULT_PLAYER_CAMERA = {"pos": (-0.32, -0.78, 1.72), "target": (0.52, 0.02, 0.76), "fovy": 55}


def build_model_xml(objects_xml="", decor_xml="", assets_xml="", player_camera=None, model_name="arcade"):
    """Return the full model XML for one game.

    `objects_xml`: worldbody children with physics (group 0 default).
    `decor_xml`: player-only decoration; every geom is forced to group 4 without collision.
    `assets_xml`: extra <asset> children (materials, textures).
    """
    if not (PANDA / "panda.xml").exists():
        raise RuntimeError(f"Panda assets missing at {PANDA}")
    tree = ET.parse(PANDA / "panda.xml")
    root = tree.getroot()
    root.set("model", model_name)
    root.find("compiler").set("meshdir", str(PANDA / "assets"))
    root.find("option").attrib.update(timestep="0.002", gravity="0 0 -9.81", iterations="60",
                                      integrator="implicitfast")
    key = root.find("keyframe")
    if key is not None:
        root.remove(key)
    visual = ET.SubElement(root, "visual")
    visual.append(ET.fromstring('<global offwidth="800" offheight="600"/>'))
    visual.append(ET.fromstring('<quality shadowsize="2048"/>'))
    visual.append(ET.fromstring('<headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>'))
    asset = root.find("asset")
    if assets_xml.strip():
        for element in ET.fromstring("<asset>" + assets_xml + "</asset>"):
            asset.append(element)

    world = root.find("worldbody")
    world.find("body[@name='link0']").set("pos", " ".join(map(str, ROBOT_BASE)))
    # Gravity compensation on the arm so position targets are tracked without sag.
    for body in world.find("body[@name='link0']").iter("body"):
        body.set("gravcomp", "1")
    hand = world.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_site", pos="0 0 0.103", size="0.004", rgba="0 0 0 0")
    # Wrist camera looks along the fingers (+z of the hand), mounted beside the palm.
    ET.SubElement(hand, "camera", name="wrist", pos="0.07 0 0.02",
                  xyaxes="0 -1 0 -0.966 0 -0.259", fovy="78")
    for name in ("left_finger", "right_finger"):
        for geom in world.find(f".//body[@name='{name}']").findall("geom"):
            if geom.get("class") != "visual":
                geom.attrib.update(friction="2.0 0.02 0.001", solref="0.006 1", condim="4")

    sc = SCENE_CAMERA
    pc = player_camera or DEFAULT_PLAYER_CAMERA
    common = f'''
    <light pos="0.4 -1.2 2.6" dir="0 0.3 -1" diffuse="0.85 0.85 0.85" castshadow="true"/>
    <light pos="1.4 1.2 2.2" dir="-0.5 -0.4 -1" diffuse="0.35 0.38 0.42" castshadow="false"/>
    <camera name="scene" pos="{' '.join(map(str, sc['pos']))}" xyaxes="{look_at(sc['pos'], sc['target'])}" fovy="{sc['fovy']}"/>
    <camera name="player" pos="{' '.join(map(str, pc['pos']))}" xyaxes="{look_at(pc['pos'], pc['target'])}" fovy="{pc['fovy']}"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.30 0.31 0.33 1"/>
    <geom name="table" type="box" pos="0.56 0 {TABLE_TOP_Z - 0.04}" size="0.62 0.55 0.04" rgba="0.62 0.56 0.48 1"
          friction="1 0.02 0.001" solref="0.004 1" solimp="0.99 0.999 0.0001"/>
    '''
    for element in ET.fromstring("<worldbody>" + common + "</worldbody>"):
        world.append(element)
    backdrop = '''
    <geom type="box" pos="2.4 0 1.2" size="0.05 3 1.2" rgba="0.78 0.79 0.80 1"/>
    <geom type="box" pos="0.6 -2.6 1.2" size="2.2 0.05 1.2" rgba="0.74 0.75 0.77 1"/>
    <geom type="box" pos="0.6 2.6 1.2" size="2.2 0.05 1.2" rgba="0.74 0.75 0.77 1"/>
    '''
    for element in ET.fromstring("<worldbody>" + backdrop + "</worldbody>"):
        element.attrib.update(group="5", contype="0", conaffinity="0")
        world.append(element)
    if objects_xml.strip():
        for element in ET.fromstring("<worldbody>" + objects_xml + "</worldbody>"):
            world.append(element)
    if decor_xml.strip():
        decor = ET.fromstring("<worldbody>" + decor_xml + "</worldbody>")
        for element in decor.iter():
            if element.tag == "geom":
                element.attrib.update(group="4", contype="0", conaffinity="0")
        for element in decor:
            world.append(element)
    return ET.tostring(root, encoding="unicode")


def camera_intrinsics(fovy_deg, width, height):
    focal = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
