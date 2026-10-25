"""Go1 model and position actuators in native FR, FL, RR, RL joint order."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

GO1_XML = Path(__file__).parent / "xmls" / "go1.xml"
GO1_FOOT_NAMES = ("FR", "FL", "RR", "RL")
GO1_JOINT_NAMES = tuple(
  f"{leg}_{joint}_joint"
  for leg in GO1_FOOT_NAMES
  for joint in ("hip", "thigh", "calf")
)
GO1_ACTION_SCALE = 0.25
GO1_KP = 20.0
GO1_KD = 0.5


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO1_XML))
  assets: dict[str, bytes] = {}
  update_assets(assets, GO1_XML.parent / "assets", spec.meshdir)
  spec.assets = assets
  return spec


def get_go1_robot_cfg() -> EntityCfg:
  """Return an independent configuration, with all 12 actuators built by mjlab."""
  foot_regex = r"^[FR][LR]_foot_collision$"
  return EntityCfg(
    spec_fn=get_spec,
    sort_actuators=True,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.30),
      joint_pos={".*_hip_joint": 0.0, ".*_thigh_joint": 0.9, ".*_calf_joint": -1.8},
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(
        BuiltinPositionActuatorCfg(
          target_names_expr=(f".*_{joint}_joint",),
          stiffness=GO1_KP,
          damping=GO1_KD,
          effort_limit=limit,
          armature=0.01,
          frictionloss=0.2,
        )
        for joint, limit in (("hip", 23.7), ("thigh", 23.7), ("calf", 35.55))
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
    collisions=(CollisionCfg(
      geom_names_expr=(".*_collision",),
      contype=1,
      conaffinity=0,
      condim={foot_regex: 3, ".*_collision": 1},
      priority={foot_regex: 1},
      friction={foot_regex: (0.6,)},
      solimp={foot_regex: (0.9, 0.95, 0.023)},
    ),),
  )
