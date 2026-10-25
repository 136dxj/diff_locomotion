# Unitree Go1 model

Source: [Google DeepMind MuJoCo Menagerie, unitree_go1](https://github.com/google-deepmind/mujoco_menagerie/tree/71f066ad0be9cd271f7ed58c030243ef157af9f4/unitree_go1),
commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. Upstream derives this model
from the [Unitree Go1 URDF](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/go1_description).
The model and all five STL meshes are covered by the included upstream
BSD-3-Clause `LICENSE`; the meshes are copied without modification.
`SOURCE_SHA256SUMS` records upstream XML and mesh checksums.

## mjlab adaptation

- Preserve all inertials, joint limits, limb transforms, meshes and collision shapes.
- Rename `trunk` to `base_link`, name the free joint and every collision geom, and
  expose foot geoms as `{FR,FL,RR,RL}_foot_collision`. Keep foot sites `FR,FL,RR,RL`.
- Add `imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc` on site `imu` and `root_angmom`
  on `base_link`. Remove scene light/camera/options and the source keyframe.
- Remove XML position actuators/defaults and passive joint damping; mjlab builds
  exactly 12 position actuators. Preserve upstream joint frictionloss `0.2`.
  Visual geoms have zero density. No extra joints or synthetic links are added.
- EntityCfg uses environment collisions without robot self-collisions, 3D foot
  contacts, nominal sliding friction `0.6`, and foot solimp `(0.9,0.95,0.023)`,
  following the repository's Go2 collision convention.

## Control and integration contract

Joint/observation/action order: `FR, FL, RR, RL`, each `hip, thigh, calf`, with
joint names `<leg>_<joint>_joint`. This is the native Go1 motor order (indices
0 through 11); do not reuse the Go2 permutation. mjlab creates and orders actuators
from these model joints. Deployment should inspect the compiled model for the
final actuator-to-joint mapping.

Training gains are `kp=20`, `kd=0.5` for all joints, with model torque limits
`23.7/23.7/35.55 Nm` per leg and armature `0.01`. These gains are task tuning,
not a hardware specification. Nominal joint pose is `[0,0.9,-1.8]` per leg,
base height `0.30 m`, and action scale `0.25 rad` about that pose.

Feet are spheres of radius `0.023 m`, with site/geom centres at `(0,0,-0.213)`
in each calf frame. Shared dynamics code should obtain this geometry, the body
tree, masses and inertias from the loaded model rather than embed Go1 dimensions.

The original mjlab PPO task registration is not included in this standalone
repository. Use `src.diff_loco.train` for training and `src.diff_loco.evaluate`
for independent evaluation. The differentiable locomotion task uses 100 Hz
control and five physics substeps (500 Hz); its randomization ranges are defined
in `src/diff_loco/task.py`.

The bundled iteration-67250 Go1 example overrides the default action scale to
`0.6 rad` and uses a `0.10 m` reference swing height. Exported metadata records
the actual gains, action scale, joint order, and physics settings for each policy.
