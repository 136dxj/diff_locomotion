# Third-party notices

This repository packages the differentiable locomotion implementation separately
from its original training checkout.

- Unitree G1 robot model, meshes, and configuration:
  [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab),
  upstream commit `1425b15` (Apache License 2.0, retained in `LICENSE`).
- Unitree Go1 robot model and meshes:
  [google-deepmind/mujoco_menagerie, unitree_go1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1).
  The Unitree BSD 3-Clause license is retained in
  `src/assets/robots/unitree_go1/LICENSE`. The XML was adapted for this training
  task; the Go1 mjlab adapter is in `go1_constants.py`.
- `src/assets/robots/__init__.py` was reduced to avoid importing unrelated
  robot packages when installing this standalone distribution.
- The included Go1 exported scene contains the same Go1 model assets, covered
  by the accompanying BSD license.

External Python dependencies (MuJoCo, mjlab, MuJoCo Warp, Warp, PyTorch,
ONNX Runtime, and others) retain their respective licenses and are installed
through `uv`; their binaries are not vendored here.
