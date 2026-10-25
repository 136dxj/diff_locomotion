# diff_locomotion

基于 MuJoCo / mjlab 与 PyTorch 的可微运动控制实验。发布包名为 `diff-locomotion`，Python 入口为 `src.diff_loco`。包含通用动力学接口、Go1 / G1 机器人资源、8 个测试文件及 Go1 示例策略。

## 方法与边界

- **通用 RobotModel**：从编译后的 `mujoco.MjModel` 或 XML 路径读取运动学树、惯量和接触几何，提供 `kinematics`、`mass_matrix`、`gravity`、`step`。支持一个自由根关节及其下的 hinge/slide 关节；`qpos` 为 `[B, 7+N]`，`qvel` 为 `[B, 6+N]`。机器人专属的关节控制、足端和步态配置留在 task / assets 中。
- **仿真前向对齐 + 短窗梯度 / critic**：前向状态采用 mjlab 仿真值，反向梯度经过解析可微替代动力学；短窗目标结合 critic 的后续价值，critic 使用 TD(lambda) 目标。没有学习世界模型。
- **解析平滑接触近似**：使用广义惯量、隐式 PD / 阻尼、平滑力矩饱和及固定四轮接触迭代，作用于水平地面。这是近似梯度模型，省略科氏/离心项、自碰撞等效应，并非完整 NCP 接触求解器或论文逐项严格复现。
- **actor 仅用实机可构造观测**：IMU 角速度与姿态推得的重力方向、速度命令、控制器步态时钟、关节位置/速度、上一动作；不读取真实机体平移速度或随机物理参数。特权状态仅用于训练 critic。动作转为 `clip(q0 + scale * action, joint_limits)`。
- **IK RSI 与域随机化**：有界 IK 生成步态重置状态及动作历史，无法满足约束时显式报错。训练默认随机化质量、惯量、质心、摩擦、控制增益、力矩上限、编码器偏置、延迟及扰动；`--nominal` 关闭训练域随机化。IK 可达性不等于动力学稳定性。

## 环境

Python 3.11 / 3.12，使用 uv，在仓库根目录执行：

```bash
uv sync --frozen --group dev
```

训练和默认导出使用 CUDA；Linux x86_64 的 PyTorch 索引为 cu128。主要版本为 `mjlab==1.2.0`、`mujoco==3.5.0`、`mujoco-warp==3.5.0`、`torch==2.8.0`，**固定 `warp-lang==1.12.0`**，其余解析版本见 `uv.lock`。NumPy 用于模型和推理；导出需 ONNX / ONNX Runtime；录像需 ImageIO、imageio-ffmpeg、Pillow 及可用的 FFmpeg/libx264。纯策略推理只需 NumPy + ONNX Runtime，CPU 仿真评估另需 MuJoCo。

## Go1 示例：67250 轮

[checkpoint](examples/go1_67250/checkpoint.pt) · [bundle](examples/go1_67250/bundle/) · [评估 JSON](examples/go1_67250/evaluation.json) · [演示视频](examples/go1_67250/go1_preview.mp4)

训练历史为：前 **30000 轮使用 3 cm clearance / 0.25 rad action scale**，随后改为 **10 cm / 0.6 rad**，累计至 67250 轮。该示例并非从零采用 10 cm 配置训练所得。发布前仅移除 checkpoint 训练元数据中的主机绝对路径，权重及 optimizer 状态不改。

附带记录是一回合 **10 秒名义 CPU MuJoCo** 评估（seed 2027）：

| 项目 | 记录 |
| --- | --- |
| 前向速度命令 | 0.2 m/s |
| 平均机体系前向速度 | 0.207057 m/s |
| 平面速度 tracking RMSE | 0.017894 m/s |
| 跌倒 | 无 |

这份记录不构成随机化测试或实机验收。**10 cm 是参考轨迹的摆动抬升参数，不等同于仿真或实机测得的脚尖峰值。** G1 29 自由度已接入，目前仅有 500 轮早期训练状态，不能称为收敛。

## 评估现有 bundle

以下评估在 CPU 上运行，无需训练、重新导出或 CUDA/GPU：

```bash
uv run --frozen python -m src.diff_loco.evaluate examples/go1_67250/bundle \
  --output outputs/go1_67250/evaluation.json \
  --episodes 1 --seconds 10 --vx 0.2 --vy 0 --yaw 0 --seed 2027
```

生成新视频使用 EGL 渲染；物理与策略推理仍在 CPU 上，但环境须提供可用的 EGL/OpenGL 渲染后端：

```bash
MUJOCO_GL=egl uv run --frozen python -m src.diff_loco.evaluate examples/go1_67250/bundle \
  --output outputs/go1_67250/video_evaluation.json \
  --video outputs/go1_67250/preview.mp4 \
  --episodes 1 --seconds 10 --vx 0.2 --vy 0 --yaw 0 --seed 2027
```

另加 `--randomized` 可做独立的随机化评估，`--zero-action` 可做默认姿态基线；这些结果应单独保存，不能与附带名义记录混用。

## 训练与续训

从零训练 Go1 的 10 cm / 0.6 rad 配置（此命令不是上述示例的历史训练流程）：

```bash
uv run --frozen python -m src.diff_loco.train \
  --robot go1 --device cuda:0 --num-envs 32 --horizon 24 --seed 42 \
  --clearance .1 --action-scale .6 --rsi .8 --max-speed .25 \
  --updates 300000 --output runs/go1_10cm_from_scratch
```

从附带的 67250 轮 checkpoint 续训到累计 300000 轮，需新增 **232750** 轮，使用单独输出目录：

```bash
uv run --frozen python -m src.diff_loco.train \
  --robot go1 --device cuda:0 --num-envs 32 --horizon 24 --seed 42 \
  --clearance .1 --action-scale .6 --rsi .8 --max-speed .25 \
  --resume examples/go1_67250/checkpoint.pt \
  --updates 232750 --output runs/go1_67250_to_300000
```

`--updates` 是新增轮数，不是累计目标；换用其他 checkpoint 时应按其 `iteration` 重新计算。续训恢复 actor、critic 和优化器状态；仅 clearance/action scale 在省略时自动继承，其他训练参数应自行保持一致。未恢复随机数及仿真现场，因此不是逐步完全相同的断点重放。输出包含 `last.pt`、`config.json`、`metrics.jsonl`。

G1 的入口 `--robot g1` 对应 29 自由度；保留其默认 clearance 和逐关节 action scale，不套用 Go1 的覆盖值：

```bash
uv run --frozen python -m src.diff_loco.train \
  --robot g1 --updates 500 --output runs/g1_29dof_early
```

## 导出与测试

默认导出会重建 CUDA task（`cuda:0`），CLI 没有 `--device` 参数：

```bash
uv run --frozen python -m src.diff_loco.export \
  runs/go1_67250_to_300000/last.pt outputs/go1_300000_bundle

uv run --frozen python -B -m pytest -p no:cacheprovider tests/test_diff*.py -q
```

导出生成 ONNX、TorchScript、观测/动作元数据、完整场景资源、独立 `runtime.py` 和数值一致性检查 `parity.json`。将评估命令的 bundle 路径替换为新目录即可评估；数值一致性不代表行走或实机验收。GPU 集成测试在无 CUDA 时会跳过。

发布验证（2026-10-25，Linux / RTX 4090）：精简副本 **138 项测试通过、2 项跳过**，有一条测试中 tensor 转标量的既有警告；独立目录的 `uv sync --frozen --group dev` 安装成功；新环境回放附带 bundle，复现上述 10 秒评估结果并成功生成视频。

保留 `src/__init__.py`、`src.diff_loco` 包结构及 Go1/G1 的配置、XML 和全部网格；G1 配置依赖 `SRC_PATH`。评估 bundle 需要 `metadata.json`、`exported/policy.onnx`、`scene/scene.xml` 及其资源，不能只复制 ONNX 文件。默认训练用解析 IK 参考，不依赖外部动作数据集。

## 来源与许可

保留 upstream Unitree 项目的 [Apache-2.0 LICENSE](LICENSE)，Go1 资源保留其 [BSD-3-Clause LICENSE](src/assets/robots/unitree_go1/LICENSE)。第三方来源、修改和导出场景的许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

思路参考：[arXiv:2403.14864](https://arxiv.org/abs/2403.14864)（仿真状态对齐）、[arXiv:2404.02887](https://arxiv.org/abs/2404.02887)（平滑接触梯度）。本项目不宣称复现论文全部结果，也不据此给出部署可用结论。
