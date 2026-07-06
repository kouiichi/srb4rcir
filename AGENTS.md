# Agent Notes for SRB 0.0.5

This checkout is a locally repaired Space Robotics Bench 0.0.5 stack under:

```bash
/home/ubuntu/yyf/space_robotics_bench_0.0.5
```

Use the wrapper for runtime commands:

```bash
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh
```

The wrapper currently selects Isaac Sim 4.5, Isaac Lab 2.1, Blender 4.3.2, and a writable Matplotlib cache dir. Do not bypass it unless you are intentionally debugging environment setup.

## Known Runtime Stack

- Isaac Sim: `/home/ubuntu/isaac-sim-4.5`
- Isaac Sim Python: `/home/ubuntu/isaac-sim-4.5/python.sh`
- Isaac Lab: `/home/ubuntu/isaaclab-2.1`
- Blender for SimForge asset generation: `/home/ubuntu/blender-4.3.2-linux-x64`
- SRB wrapper: `/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh`

SRB 0.0.5 expects Blender 4.3.2 for the current `simforge==0.2.1` / `simforge_foundry==0.2.0` asset scripts. Blender 4.5.3 generated node-socket errors for cubesat solar-panel materials.

## Dependency Policy

Use Isaac Sim Python, not conda/base Python:

```bash
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh -- \
  /home/ubuntu/isaac-sim-4.5/python.sh -m pip ...
```

Avoid broad dependency installs such as:

```bash
pip install -e ".[all]"
pip install -e ".[sb3]"
```

Those can cause pip to rewrite torch/CUDA packages. Prefer narrow installs with `--no-deps`, then add missing imports one by one.

Packages already added for SB3/rl_zoo3 in this stack include:

- `stable-baselines3==2.4.1`
- `sb3-contrib==2.4.0`
- local `_external/rl_zoo3`
- `pandas`
- `huggingface-sb3`
- `wasabi`
- `tzdata`
- `optuna`
- `alembic`
- `colorlog`
- `sqlalchemy`

## CPU and GPU Separation

Do not treat `Environment device: cpu` as meaning "no GPU is used".

There are several separate device layers:

- Isaac Sim / Kit / Vulkan / RTX layer: selected by AppLauncher and normally uses GPU.
- AppLauncher device selection: printed as `Using device: cuda:0`; it sets `active_gpu` and `physics_gpu`.
- SRB environment tensor backend: `env.device`, derived from `env.cfg.sim.device`.
- PhysX tensor API frontend/backend: some getters/setters use CPU tensors even when the process has CUDA.
- SB3 policy/training device: currently passed as `device=env.unwrapped.device`.

In original SRB 0.0.5 code, the environment config sets:

```python
sim = SimulationCfg(
    device="cpu",
)
```

That means the stable default logic is:

```text
Isaac Sim / Kit / renderer: GPU
SRB env tensor backend: CPU
PhysX tensor setter data: CPU
SB3 PPO model device: CPU, because SRB passes env.unwrapped.device
```

This is why the unmodified default avoids CPU/CUDA tensor mismatch in the rendezvous task.

## Recommended Stable Training Command

Use `CUDA_VISIBLE_DEVICES` to choose the physical GPU for the Isaac Sim process, but leave `env.sim.device` at SRB's default CPU value:

```bash
CUDA_VISIBLE_DEVICES=3 \
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh srb agent train \
  --headless --algo sb3_ppo -e rendezvous \
  env.num_envs=2
```

Inside the process, `cuda:0` maps to the first visible GPU, so with `CUDA_VISIBLE_DEVICES=3`, process-local `cuda:0` means physical GPU 3.

Do not add this override for the current stable path:

```bash
env.sim.device=cuda:0
```

That override moves SRB env tensors and SB3 to CUDA, but the rendezvous `ThrustAction` uses PhysX tensor setters such as `set_masses()` and `set_inertias()`. In this stack, those setters can expect CPU tensors and fail with messages like:

```text
Expected all tensors to be on the same device, but found cuda:0 and cpu
Incompatible device of mass tensor in function setMasses: expected device -1, received device 0
```

`device -1` means CPU in the PhysX tensor backend; `device 0` means CUDA GPU 0.

## Current Local Code Changes

The local checkout currently contains experimental edits in:

- `srb_0.0.5_env.sh`: wrapper for Isaac Sim 4.5, Blender 4.3.2, CUDA library paths, and `MPLCONFIGDIR`.
- `srb/core/action/term/mobile/thrust.py`: experimental attempt to bridge env tensor device and PhysX tensor device.
- `hyperparams/sb3/ppo.yaml`: modified locally.
- `_external/rl_zoo3`: local external dependency.

Before committing or treating this as a clean upstream patch, inspect:

```bash
git -C /home/ubuntu/yyf/space_robotics_bench_0.0.5 status --short
git -C /home/ubuntu/yyf/space_robotics_bench_0.0.5 diff
```

The `thrust.py` edits are not yet proven as a final GPU backend solution. The robust baseline remains to run without `env.sim.device=cuda:0`.

## Future Direction for CPU/GPU Fixes

The clean long-term goal is to decouple policy training device from simulation tensor backend:

```text
SRB env / PhysX tensor backend: CPU for compatibility
SB3 policy / neural network: CUDA for backprop
Isaac Sim / Kit / renderer: selected GPU via CUDA_VISIBLE_DEVICES/AppLauncher
```

Do not accomplish this by forcing `env.sim.device=cuda:0`. Instead:

1. Add an SB3-specific training device option, for example `agent.device=cuda:0` or `agent.policy_device=cuda:0`.
2. In `srb/integrations/sb3/main.py`, stop hard-wiring:

   ```python
   device=env.unwrapped.device
   ```

   and use the new SB3 device option while keeping the env device unchanged.

3. Keep `srb/integrations/sb3/wrapper.py` converting actions to `self.sim_device` before calling `env.step()`, since the environment still expects CPU tensors in the stable path.
4. Ensure observations/rewards/dones exported from the VecEnv remain NumPy/CPU-compatible for Stable-Baselines3 vector env semantics.
5. Only revisit full `env.sim.device=cuda` after auditing all SRB task/action code that calls PhysX tensor setters, especially:

   - `root_physx_view.set_masses`
   - `root_physx_view.set_inertias`
   - `root_physx_view.apply_forces_and_torques_at_position`
   - any setter taking `indices`

When writing to PhysX setter APIs, follow the backend's expected tensor device, not blindly `env.device`.

## Fast Validation Checklist

Check Blender selected by wrapper:

```bash
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh -- which blender
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh -- blender --version
```

Check SB3 import path:

```bash
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh -- \
  /home/ubuntu/isaac-sim-4.5/python.sh -c "from srb.integrations.sb3 import main; print('srb sb3 main ok')"
```

Check CUDA visible to Isaac Sim Python:

```bash
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh torch-check
```

Start with low env count:

```bash
CUDA_VISIBLE_DEVICES=3 \
/home/ubuntu/yyf/space_robotics_bench_0.0.5/srb_0.0.5_env.sh srb agent train \
  --headless --algo sb3_ppo -e rendezvous \
  env.num_envs=2
```
## 项目目标
- 使用状态RL训练轨迹专家获取轨迹
- 通过act/dp方式进行模仿学习的训练
