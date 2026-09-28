# 修改版 SRB 0.0.5 部署说明

本文档针对本仓库的修改版（NMPC、matched executor、湿质量重置修复和数据记录脚本），不是官方源码的通用安装说明。官方 0.0.5 原生安装基线是 Ubuntu 22.04、NVIDIA Isaac Sim 4.5、Isaac Lab 2.1、Blender 4.3，并在 Isaac Sim 自带 Python 中以 editable 模式安装 SRB；官方文档还要求 NVIDIA RTX、至少 16 GB RAM、4 GB VRAM 和 32 GB 磁盘空间。参见 [官方 0.0.5 原生安装文档](https://github.com/AndrejOrsula/space_robotics_bench/blob/0.0.5/docs/src/getting_started/install_native.md) 和 [系统要求](https://github.com/AndrejOrsula/space_robotics_bench/blob/0.0.5/docs/src/getting_started/requirements.md)。

## 与官方 0.0.5 的差异

| 项目 | 官方 0.0.5 | 本修改版的部署要求 |
| --- | --- | --- |
| Python | Isaac Sim 自带 Python 3.10 | 仍使用 `/home/ubuntu/isaac-sim-4.5/python.sh` 或等价路径，不能混用系统/conda Python |
| Isaac Sim / Lab | Isaac Sim 4.5、Isaac Lab 2.1 | 保持 4.5 / 2.1 组合；当前验证的 `isaaclab` 是 0.36.21 |
| Blender | 4.3 | 使用 Blender 4.3.2；SimForge 资产依赖安装在同一 Blender Python 环境 |
| NMPC | 官方文档未包含本仓库的 NMPC 专家 | `casadi==3.8.0`、`acados_template==0.5.1`、`osqp==0.6.5`，并使用仓库内 `third_party/acados` 的共享库 |
| 数据记录 | 官方基础安装 | 可选 `pyarrow==18.1.0`、`Cython==3.0.12`，隔离安装到 `third_party/python` |
| 设备分工 | 未规定本修改版实验分工 | Kit/渲染使用 GPU；SRB 张量、PhysX、NMPC/OSQP 使用 CPU；物理验证使用 `--physics-only` |
| 启动方式 | `srb` 或 `"$ISAAC_SIM_PYTHON" -m srb` | 优先通过 `srb_0.0.5_env.sh`，它还设置 acados、Blender、Matplotlib 和 CUDA 运行库路径 |

## 1. 主机前提

建议使用 Ubuntu 22.04 x86_64、NVIDIA RTX GPU 和驱动 535–559。官方最低配置是 16 GB RAM、4 GB VRAM、32 GB 磁盘；完整渲染和并行任务建议至少 12 GB VRAM。当前修改版的无窗口物理验证仍需要 Kit 初始化 GPU，但 SRB/PhysX 张量保持 CPU。

准备基础工具：

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip xz-utils build-essential cmake pkg-config
```

## 2. 获取源码和外部运行时

```bash
git clone --recurse-submodules <修改版仓库地址> space_robotics_bench_0.0.5
cd space_robotics_bench_0.0.5
```

安装或准备以下固定组件：

* Isaac Sim 4.5，入口为 `/home/ubuntu/isaac-sim-4.5/python.sh`；
* Isaac Lab 2.1，安装到该 Isaac Sim Python 环境；
* Blender 4.3.2，并在 Blender 的 Python 3.11 环境中安装 `simforge[assets]`；
* 本仓库的 `third_party/acados`，其中应存在 `lib/libacados.so`、`lib/libhpipm.so` 和 `bin/t_renderer`。

仓库提供官方版本对应的安装脚本：

```bash
./scripts/install_isaacsim.bash /home/ubuntu/isaac-sim-4.5
export ISAAC_SIM_PYTHON=/home/ubuntu/isaac-sim-4.5/python.sh
./scripts/install_isaaclab.bash /home/ubuntu/isaaclab
```

如果 Isaac Sim/Lab 已由系统管理员安装，不要重复执行覆盖安装脚本，只需把路径改成实际位置。

## 3. 配置修改版环境

`srb_0.0.5_env.sh` 当前包含本机路径默认值。部署到其他机器时先修改 `SRB_ROOT`、`ISAAC_SIM_PATH` 和 `BLENDER_PATH`，或者在 source 前导出同名变量并按需调整脚本。然后执行：

```bash
source ./srb_0.0.5_env.sh
"$ISAAC_SIM_PYTHON" -m pip install --upgrade pip
"$ISAAC_SIM_PYTHON" -m pip install --no-deps -r requirements-srb-modified.txt
"$ISAAC_SIM_PYTHON" -m pip install --no-deps --editable .
```

`--no-deps` 是有意的：Isaac Sim 已经携带 PyTorch、NumPy、SciPy 等二进制包，重新解析依赖可能破坏 Kit/PhysX ABI。若某个基础包确实缺失，再只安装该包，并先记录 `pip show` 结果。

可选安装 Parquet 支持：

```bash
"$ISAAC_SIM_PYTHON" -m pip install --no-deps --only-binary=:all: \
  --target "$SRB_ROOT/third_party/python" \
  -r requirements-nmpc-isolated.txt
```

## 4. 验证安装

先做不启动 Kit 的 Python 导入检查，再做无窗口仿真检查：

```bash
./srb_0.0.5_env.sh torch-check
./srb_0.0.5_env.sh sim-check
./srb_0.0.5_env.sh srb-help
"$ISAAC_SIM_PYTHON" -m pip check
```

检查 NMPC 关键依赖和 acados 动态库：

```bash
"$ISAAC_SIM_PYTHON" - <<'PY'
import importlib.metadata as md
import casadi, numpy, osqp, scipy
print("acados_template", md.version("acados_template"))
print("casadi", casadi.__version__)
print("numpy", numpy.__version__)
print("osqp", osqp.__version__)
print("scipy", scipy.__version__)
PY
test -x third_party/acados/bin/t_renderer
test -r third_party/acados/lib/libacados.so
```

## 5. 修改版 NMPC 冒烟运行

在当前 GPU Kit / CPU PhysX 主机上，使用未屏蔽的 CUDA 枚举，并通过 wrapper 启动：

```bash
env -u CUDA_VISIBLE_DEVICES \
  CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./srb_0.0.5_env.sh -- \
  "$ISAAC_SIM_PYTHON" -u scripts/validate_nmpc_replay.py replay \
  --plan logs/nmpc_validation_20260907/approach_hcw_plan \
  --output logs/deployment_smoke_$(date +%Y%m%d_%H%M%S) \
  --physics-only --device cuda:1 --closed-loop --align-goal \
  --matched-executor --horizon-steps 40 --execute-steps 3 --steps 20
```

部署验收至少应确认：`torch-check` 能看到 CUDA、`sim-check` 能创建 Isaac Sim/Isaac Lab、NMPC 冒烟运行执行 20 步且输出目录成功生成。完整 1200 步实验应使用新的输出目录，并记录 Kit GPU、CUDA 可见性、CPU PhysX、任务/目标、参数、请求和实际步数、求解器 P50/P99 及失败原因。

## 6. 常见错误

* `No device could be created` 或 `GPU Foundation is not initialized`：检查 NVIDIA 驱动、Vulkan 和是否错误设置了 `CUDA_VISIBLE_DEVICES`；本机验证命令使用 `env -u CUDA_VISIBLE_DEVICES`。
* `Expected all tensors to be on the same device`：不要把 `env.sim.device` 强制改成 CUDA；当前 PhysX setter 路径要求 CPU 张量。
* 找不到 `libacados.so` 或 `t_renderer`：确认 `ACADOS_SOURCE_DIR` 指向仓库内 `third_party/acados`，并已 source wrapper。
* `pyarrow` 导入失败：按第 3 节将 `requirements-nmpc-isolated.txt` 安装到 `third_party/python`，不要升级 Isaac Sim 的整套依赖。
* `srb` 命令不存在：使用 `"$ISAAC_SIM_PYTHON" -m srb`，或在 `PATH` 中准备好 `$HOME/.local/bin` 后运行 `./scripts/setup_cli.bash`。

## 7. 复现信息

部署完成后保存以下输出，便于比较机器之间的差异：

```bash
git rev-parse HEAD
uname -a
nvidia-smi
"$ISAAC_SIM_PYTHON" -m pip list --format=freeze > logs/pip-freeze-$(date +%Y%m%d).txt
```

不要把完整 `pip freeze` 当作可移植 requirements；Isaac Sim 的预装扩展和 CUDA wheel 必须由对应 Isaac Sim 发行包提供。可移植的直接依赖清单见本仓库的 [`requirements-srb-modified.txt`](requirements-srb-modified.txt)。
