# 当前仓库运行指导

本文档说明如何判断 Isaac Sim/Isaac Lab 能否无图像启动，以及如何启动一个实际环境。所有命令都从仓库根目录执行：

```bash
cd /home/ubuntu/yyf/space_robotics_bench_0.0.5
source ./srb_0.0.5_env.sh
```

## 先做不启动仿真的检查

```bash
./srb_0.0.5_env.sh torch-check
./srb_0.0.5_env.sh srb-help
```

再检查 Isaac Sim/Isaac Lab 是否能够创建 headless Kit：

```bash
./srb_0.0.5_env.sh sim-check
```

这一步只检查应用启动，不创建具体 Gym 环境。

## 查看目前可用环境

```bash
./srb_0.0.5_env.sh srb ls env
```

显示隐藏的模板和 visual 环境：

```bash
./srb_0.0.5_env.sh -- "$ISAAC_SIM_PYTHON" -m srb ls env -a
```

`srb ls env` 本身会启动一次 headless Kit，因此也能发现 Isaac Sim、GPU 或扩展加载问题。

## 无图像模式启动一个环境

使用随机动作运行 `rendezvous_16rcs`，并在 5 秒后自动结束，适合第一次验证：

```bash
env -u CUDA_VISIBLE_DEVICES \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./srb_0.0.5_env.sh srb agent rand \
  --headless -e rendezvous_16rcs \
  --perf --perf_duration 5 \
  --logdir logs/run_smoke
```

这里的 `--headless` 表示不显示窗口，但仍然会初始化 Isaac Sim 的 Kit/渲染后端；它不是纯 CPU 模式。当前仓库验证的分工是 Kit 使用 GPU、SRB 环境张量和 PhysX 使用 CPU。若主机没有可用 RTX/Vulkan/CUDA 设备，应用可能在启动阶段报 `No device could be created` 或 `GPU Foundation is not initialized`。

如果只想持续运行环境而不做性能计时，可以去掉 `--perf --perf_duration 5`，然后使用 `Ctrl-C` 结束：

```bash
./srb_0.0.5_env.sh srb agent rand --headless \
  -e rendezvous_16rcs env.num_envs=1
```

其它环境只需替换 `-e` 后的 ID，例如：

```bash
./srb_0.0.5_env.sh srb agent rand --headless \
  -e locomotion_velocity_tracking env.num_envs=1
```

## 修改版 NMPC 的无视频物理验证

NMPC 回放脚本使用仓库中的已保存计划。它明确关闭视频和窗口，并使用 `--physics-only`：

```bash
env -u CUDA_VISIBLE_DEVICES \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./srb_0.0.5_env.sh -- \
  "$ISAAC_SIM_PYTHON" -u scripts/validate_nmpc_replay.py replay \
  --plan logs/nmpc_validation_20260907/approach_hcw_plan \
  --output logs/run_nmpc_smoke \
  --physics-only --device cuda:1 --closed-loop --align-goal \
  --matched-executor --horizon-steps 40 --execute-steps 3 --steps 20
```

这是当前修改版最有代表性的启动检查；`--physics-only` 不代表 Kit 不需要 GPU，而是禁用相机/视频并使用 CPU 兼容的 PhysX setter 路径。完整 1200 步运行时必须换一个新的 `--output` 目录。

## 换到同一台电脑的另一个文件夹

不需要重复安装 Isaac Sim、Isaac Lab、Blender 或已经存在的 Python wheel。这些组件安装在共享位置，通常只安装一次。

但本仓库的 `srb_0.0.5_env.sh` 含有绝对路径，并且 editable 安装会记录源码目录，所以第二个 checkout 不能无条件复用第一个 checkout 的 wrapper。推荐按下面方式处理：

1. 在新目录复制或重新 clone 仓库。
2. 修改新目录的 `srb_0.0.5_env.sh` 中的 `SRB_ROOT`，并确认 `ISAAC_SIM_PATH`、`BLENDER_PATH` 指向共享安装。
3. 在新目录重新执行一次本地 editable 安装，让 `srb` 指向新源码；这不会重新安装 Isaac Sim：

```bash
cd /path/to/second/space_robotics_bench_0.0.5
source ./srb_0.0.5_env.sh
"$ISAAC_SIM_PYTHON" -m pip install --no-deps --editable .
```

4. 之后直接运行本文件中的命令即可。

也可以不安装 CLI，直接从新目录调用模块：

```bash
cd /path/to/second/space_robotics_bench_0.0.5
source ./srb_0.0.5_env.sh
"$ISAAC_SIM_PYTHON" -m srb agent rand --headless \
  -e rendezvous_16rcs --perf --perf_duration 5
```

不过如果 wrapper 仍然把 `SRB_ROOT` 指向旧目录，acados、Matplotlib cache 和本地 optional Python 支持目录也会从旧目录读取；部署新 checkout 时应当改正这些路径。

## 能否直接尝试运行

可以。最安全的顺序是：

```bash
./srb_0.0.5_env.sh torch-check
./srb_0.0.5_env.sh sim-check
./srb_0.0.5_env.sh srb ls env
env -u CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID \
  ./srb_0.0.5_env.sh srb agent rand --headless \
  -e rendezvous_16rcs --perf --perf_duration 5
```

如果失败，先看终端中较早出现的决定性错误，并检查 Kit 日志：

```bash
rg -n -i 'CUDA error|bad state|No device could|no suitable|GPU Foundation|vkCreateDevice|GLFW|GLInterop' \
  /home/ubuntu/isaac-sim-4.5/kit/logs/Kit /tmp/nmpc_*.log
```

`GLFW initialization failed` 在无显示服务器上可能只是预期警告；`No device could be created`、`GPU Foundation is not initialized` 和 `CUDA libs are present, but no suitable CUDA GPU was found` 则表示本次 Kit/物理启动没有成功。

在当前执行环境中，我实际尝试了上面的 `rendezvous_16rcs` headless 性能命令。Kit 能进入启动流程，但主机当前没有可用 CUDA/Vulkan GPU，随后报告 `No device could be created`、`GPU Foundation is not initialized` 和 `no CUDA-capable device is detected`，因此这次没有完成环境运行。这是主机设备状态问题，不是命令格式问题。
