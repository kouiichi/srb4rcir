# Agent Notes for SRB 0.0.5

This is the locally repaired SRB 0.0.5 checkout:

/home/ubuntu/yyf/space_robotics_bench_0.0.5

Current development focuses on: (1) validating and improving the NMPC trajectory expert in SRB; (2) making SRB dynamics, actuators, task references, and recorded actions consistent enough for ACT / Diffusion Policy imitation learning; and (3) keeping a reproducible GPU Kit / CPU PhysX environment split.

Updated: 2026-09-11. This file describes the current state, commands, and development boundaries. It is not an upstream SRB configuration.

Preserve existing local edits. Before changing code or committing, inspect git status --short and git diff; do not assume older lists of modified files still describe the checkout.

## Runtime and dependency rules

Always use the wrapper and Isaac Sim Python:

    cd /home/ubuntu/yyf/space_robotics_bench_0.0.5
    source ./srb_0.0.5_env.sh
    /home/ubuntu/isaac-sim-4.5/python.sh -m pip ...

Do not use base or conda Python for SRB. Avoid broad installs such as pip install -e ".[all]" or pip install -e ".[sb3]"; use narrow Isaac Sim Python installs with --no-deps only after identifying a missing import. The wrapper selects Isaac Sim 4.5, Isaac Lab 2.1, Blender 4.3.2, acados support paths, and a writable Matplotlib cache. Blender 4.3.2 is required by the current SimForge asset stack.

## Device layers

These are separate:

    Kit / Vulkan / RTX renderer: AppLauncher --device
    SRB environment tensors: currently CPU
    PhysX tensor setters: API/backend-specific; stable path is CPU-compatible
    NMPC/acados and OSQP: CPU
    SB3 policy: currently follows env device unless integration is changed

Environment device: cpu does not mean that Kit is not using a GPU. The validated configuration uses a GPU for Kit and CPU for SRB/PhysX-compatible tensors.

### Validated GPU Kit command

On this five-GPU host, masked runs reported CUDA/Omniverse enumeration warnings and failed Kit GPU initialization; the unmasked comparison succeeded. This supports avoiding the masked configuration here, but individual environment variables were not independently ablated. The validated NMPC replay configuration removes that variable and explicitly selects Kit GPU 1:

    env -u CUDA_VISIBLE_DEVICES \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    ./srb_0.0.5_env.sh -- \
    /home/ubuntu/isaac-sim-4.5/python.sh -u \
    scripts/validate_nmpc_replay.py replay \
    --plan logs/nmpc_validation_20260907/approach_hcw_plan \
    --output logs/nmpc_validation_gpu1_nomask_next \
    --physics-only \
    --device cuda:1 \
    --closed-loop \
    --align-goal

Do not combine this with CUDA_VISIBLE_DEVICES=1. With no masking, cuda:1 is the Kit/Omniverse ordinal for the physical GPU whose Bus-ID is 81 on this host. This configuration passed both a 20-step smoke test and a 1200-step / 60-second replay.

Do not unconditionally add unset CUDA_VISIBLE_DEVICES to srb_0.0.5_env.sh. The wrapper is also used by RL commands that may intentionally select a GPU with CUDA_VISIBLE_DEVICES. Keep this prefix local to NMPC/Kit diagnostics unless an opt-in wrapper mode is implemented.

### Conservative CPU replay

This is the historical CPU physics fallback; --device cpu does not guarantee that Kit skips all GPU initialization.

    env -u CUDA_VISIBLE_DEVICES \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    ./srb_0.0.5_env.sh -- \
    /home/ubuntu/isaac-sim-4.5/python.sh -u \
    scripts/validate_nmpc_replay.py replay \
    --plan logs/nmpc_validation_20260907/approach_hcw_plan \
    --output logs/nmpc_validation_cpu_next \
    --physics-only \
    --device cpu \
    --closed-loop \
    --align-goal

Run all commands from the repository root shown above. Every run needs a new output directory; the validation script uses exist_ok=False. For a short smoke test, append --steps 20. Omit --steps for the full saved-plan length (currently 1200). The thread limits are the NMPC validation settings, not universal GPU requirements.

## Headless/no-window policy

For physics and controller validation, use --physics-only. The replay script starts Kit headless/no-window, creates the stage, and disables rendering after setup. This path has completed a full GPU-Kit/CPU-PhysX replay.

GLFW initialization failed and GLInteropContext: carb::windowing is not available are expected on the current no-display host and do not by themselves mean that GPU creation failed. Check earlier decisive messages instead:

    Skipping NVIDIA GPU due CUDA being in bad state
    No device could be created
    GPU Foundation is not initialized
    CUDA libs are present, but no suitable CUDA GPU was found

Video and camera rendering are a separate path; do not infer that video works from physics-only success.

## Current NMPC expert

The expert is an optimization controller, not a trained neural network. Its current contract is:

- 13-state position, velocity, scalar-first quaternion, and body angular velocity;
- 16 one-sided thruster duties in [0,1];
- 0.05 s / 20 Hz control period;
- 20 intervals / 1 s prediction horizon;
- five 0.01 s RK4 substeps per interval;
- optimizer move blocking (3,3,3,3,2,2,2,2);
- nominal expert HCW mean motion 0.001027 rad/s, mass 12 kg, diagonal inertia [0.20,0.24,0.30], COM zero;
- aligned target [-1.5,0,0] m with unit attitude;
- default KOZ center [0,0,0] m and radius 0.5 m.

Optimizer move blocking is not the same as ACT/DP student action-chunk execution.

The 16 duties map to a six-dimensional wrench:

    [Fx,Fy,Fz,tx,ty,tz] = B(6x16) @ duty16

ThrustAction also updates mass and inertia as fuel is consumed and calls PhysX setters for masses, inertias, forces, torques, and indices. Do not treat a nominal 6D wrench as a calibrated real-executor label.

## Latest verified NMPC result

The GPU-Kit comparison used:

    CUDA_VISIBLE_DEVICES: unset
    CUDA_DEVICE_ORDER: PCI_BUS_ID
    Kit device: cuda:1, physical RTX 4090 Bus-ID 81
    physics-only: true
    SRB environment: CPU
    closed-loop: true
    aligned target: true, [-1.5,0,0] m, unit attitude
    initial position: [-3,+0.2,0.15] m; unit attitude, zero linear/angular velocity
    requested/executed: 1200/1200 steps, 60 s

Strict hover requires position error <0.1 m, relative speed <0.05 m/s and attitude error <5 degrees continuously for at least 1 s. Complete success also requires all requested steps and no failure. SRB native success is a different, looser criterion.

Results:

| Metric | Value |
|---|---:|
| final position error | 0.00145176 m |
| final relative speed | 7.15e-7 m/s |
| final attitude error | 0.0003526 deg |
| longest strict hover | 27.35 s |
| minimum KOZ clearance | 0.3159 m |
| maximum contact force | 0 N |
| solver P50 | 17.34 ms |
| solver P99 | 65.39 ms |
| deadline miss fraction | 1.668% |
| fuel remaining | 1.506 kg from 5 kg |

This is a successful closed-loop result for a narrow aligned-target condition. It does not prove general-task success, GPU PhysX execution, or real-time compliance.

Artifacts: logs/nmpc_validation_gpu1_nomask_full_20260911/replay.json, replay.npz, comparison.png, and /tmp/nmpc_gpu1_nomask_full_20260911.log. The broader report is logs/nmpc_validation_20260908/report.md.

Boundary results:

- Saved HCW actions replayed open-loop in stationary SRB: final position error about 13.99 m; failed.
- Two same-side feedback closed loops: strict complete success, with overshoot and correction before hover.
- Replacing initial measured mass/inertia/COM: first acados status 4 and zero executed commands; diagnostic only, not a completed calibration.
- Native [0,0,0.5] target with large attitude change: stopped after 44 steps because a predicted angular-rate margin was slightly negative.
- NMPC P99 was about 144 ms in a separate benchmark/load setup; safety OSQP P99 was about 1.21 ms.
- Expert HCW uses n=0.001027 while SRB free-floating uses n=0; the free-floating expert path is not yet consistent.

## PhysX device rule

Forcing env.sim.device=cuda:0 can make environment tensors and PhysX setter inputs disagree. Known errors include:

    Expected all tensors to be on the same device, but found cuda:0 and cpu
    Incompatible device of mass tensor in function setMasses:
    expected device -1, received device 0

Here PhysX device -1 means CPU. Until every setter and index path is audited, keep:

    Kit/Vulkan: GPU
    SRB environment and PhysX-compatible tensors: CPU
    NMPC/acados/OSQP: CPU

The desired future split is GPU for ACT/DP/SB3 neural-network forward/backward computation, with CPU SRB/PhysX and explicit conversion of actions before env.step(). Add an explicit policy-device option rather than forcing env.sim.device=cuda:0. Keep VecEnv observations/rewards/dones CPU/NumPy-compatible.

## SRB modification priorities

Prioritize these NMPC/SRB tasks over unrelated RL training. They are planned work, not completed features. Research context: docs/src/workflows/nmpc_experiments_and_research_direction_20260909.md. Known model mismatch, KOZ sizing, fuel efficiency and P99 limits remain unresolved.

### 1. Model and task consistency

- Decide whether the task is HCW/LVLH or true free-floating; do not silently mix n=0.001027 and n=0.
- Connect actual wet/dry mass, fuel evolution, inertia, COM, and actual 16-RCS mapping to the expert, or model their uncertainty.
- Pass absolute episode time when using time-varying references; the current reference cache covers only the 0..1 s horizon.
- Derive KOZ geometry from vehicle dimensions, reference point, uncertainty, and approach phase. Radius 0.5 m is an experiment default, not a traced engineering specification.

### 2. Expert quality and validation

- Run horizon, braking-reference, and cost-weight ablations to reduce overshoot.
- Stabilize the native large-attitude goal and record quality-gate rejection causes.
- Add multiple initial states, velocities, attitudes, disturbances, and actuator/model perturbations.
- Measure wall-clock P50/P99 with explicit delay; synchronous stepping is not a real-time guarantee.
- Save actual closed-loop executed state/action sequences for demonstrations; predicted future actions are not necessarily later executed after feedback.

### 3. Imitation learning

- Establish single-step BC and fixed-chunk ACT baselines; add DP after ACT is reproducible.
- Treat DAgger as a data-collection baseline, not automatically the contribution.
- Study dynamics- and delay-aware action-prefix selection: execute the longest prefix that remains safe and recoverable, and query NMPC before braking/recovery feasibility is lost.
- Compare fixed chunks, risk-triggered correction, adaptive horizon, safety filtering, and recoverability criteria under identical demonstrations and budgets.

## Checks

CUDA visibility through Isaac Sim Python:

    cd /home/ubuntu/yyf/space_robotics_bench_0.0.5
    source ./srb_0.0.5_env.sh
    env -u CUDA_VISIBLE_DEVICES \
    /home/ubuntu/isaac-sim-4.5/python.sh -c \
    'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'

Vulkan is a separate check:

    vulkaninfo --summary

Find decisive Kit messages:

    rg -n -i 'CUDA error|bad state|No device could|no suitable|GPU Foundation|vkCreateDevice|GLFW|GLInterop' /tmp/nmpc_*.log

For RL experiments, read AGENTS-RL.md first and use scripts/srb_rl_agent.py where applicable. The --gpu/CUDA_VISIBLE_DEVICES examples in AGENTS-RL.md are historical RL workflow instructions, not proof that masked Kit initialization is currently reliable. Inspect that entry point before adapting GPU selection; do not blindly transfer the NMPC command or force env.sim.device=cuda. This update does not change the RL entry point.

## Reporting requirements

Every experiment report must include the exact command and working directory; Kit device and CUDA visibility; environment device, PhysX path, and --physics-only status; task, target, initial state, model/actuator parameters, horizon, period and seed; requested versus executed steps; success criteria and safety metrics; solver P50/P99 and deadline misses; output/log paths; failures, warnings, and what the result does not prove.

Do not commit, push, or modify unrelated SRB 0.0.6 files unless explicitly requested.
