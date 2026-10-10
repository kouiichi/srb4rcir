# Agent Notes for SRB 0.0.5

This is the locally repaired SRB 0.0.5 checkout:

/home/ubuntu/yyf/space_robotics_bench_0.0.5

Current development focuses on: (1) validating and improving the NMPC trajectory expert in SRB; (2) making SRB dynamics, actuators, task references, and recorded actions consistent enough for ACT / Diffusion Policy imitation learning; and (3) keeping a reproducible GPU Kit / CPU PhysX environment split.

Updated: 2026-09-17. This file describes the current state, commands, and development boundaries. It is not an upstream SRB configuration.

Preserve existing local edits. Before changing code or committing, inspect git status --short and git diff; do not assume older lists of modified files still describe the checkout.

-所有文档和代码注释均描述当前状态，不得记录任何AI历史修改记录。更新当前状态描述不得将尚未确认的实现偏差写成新的业务规则。
-比如某次开发工作临时规划了phase1到phase5阶段，那么任何文档不允许记录phase1-phase5相关字段描述，只需要按照代码和项目现状更新文档内容即可。
-比如某次开发需求A决策是当前项目现状，B决策是这次开发任务新的决策，且A决策作废，那么文档记录就不能留下A决策的历史，只需要记录这次新的决策B。
## Standing user defaults for new NMPC experiments (2026-09-16)

Unless the user explicitly requests another setting, **plan 40 intervals and execute the first 3 successive planned actions before measured-state replanning (N=40, K=3)**. At dt=0.05 s this is a 2 s prediction horizon and a nominal 0.15 s replanning interval. K=3 does not mean apply only the first action for three steps, and does not change the physics/control period to 0.15 s. Optimizer move blocking is a separate concept.

Use the matched **free-floating** SRB closed loop (`--matched-executor`, n=0 on both sides), GPU Kit / CPU PhysX, and `--physics-only`. Keep the prior aligned initial state/goal unless explicitly changed. Pass `--horizon-steps 40 --execute-steps 3` in recorded commands even though the replay closed-loop CLI now defaults to these values. Explicit user requests and named ablations override these defaults; do not rewrite historical reports or archived plans.

The library's legacy HCW `TaskParams`/offline config and open-loop saved-plan replay are retained for historical reproduction. They are not the default choice for new SRB experiments. Do not infer current model parameters from the historical sections below: matched mode reads actual mass, full inertia, COM, fuel and RCS mapping.

Do not silently tune duty-rate limits, smoothness costs or jerk penalties when testing a dynamics/initialization fix. Record such changes as separate experimental factors.

### Wet-mass initialization/reset invariant (verified 2026-09-16)

`ThrustAction` now writes wet mass and inertia during construction and every reset, before the first force. PhysX mass must equal dry mass + remaining fuel. For the current vehicle: dry=6.108697891 kg, full fuel=5 kg, initial wet=11.108697891 kg. Inertia is restored from an immutable dry baseline with the existing proportional-to-mass approximation; this does not model tank geometry or shifting COM. Partial resets must leave unselected environments unchanged. Do not move fuel burn ahead of force submission without revalidating the matched executor. External mass/inertia domain randomization must explicitly reconcile ownership of this baseline; it was not validated here.

Before this fix, the first force used dry mass and the first fuel update then installed wet mass; reset also replenished fuel without synchronizing PhysX. Historical experiments retain that behavior in their recorded data and are not bit-for-bit reproducible with the repaired actuator.

Actual two-environment reset regression: `scripts/validate_nmpc_replay.py replay --plan logs/nmpc_validation_20260907/approach_hcw_plan --output NEW_DIRECTORY --physics-only --device cuda:1 --check-mass-reset` (use the wrapper and unmasked GPU prefix below). It tests constructor properties before the first reset, wet mass/inertia, repeated resets, first-force mass, partial/empty/tensor-index resets, and full environment reset after a fresh fuel burn. Final evidence: `logs/nmpc_wet_mass_20260916/reset_final_green/mass_reset_check.json`.

Same-condition N40/K3 before/after runs both completed 1200/1200 steps. After repair: strict hover 36.9 s, minimum KOZ clearance 0.642606 m, contact force 0 N. First sampled jerk fell from 0.656996 to 0.000508 m/s³, but whole-run jerk-norm RMS stayed approximately 0.108 m/s³: this is an initialization repair, not a smoothness solution. Hot solve P50/P99=60.30/153.37 ms; 1.253% exceeded the 150 ms replanning budget. Synchronous simulation does not establish real-time performance. Full commands, parameters, warnings, raw records and plots: `logs/nmpc_wet_mass_20260916/report.md`, `before_N40_K3/`, `after_N40_K3/`.

## Optional smoothness experiments (2026-09-17)

The replay CLI accepts `--duty-rate RATE --jerk-weight WEIGHT --jerk-scale 0.1`. Defaults remain rate20/s (from the historical plan) and jerk weight0; N40/K3 and matched free-floating remain the standing experiment defaults. Positive jerk weight requires matched mode and adds three previous-world-acceleration memory states to acados (34 internal states, public State13 unchanged). The first stage is anchored to the last actual 0.05 s velocity difference, not a saved predicted acceleration or a K×dt average. Initial pre-ignition acceleration is zero. Config/anchors are saved in `smoothness_config.json` and `predictions.npz`.

Approved eight-group evidence: `logs/nmpc_smoothness_ablation_20260917/report.md`. Baseline, rate4 and rate2 completed1200 steps; rate1 exhausted fuel and stopped837 steps. Jerk weights0.001/0.01/0.1 and rate2+jerk0.001 stopped744/678/831/855 steps due to acados status4 near hover. Saved-primal reproduction on fresh solvers yields QP MINSTEP; cold initialization at the same four states passes. No hidden recovery was added. These jerk configurations are experimental, not complete-success demonstrations. Compare common first30s windows, retain failure artifacts, and do not interpret early termination as efficiency/smoothness success.

The runner `scripts/run_nmpc_smoothness_ablation.py NEW_ROOT --phase single` performs sequential 30-step debug and1200-step attempts for seven baseline/single-factor settings; `--phase combo` uses the saved selection rule and adds the eighth. Use wrapper/Isaac Python and the GPU/CPU device rules below. Analysis/report: `scripts/analyze_nmpc_ablation.py ROOT`, `scripts/report_nmpc_ablation.py ROOT`. Preserve default controller settings unless the user explicitly adopts a candidate. Stabilizing the jerk-cost warm start and revalidating full closed-loop runs remains follow-up work.

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
    --align-goal \
    --matched-executor \
    --horizon-steps 40 --execute-steps 3

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
    --align-goal \
    --matched-executor \
    --horizon-steps 40 --execute-steps 3

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

The expert is an optimization controller, not a trained neural network. New SRB experiments use the standing matched free-floating N=40/K=3 defaults above. The following is the **historical HCW baseline contract**, retained to interpret old artifacts:

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

- Use true free-floating for new SRB experiments unless explicitly overridden; do not silently mix n=0.001027 and n=0.
- Preserve the matched executor's actual wet/dry mass, fuel evolution, inertia, COM and 16-RCS mapping. Validate any new geometry, fuel distribution or executor behavior against real steps.
- Pass absolute episode time for time-varying references. The current stationary matched experiment passes an explicit stationary COM reference and avoids reintegrating it from t=0 on each solve.
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

