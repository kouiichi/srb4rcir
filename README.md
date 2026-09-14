<h1 align="center">Space Robotics Bench</h1>

<p align="center">
  <a href="https://AndrejOrsula.github.io/space_robotics_bench"><img alt="" src="https://github.com/user-attachments/assets/049289be-0c99-497b-be37-c4975d924524" width="100%"></a>
</p>

[![Discord](https://img.shields.io/badge/Discord-invite-5865F2?logo=discord)](https://discord.gg/p9gZAPWa65)
[![Docs](https://img.shields.io/badge/docs-online-blue?logo=markdown)](https://AndrejOrsula.github.io/space_robotics_bench)
[![Rust](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/rust.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/rust.yml)
[![Python](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/python.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/python.yml)
[![Docker](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docker.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docker.yml)
[![Docs](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docs.yml/badge.svg)](https://github.com/AndrejOrsula/space_robotics_bench/actions/workflows/docs.yml)

<!-- [![Codecov](https://codecov.io/gh/AndrejOrsula/space_robotics_bench/graph/badge.svg)](https://codecov.io/gh/AndrejOrsula/space_robotics_bench) -->

**Space Robotics Bench (SRB)** is a comprehensive collection of environments and tasks for robotics research in the challenging domain of space. It provides a unified framework for developing and validating autonomous systems under diverse extraterrestrial scenarios. At the same time, its design is flexible and extensible to accommodate a variety of development workflows and research directions beyond Earth.

**srb4rcir**:Due to issues such as server version compatibility, only version 0.0.5 of srb could be used; furthermore, as modifications were required, I created a version optimized for execution on the rcir server to facilitate future research using this simulation.



## Key Features

- **Parallelized Simulation**: Highly parallelized simulation instances for accelerated workflows
- **Procedural Generation**: On-demand generation of diverse simulation assets and scenes
- **Domain Randomization**: Extensive randomization for robustness and generalization
- **Gymnasium API**: Compatibility with standard API and frameworks for robot learning
- **ROS 2 Interface**: Seamless interoperability with ROS 2 and Space ROS ecosystems
- **Abstract Architecture**: Flexibility across different robots and space domains

## Documentation

SRB documentation with detailed installation instructions, usage guides, and development resources is available [online](https://AndrejOrsula.github.io/space_robotics_bench).

## 20 Hz NMPC expert

The standalone privileged expert is under `srb/nmpc/`.  It uses a 13-state
HCW/CW plus quaternion rigid-body model, a full-rank 16-channel one-sided RCS,
`CasADi+IPOPT` for the high-accuracy oracle, and `acados+HPIPM` for the
production backend.  The global grid is 20 Hz (`Ts=0.05 s`, `N=20`, five
`0.01 s` RK4 substeps, `K=8` action blocks).

Run the checks through the SRB wrapper:

```bash
./srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh scripts/validate_rcs.py
./srb_0.0.5_env.sh -- /home/ubuntu/isaac-sim-4.5/python.sh scripts/smoke_test.py --config configs/default.yaml
```

The current SRB Cubesat remains an explicitly labelled rank-5, eight-thruster
contrast adapter; it is not used to create the main 6-D expert labels.  See
`third_party/acados/` and `srb/nmpc/THIRD_PARTY_NOTICES.md` for the pinned solver build
and license provenance.

An opt-in `srb/rendezvous_16rcs` task uses the canonical full-rank 16-channel
RCS configuration.  It fixes the actuator-rank mismatch while preserving the
legacy 8-thruster task for contrast; SRB's current rendezvous physics remains
zero-gravity, so HCW/CW expert trajectories continue to come from `srb/nmpc/`.

<div align="right">
<a href="https://AndrejOrsula.github.io/space_robotics_bench"><img alt="Documentation" src="https://github.com/user-attachments/assets/c8663796-3ef1-4ff7-860b-cf8080d0a07a" width="96" height="96"></a>
</div>

## License

This project is dual-licensed under either the [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) licenses.

All assets created by contributors of this repository and those generated from [SimForge](https://github.com/AndrejOrsula/simforge) procedural pipelines are licensed under the [CC0 1.0 Universal](https://github.com/AndrejOrsula/srb_assets/blob/main/LICENSE-CC0) license. Resources from third-party sources are listed under [attributions](https://andrejorsula.github.io/space_robotics_bench/misc/attributions.html).

[![CC0 1.0 Universal](https://licensebuttons.net/l/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0)
