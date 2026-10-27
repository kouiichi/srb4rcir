# NEAR Shoemaker assets

This directory contains the source NASA visual asset and the reproducible
target variant used by the current rendezvous geometry.

| File | Use |
|---|---|
| `visual/near_shoemaker_original.usdc` | Immutable source visual USD |
| `visual/near_shoemaker_antenna_boss.usdc` | Derived visual and collision source |
| `visual/near_shoemaker_antenna_boss.metadata.json` | Source hash and geometry audit |
| `visual/textures/` | Source textures referenced by both USD assets |

The derived variant removes the nine central antenna/support connected
components and adds `/Satellite/antenna_replacement_boss`. Its square side
follows the measured central bus top deck and its source-coordinate height is
one twentieth of the bus height. The boss bottom is coincident with the
existing NEAR top reference plane; the boss top is the current rendezvous
target plane.

The source model has 9,312 triangles, four materials, and an overall source
envelope of approximately `5.294 x 5.294 x 2.865` units. Source coordinates
are not a flight-scale calibration. Mass, inertia, collision policy, and
actuator parameters remain owned by the separate NMPC physics configuration.

The derived asset is generated with:

```text
source ./srb_0.0.5_env.sh
export PYTHONPATH="/home/ubuntu/isaac-sim-4.5/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310:$PYTHONPATH"
export LD_LIBRARY_PATH="/home/ubuntu/isaac-sim-4.5/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin:$LD_LIBRARY_PATH"
/home/ubuntu/isaac-sim-4.5/python.sh scripts/build_near_antenna_variant.py
```

The original USD and its textures are not modified.
