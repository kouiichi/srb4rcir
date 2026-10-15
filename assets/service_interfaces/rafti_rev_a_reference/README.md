# RAFTI Rev A 参考接口资产

本目录提供依据 RAFTI Rev A 正视图、侧视图和实物照片重建的被动服务接口。资产以米为单位，适合引用到卫星机身或封板上。图纸标注约束外包络，未标注的内部结构和配合位置采用明确的研究参数。

## 模型内容与尺寸

模型包含八边形主体、四个安装耳及八个贯通孔、双阀口与凹入阀座、紧固件、后侧双管与接头。前端阀座按关闭状态表现。

| 参数 | 数值 | 依据 |
|---|---:|---|
| 正面 X/Y 外包络 | 96.012 mm | 图纸 3.780 in |
| 主体前向高度 | 48.260 mm | 图纸 1.900 in |
| 后侧管段长度 | 51.6128 mm | 图纸 2.032 in |
| 模型总轴向包络 | 99.8728 mm | 上述前向、后向包络相加 |
| 两阀口中心距离 | 30 mm | 图像比例重建参数 |
| 安装耳厚度、孔径、管径 | 见 reference_parameters.json | 研究重建参数 |

原始图纸保存在 `source/rafti_DOC-00133A.pdf`，用户提供的尺寸截图保存在 `source/dimension_reference.png`。图纸没有给出完整制造公差、内部阀机构或主动端 CAD；本模型用于外观、布局及接口坐标研究，配合几何按研究参数定义。

## 文件

| 文件 | 用途 |
|---|---|
| `rafti_reference.blend` | 可编辑 Blender 4.3.2 工程，37 个网格对象 |
| `rafti_reference.usdc` | 独立视觉 USD，默认 prim `/Rafti`，米制、Z-up |
| `reference_parameters.json` | 图纸尺寸、重建参数和参考坐标 |
| `preview_front.png`、`preview_side.png`、`preview_oblique.png`、`preview_rear.png` | 模型视图 |
| `usd_audit.json` | USD 尺寸与引用组合检查 |
| `blender_geometry_audit.json` | 阀口、安装孔及后管开口检查 |
| `tools/build_rafti_reference.py` | 根据参数生成 BLEND、USD、预览 |
| `tools/audit_rafti_asset.py` | 使用 Isaac Sim USD 库检查资产 |
| `tools/audit_rafti_blend.py` | 使用 Blender 射线检查开口 |

## 挂载坐标

局部 +Z 为接口外向方向，后侧管段伸向 −Z。安装原点定义为研究模型的主体后平面。

```text
/Rafti
  /Visuals
  /Frames
    /Mount    (0, 0, 0)
    /Mate     (0, 0, 0.048260)
    /FluidA   (-0.015, 0, 0.048260)
    /FluidB   ( 0.015, 0, 0.048260)
```

这些是研究安装与前端参考平面，不是厂家已经核实的配合基准。流体参考系只表示几何位置。

要在 DESDynI 底部安装，使局部 +Z 指向卫星源坐标 −Z，可使用绕 X 轴 180 度的旋转，即 wxyz quaternion `[0,1,0,0]`。模型应挂在已校准、尺度为 1 的机身坐标下；NASA 视觉子树的缩放单独处理，避免把 96 mm 接口随整星源模型一起缩放。

在已有 USD stage 中可按下面的方式添加引用，`mount_position_m` 使用选定封板安装点的米制坐标：

```python
from pxr import Gf, UsdGeom

port = UsdGeom.Xform.Define(stage, body_path + "/RefuelingPort")
port.GetPrim().GetReferences().AddReference(
    "/home/ubuntu/yyf/space_robotics_bench_0.0.5/assets/service_interfaces/"
    "rafti_rev_a_reference/rafti_reference.usdc"
)
port.AddTranslateOp().Set(Gf.Vec3d(*mount_position_m))
port.AddOrientOp().Set(Gf.Quatf(0, Gf.Vec3f(1, 0, 0)))
```

资产没有单独刚体、碰撞体或燃料转移行为。固定挂载时可归属于卫星刚体；其质量、碰撞代理及后管穿过封板的孔洞由装配配置定义。RAFTI 是被动服务口，配套主动连接器需要单独建模。

## 生成与检查

工作目录：`/home/ubuntu/yyf/space_robotics_bench_0.0.5`。

```sh
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/blender-4.3.2-linux-x64/blender -b --factory-startup -t 2 \
  --python-exit-code 1 \
  --python assets/service_interfaces/rafti_rev_a_reference/tools/build_rafti_reference.py -- \
  --output assets/service_interfaces/rafti_rev_a_reference
```

生成脚本会写入并更新该输出目录的模型、参数与预览。检查命令：

```sh
./srb_0.0.5_env.sh -- env \
  PYTHONPATH=/home/ubuntu/isaac-sim-4.5/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310 \
  LD_LIBRARY_PATH=/home/ubuntu/isaac-sim-4.5/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin \
  /home/ubuntu/isaac-sim-4.5/python.sh \
  assets/service_interfaces/rafti_rev_a_reference/tools/audit_rafti_asset.py \
  assets/service_interfaces/rafti_rev_a_reference

env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/blender-4.3.2-linux-x64/blender \
  -b assets/service_interfaces/rafti_rev_a_reference/rafti_reference.blend -t 2 \
  --python-exit-code 1 \
  --python assets/service_interfaces/rafti_rev_a_reference/tools/audit_rafti_blend.py -- \
  assets/service_interfaces/rafti_rev_a_reference
```

已验证图纸外包络、参考坐标、USD 引用组合、八个安装孔贯通、双阀口凹入和后管空心。预览由 Blender CPU Cycles 生成；Kit 渲染、PhysX 接触和流体连接行为分别需要在装配任务中验证。
