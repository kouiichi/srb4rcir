# DESDynI 服务星：底部封闭、储箱与 RAFTI 接口修改方案

## 任务目标与当前交付

将 NASA DESDynI 外观纳入 SRB 服务星候选范围。在其底部凹入区域建立封闭外壳与独立储箱布局，并在封板外侧安装 RAFTI 参考接口，供后续卫星逼近及加注任务使用。

当前交付是原始模型、可编辑工程、视觉 USD、底部几何评估和修改方案。封板、储箱和 RAFTI 尚未添加到模型；当前实验资产与控制器保持原配置。后续线程可以从本文件独立开展模型修改，不需要读取此前聊天。

工作目录：`/home/ubuntu/yyf/space_robotics_bench_0.0.5`。

候选资产目录：`assets/nasa_candidates/desdyni/`。它独立于 `assets/srb_assets` 子模块，不修改已有第三方资产。

## 输入资产与来源

| 仓库相对路径 | 内容 |
|---|---|
| `assets/nasa_candidates/desdyni/source/desdyni_nasa.glb` | NASA 原始模型 |
| `assets/nasa_candidates/desdyni/visual/desdyni_original.blend` | Blender 4.3.2 原模型工程 |
| `assets/nasa_candidates/desdyni/visual/desdyni_original.usdc` | 视觉 USD |
| `assets/nasa_candidates/desdyni/source/rafti_DOC-00133A.pdf` | 用户提供的 RAFTI 尺寸资料 |
| `assets/nasa_candidates/desdyni/candidate_manifest.json` | 来源、校验值、验证范围 |
| `assets/nasa_candidates/desdyni/bottom_closure_proposal.json` | 封板候选位置、轮廓及接口安装约定 |
| `assets/nasa_candidates/desdyni/measurements/bottom_inspection.json` | 底部材质分组、射线测量和轮廓候选 |
| `assets/nasa_candidates/desdyni/previews/` | 整星、底部、修改示意图 |

原始模型来源：[NASA DESDynI 模型页](https://science.nasa.gov/3d-resources/deformation-ecosystem-structure-and-dynamics-of-ice-desdyni/)，署名 NASA/Christopher R. Meaney。NASA 3D Resources 的使用说明另见其 [README](https://github.com/nasa/NASA-3D-Resources/blob/master/README.md)。本研究构型不是 NASA 原始飞行配置或厂家服务星产品。

原始 GLB 的 SHA-256：`fef03075b3b87da9c35e4022a5c39b630404b571d1114b97ca34d7159a9424f1`。

当前视觉模型有 2 组网格、67,580 个三角面、32 个材质。USD 默认 prim 为 `/Satellite`，Z-up；源坐标整体包络约 `4.790 × 21.281 × 10.201`。`metersPerUnit=1` 是导出元数据，实际尺度尚未校准。

## 底部几何已经确认的内容

本文件将 Blender 源坐标 **−Z 方向**定义为底部操作方向，太阳翼主要向 ±Y 展开。不要用太阳翼的整体包围盒推断机身或储箱尺寸。

`Layer_0` 包含机身与外部结构；`Layer_1` 包含太阳翼及其支撑。两者不是完整的工程零件树。

底部局部视图显示：

- 机身底端有近八边形开口及侧壁。
- 后方有碟状曲面，材质槽名为 `DESDYNI-dish.003`，中心存在向开口伸出的柱状部件。
- 因此该区域不是没有内部结构的空腔；材质名和外观仅用于定位，不据此认定真实工程功能。
- 原模型有许多开放边；即使添加一块视觉封板，也不能把整星网格当作已闭合的储箱。

底部轮廓候选来自 `Layer_0` 最小 Z 附近顶点的 XY 投影凸包。选择容差为 `0.0005` 源坐标单位，共 36 个去重投影点。这个凸包用于定位和初步封板外形；正式建模应追踪实际接缝边界，不能把凸包视为已验证的密封轮廓。

| 几何量 | 测量结果，均为源坐标单位 |
|---|---|
| 底端候选平面 | `Z = −5.100305557` |
| 候选轮廓 X 范围 | `−2.035970 … 2.005981` |
| 候选轮廓 Y 范围 | `−2.020952 … 2.021000` |
| 候选轮廓中心 | `X ≈ −0.0149945, Y ≈ 0.000024` |
| 候选轮廓宽度 | `4.041951 × 4.041952` |
| 碟状曲面 Z 范围 | `−3.179711 … −2.999111` |
| 沿 `(X,Y)=(0,0)` 从 −Z 射入的第一表面 | `Z ≈ −4.589625` |
| 上述中心射线表面距封板候选平面 | `约 0.510680` |

中心射线不严格等于偏移约 0.015 的候选安装中心，也不证明有限直径管段的完整净空。后续必须在实际接口安装点和管段包络上再次检查。

## 封板、储箱和接口的修改设计

### 外壳封闭

保留原始 GLB、BLEND 和 USD，以派生文件进行修改。新增独立 `BottomCover`，沿底端实际轮廓形成带厚度的近八边形封板；其内侧安装面初步取 `Z=−5.100305557`，板厚沿 −Z 伸出。封板边界与侧壁的连接另建安装结构，检查穿插、缝隙和表面法向。

参数文件中的 `5 mm` 板厚仅是布局起点，不是已计算的结构承载或压力壳厚度。若选定比例 `s`，单位为米/源坐标单位，则：

```text
坐标（m） = s × 源坐标
封板厚度（源坐标） = 0.005 / s
封板外侧安装面 Z = −5.100305557 − 封板厚度（源坐标）
```

实际尺度选定前，不固定米制储箱容量或接口与机身的相对尺寸。可选择公开工程尺寸校准，或明确采用一套研究尺寸；后者不能描述为 DESDynI 的真实工程尺寸。

### 内部储箱

储箱单独建立，给出外形、壁厚或等效质量、容量、介质密度假设、空箱与介质质量、COM 和惯量。底部外壳提供布局与安装空间，不能通过“填满空腔”直接得到真实储箱。

先在局部视图中分离或定位碟状曲面与中心部件。对于服务星改装，可以用新的储箱组件替换选定内部部件；若保留它们，则储箱和管路需要绕开其实际包络。两种处理都应在派生资产中显式定义，不删除原始资料。

不要按材质名批量删除全部机身部件。碟状曲面可由 `Layer_0` 的 `DESDYNI-dish.003` 材质区域定位；中心部件与其他零件共享材质，应通过连通部件和空间范围逐一分离。

### 底部 RAFTI 接口

RAFTI Rev A 是被动流体服务阀与抓取接口，可作为 DESDynI 自身储箱的补给/排放口。主动抓取与流体连接装置属于另一侧配合件；该服务星若还需主动向目标输送介质，需另定义主动连接器或末端工具。

用户提供的 PDF 第 2 页约束以下外形：

| 图纸量 | 原值 | 米制 |
|---|---|---|
| 正面方形外包络 | `3.780 in` | `96.012 mm` |
| 图示主体高度 | `1.900 in` | `48.260 mm` |
| 两处后侧管段图示长度 | `2.032 in` | `51.6128 mm` |

图示主体高度不直接等于安装面到配合面的距离。保护结构、安装法兰和后侧管段应分别建模，不能仅添加圆环。孔位、阀芯中心距、配合面与主动端完整 CAD 尚未取得；自行定义的部分应标为研究参数。该 PDF 的压力和流量参数不作为本次几何模型的仿真性能。

候选安装位置取封板轮廓中心，接口局部 **+Z 朝机身外，即源坐标 −Z**；局部 +X 沿源 +X。这个坐标约定对应 scalar-first quaternion `[0,1,0,0]`，使局部 +Y 沿源 −Y。

建议分别定义 `FrameMount`、`FrameMate` 和两个流体连接参考系。`FrameMount` 是研究安装面；`FrameMate` 的位置必须与实际重建几何一致，不能默认与安装面重合。

后侧管段伸向机身内部。当前中心射线净空约为 `0.510680 × s m`；需要检查管段、接头、弯管和储箱入口是否与现有中心部件相交。上述单射线结果只用于初筛。

## 派生资产与 SRB 接入要求

建议派生资产保留以下逻辑分组：

```text
DesdyniService
  Body
    Visuals
    BottomCover
    TankAssembly
    PassiveRefuelingPort
      FrameMount
      FrameMate
      FrameFluidA
      FrameFluidB
    Colliders
    CameraMount
```

在当前几何范围内，封板、储箱与固定接口可属于同一个卫星刚体；仅在明确需要运动的机构上定义关节。物理配置中避免对整星展示网格统一按密度估算质量，避免在接口孔内生成实心凸包。

后续 SRB 接入参考文件：

- `srb/assets/robot/mobile/spacecraft.py`：新增显式研究服务星资产，使用 UsdFileCfg + RigidObjectCfg。
- `srb/tasks/mobile/rendezvous/task.py` 和 `task_visual.py`：在配置初始化时选择服务星、目标星与相机。
- `srb/nmpc/surface_approach.py` 和 `scripts/validate_nmpc_replay.py`：从显式接口坐标生成目标，安全几何使用指定集合。
- `srb/core/action/term/mobile/thrust.py`：保留干质量基线、燃料和湿质量 reset 约定；新增储箱的干质量应在初始化前计入。转移燃料与 RCS 燃料是否共用需要显式定义。

当前任务只验证接口前逼近。现有接触即停止、球形 KOZ 和支撑平面检查不能直接作为插接成功判据。扩大外形和质量后，起点、制动能力与安全通道需要重新验证；不能沿用 CubeSat 几何或执行器数值冒充新卫星参数。

若开展自由飞行 NMPC 验证，使用项目约定的 matched free-floating、N=40/K=3、dt=0.05 s、GPU Kit / CPU PhysX 和 physics-only；记录新模型、质量与 RCS 参数。渲染验证单独进行。

## 后续线程的具体任务与验收

后续线程应先读取本文件、`bottom_closure_proposal.json` 和底部视图，然后完成：

1. 选定并记录模型尺度；识别真实开口边界与中心部件。
2. 在派生 BLEND 中分离操作区域，添加带厚度封板和安装结构。
3. 明确内部部件保留或替换方式，建立独立储箱与管路占位。
4. 在封板外侧添加 RAFTI Rev A 外包络约束的研究接口，并输出安装/配合参考系。
5. 导出独立 USD，提供封闭前后对照、底部正视、接口特写与局部剖面。

几何验收要求：原始模型不变；派生模型尺度和坐标约定明确；新封板和储箱各自闭合且法向正确；接缝、管段与内部零件无未说明穿插；接口外包络与记录参数一致；来源尺寸和研究参数分别标明。SRB 加载、PhysX、动力学和闭环测试应在报告中分别记录，不以 Blender 预览代替。

建议输出命名：`visual/desdyni_service_bottom_closed.blend`、`visual/desdyni_service_bottom_closed.usdc`、接口参数文件和修改说明。这些文件当前尚未生成。

可直接发送给后续线程的任务文本：

> 请在 `/home/ubuntu/yyf/space_robotics_bench_0.0.5` 读取 `assets/nasa_candidates/desdyni/HANDOFF.md`，使用目录中的原始 GLB、BLEND、USD 和 RAFTI Rev A PDF，完成 DESDynI 研究服务星的底部封板、独立储箱布局及被动加注接口几何。先明确实际尺度与内部碟状曲面/中心部件的处理方式，使用派生资产保留原文件，输出可编辑 BLEND、USD、参考坐标和修改前后视图。当前先完成几何资产，不修改已有实验控制器，不执行燃料转移或完整加注仿真。

## 复现检查与验证范围

Blender 源模型检查及整星预览：

```sh
cd /home/ubuntu/yyf/space_robotics_bench_0.0.5
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/blender-4.3.2-linux-x64/blender -b --factory-startup -t 2 \
  --python-exit-code 1 --python assets/nasa_candidates/desdyni/tools/inspect_desdyni.py -- \
  --source assets/nasa_candidates/desdyni/source/desdyni_nasa.glb \
  --output /tmp/desdyni_inspection \
  --blend /tmp/desdyni_original.blend
```

底部射线与轮廓测量，追加 `--measure-only` 可跳过渲染：

```sh
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /home/ubuntu/blender-4.3.2-linux-x64/blender \
  -b assets/nasa_candidates/desdyni/visual/desdyni_original.blend -t 2 \
  --python-exit-code 1 --python assets/nasa_candidates/desdyni/tools/analyze_bottom.py -- \
  /tmp/desdyni_inspection --measure-only
```

已完成：原始 GLB 校验与入库；Blender 解码、可编辑工程保存；CPU Cycles 整星和底部渲染；材质/空间分组与轴向射线测量；轮廓候选与封底方案输出。实际检查命令和日志位于 `logs/desdyni_service_candidate_20261013/`。USD 转换与读取依据为 `logs/nasa_glb_srb_probe_20261013/README.md`，本目录 USD 的物理 API 另由入库检查核对。

入库 USD 检查命令：

```sh
./srb_0.0.5_env.sh -- env \
  PYTHONPATH=/home/ubuntu/isaac-sim-4.5/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310 \
  LD_LIBRARY_PATH=/home/ubuntu/isaac-sim-4.5/extscache/omni.usd.libs-1.0.1+d02c707b.lx64.r.cp310/bin \
  /home/ubuntu/isaac-sim-4.5/python.sh \
  assets/nasa_candidates/desdyni/tools/audit_candidate.py assets/nasa_candidates/desdyni
```

检查结果：文件校验值一致；USD 默认 prim `/Satellite`、Z-up；2 组网格、67,580 个三角面、32 个材质；刚体与碰撞 API 数量均为 0；没有外部贴图引用。检查结果保存为 `measurements/library_audit.json`。

未完成：实际尺度校准、封板/储箱/接口建模、SRB 机器人注册、Kit 渲染、PhysX 碰撞与 NMPC 闭环。没有新的接触力、轨迹成功率或求解耗时指标；本次使用 Blender CPU 渲染，不启动 Kit/PhysX。
