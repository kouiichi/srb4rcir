# DESDynI 候选服务星资产

本目录提供用于仿真的 NASA DESDynI 视觉 USD 和喷口几何参数。来源署名：NASA/Christopher R. Meaney。

可通过 `srb/tasks/mobile/rendezvous/nasa_pair.py` 的显式配置与原始 NEAR Shoemaker 组成逼近任务。运行入口为 `scripts/validate_nmpc_nasa_pair.py`，研究参数位于 `configs/nmpc/desdyni_near_research.json`。视觉模型没有增加底部封板、储箱或 RAFTI 接口。

可复用的 RAFTI 被动接口参考资产位于 `assets/service_interfaces/rafti_rev_a_reference/`，提供独立 USD 和挂载参考系。装到本模型上前，应确定米制尺度、封板安装点和后管净空。

| 文件 | 用途 |
|---|---|
| `visual/desdyni_original.usdc` | 视觉 USD，默认 prim `/Satellite`，Z-up |
| `measurements/thruster_geometry.json` | 运行时读取的原模型四喷口位置、方向与几何参数 |

模型有 2 组网格、67,580 个三角面、32 个材质。源坐标整体外包络约 `4.790 × 21.281 × 10.201`；这不是已校准的实星米制尺寸。物理属性、质量、惯量、RCS 与碰撞体均未定义。

上述物理属性缺口指原始视觉 USD；运行配置在派生 USD 中单独定义刚体和碰撞属性。`measurements/thruster_geometry.json` 记录 `DESDYNI-thrusters.003` 材质对应的 4 个喷口及拟合轴线。四喷口基线采用近似 +Z 排气轴与单向占空比，不将外观喷口数量解释为完整实星 RCS 规格。

可选的 `configs/nmpc/desdyni_near_rcs12_research.json` 配置保留原 4 路，追加 4 路反向轴向喷口和 4 路侧向喷口，共 12 路。`virtual_thrusters` 的位置为机体坐标系米制坐标，方向为单位排气轴；`thruster_power_N` 按原 4 路、追加 8 路的顺序给出额定推力。规划器与 PhysX 共用这组力臂、方向、额定推力和耗油律。新增通道属于虚拟 RCS 研究模型，未增加网格或硬件质量，质量、惯量、COM 和燃料容量保持显式基线值；这不代表原 NASA 卫星的推进系统规格。

双星入口默认采用同步逼近：位置与姿态共享 50 s 的五次进度曲线，边平移边转身，采用 N40/K3、0.05 s 控制周期。`--video --record-camera` 可添加虚拟 RGB/深度相机；默认固定于底面外侧，640×480、水平视场 90°，光轴沿机体 −Z，初始姿态下目标不在其视野内。`--camera-tracking limited-gimbal` 使用 +X/−Z 外侧角点的双轴运动学支架，偏航范围 ±120°、俯仰范围 ±80°，各轴最大转速 30°/s，复位时对准目标。其跟踪输入为模拟器目标位置；没有硬件网格、质量、电机动力学或视觉状态估计。

`--camera-fov-gate` 在预测节点检查有限角速支架下完整目标凸包的矩形图像投影，要求所有顶点在相机前方且满足图像边界余量；不满足时拒绝执行该计划。该检查独立于 NMPC 优化器，不会重新优化轨迹，也不保证无遮挡。真实渲染另存语义可见性、同步误差、逐帧机体到相机的外参及云台角度/角速度。`--camera-save-every 1` 在每个控制步采集图像，供 20 Hz 数据对齐。

`desdyni_near_scale2_massfixed_research.json` 与 `desdyni_near_scale2_mass8_research.json` 是相同几何、相同推力的质量对照配置：DESDynI 采用 0.06 m/源单位，NEAR 采用 0.2 m/源单位，全部 12 路额定推力均为 0.40 N。两星基础/干质量分别保持 6.108697891 kg 或乘以 8；DES 燃料均为 5 kg。惯量按质量与线尺度平方缩放，再由执行器按湿/干质量比更新；虚拟喷口力臂同步放大。质量、惯量、推力和米制比例均为研究假设。原始外观文件不变，运行相机与物理属性在派生场景中定义。

模型来源：[NASA 模型页面](https://science.nasa.gov/3d-resources/deformation-ecosystem-structure-and-dynamics-of-ice-desdyni/)。使用说明：[NASA 3D Resources README](https://github.com/nasa/NASA-3D-Resources/blob/master/README.md)。本目录的 NASA 模型不自动继承 `assets/srb_assets` 子模块的 CC0 声明。
