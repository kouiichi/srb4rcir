# DESDynI 候选服务星资产

本目录保存 NASA DESDynI 原始外观模型、Blender 可编辑工程、转换后的视觉 USD，以及底部封闭的几何评估。来源署名：NASA/Christopher R. Meaney。

当前状态：已作为候选资产入库；没有注册 SRB 机器人类，没有替换当前逼近任务。原始模型保持不变，底部封板、储箱和 RAFTI 接口均尚未写入模型。封底方案位于 `bottom_closure_proposal.json` 和 `HANDOFF.md`。

可复用的 RAFTI 被动接口参考资产位于 `assets/service_interfaces/rafti_rev_a_reference/`，提供独立 BLEND、USD 和挂载参考系。装到本模型上前，应确定米制尺度、封板安装点和后管净空。

| 文件 | 用途 |
|---|---|
| `source/desdyni_nasa.glb` | 原始 NASA GLB |
| `visual/desdyni_original.blend` | Blender 4.3.2 可编辑工程；保留原始两组网格与材质 |
| `visual/desdyni_original.usdc` | 视觉 USD，默认 prim `/Satellite`，Z-up |
| `source/rafti_DOC-00133A.pdf` | 用户提供的 RAFTI Rev A 数据表 |
| `candidate_manifest.json` | 来源、校验值和资产验证范围 |
| `measurements/` | 网格边界、底部射线和轮廓候选数据 |
| `previews/` | 原模型整星、底部视图和封底方案示意 |
| `tools/` | Blender 几何检查和预览脚本 |
| `HANDOFF.md` | 独立修改方案与后续工作要求 |

模型有 2 组网格、67,580 个三角面、32 个材质。源坐标整体外包络约 `4.790 × 21.281 × 10.201`；这不是已校准的实星米制尺寸。物理属性、质量、惯量、RCS 与碰撞体均未定义。

模型来源：[NASA 模型页面](https://science.nasa.gov/3d-resources/deformation-ecosystem-structure-and-dynamics-of-ice-desdyni/)。使用说明：[NASA 3D Resources README](https://github.com/nasa/NASA-3D-Resources/blob/master/README.md)。本目录的 NASA 模型及用户提供 PDF 不自动继承 `assets/srb_assets` 子模块的 CC0 声明。
