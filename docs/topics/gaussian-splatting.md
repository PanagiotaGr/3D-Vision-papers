# Gaussian Splatting & 3DGS

_Updated: 2026-02-16 07:19 UTC_

Total papers shown: **1**


---

- **GSM-GS: Geometry-Constrained Single and Multi-view Gaussian Splatting for Surface Reconstruction**  
  Xiao Ren, Yu Liu, Ning An, Jian Cheng, Xin Qiao, He Kong  
  _2026-02-13_ · https://arxiv.org/abs/2602.12796v1  
  <details><summary>Abstract</summary>

  Recently, 3D Gaussian Splatting has emerged as a prominent research direction owing to its ultrarapid training speed and high-fidelity rendering capabilities. However, the unstructured and irregular nature of Gaussian point clouds poses challenges to reconstruction accuracy. This limitation frequently causes high-frequency detail loss in complex surface microstructures when relying solely on routine strategies. To address this limitation, we propose GSM-GS: a synergistic optimization framework integrating single-view adaptive sub-region weighting constraints and multi-view spatial structure refinement. For single-view optimization, we leverage image gradient features to partition scenes into texture-rich and texture-less sub-regions. The reconstruction quality is enhanced through adaptive filtering mechanisms guided by depth discrepancy features. This preserves high-weight regions while implementing a dual-branch constraint strategy tailored to regional texture variations, thereby improving geometric detail characterization. For multi-view optimization, we introduce a geometry-guided cross-view point cloud association method combined with a dynamic weight sampling strategy. This constructs 3D structural normal constraints across adjacent point cloud frames, effectively reinforcing multi-view consistency and reconstruction fidelity. Extensive experiments on public datasets demonstrate that our method achieves both competitive rendering quality and geometric reconstruction. See our interactive project page

  </details>


