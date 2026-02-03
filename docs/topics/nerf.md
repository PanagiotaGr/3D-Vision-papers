# NeRF & Neural Radiance Fields

_Updated: 2026-02-03 07:08 UTC_

Total papers shown: **3**


---

- **Learning Topology-Aware Implicit Field for Unified Pulmonary Tree Modeling with Incomplete Topological Supervision**  
  Ziqiao Weng, Jiancheng Yang, Kangxian Xie, Bo Zhou, Weidong Cai  
  _2026-02-02_ · https://arxiv.org/abs/2602.02186v1  
  <details><summary>Abstract</summary>

  Pulmonary trees extracted from CT images frequently exhibit topological incompleteness, such as missing or disconnected branches, which substantially degrades downstream anatomical analysis and limits the applicability of existing pulmonary tree modeling pipelines. Current approaches typically rely on dense volumetric processing or explicit graph reasoning, leading to limited efficiency and reduced robustness under realistic structural corruption. We propose TopoField, a topology-aware implicit modeling framework that treats topology repair as a first-class modeling problem and enables unified multi-task inference for pulmonary tree analysis. TopoField represents pulmonary anatomy using sparse surface and skeleton point clouds and learns a continuous implicit field that supports topology repair without relying on complete or explicit disconnection annotations, by training on synthetically introduced structural disruptions over \textit{already} incomplete trees. Building upon the repaired implicit representation, anatomical labeling and lung segment reconstruction are jointly inferred through task-specific implicit functions within a single forward pass.Extensive experiments on the Lung3D+ dataset demonstrate that TopoField consistently improves topological completeness and achieves accurate anatomical labeling and lung segment reconstruction under challenging incomplete scenarios. Owing to its implicit formulation, TopoField attains high computational efficiency, completing all tasks in just over one second per case, highlighting its practicality for large-scale and time-sensitive clinical applications. Code and data will be available at https://github.com/HINTLab/TopoField.

  </details>



- **ProxyImg: Towards Highly-Controllable Image Representation via Hierarchical Disentangled Proxy Embedding**  
  Ye Chen, Yupeng Zhu, Xiongzhen Zhang, Zhewen Wan, Yingzhe Li, Wenjun Zhang, Bingbing Ni  
  _2026-02-02_ · https://arxiv.org/abs/2602.01881v1  
  <details><summary>Abstract</summary>

  Prevailing image representation methods, including explicit representations such as raster images and Gaussian primitives, as well as implicit representations such as latent images, either suffer from representation redundancy that leads to heavy manual editing effort, or lack a direct mapping from latent variables to semantic instances or parts, making fine-grained manipulation difficult. These limitations hinder efficient and controllable image and video editing. To address these issues, we propose a hierarchical proxy-based parametric image representation that disentangles semantic, geometric, and textural attributes into independent and manipulable parameter spaces. Based on a semantic-aware decomposition of the input image, our representation constructs hierarchical proxy geometries through adaptive Bezier fitting and iterative internal region subdivision and meshing. Multi-scale implicit texture parameters are embedded into the resulting geometry-aware distributed proxy nodes, enabling continuous high-fidelity reconstruction in the pixel domain and instance- or part-independent semantic editing. In addition, we introduce a locality-adaptive feature indexing mechanism to ensure spatial texture coherence, which further supports high-quality background completion without relying on generative models. Extensive experiments on image reconstruction and editing benchmarks, including ImageNet, OIR-Bench, and HumanEdit, demonstrate that our method achieves state-of-the-art rendering fidelity with significantly fewer parameters, while enabling intuitive, interactive, and physically plausible manipulation. Moreover, by integrating proxy nodes with Position-Based Dynamics, our framework supports real-time physics-driven animation using lightweight implicit rendering, achieving superior temporal consistency and visual realism compared with generative approaches.

  </details>



- **Radioactive 3D Gaussian Ray Tracing for Tomographic Reconstruction**  
  Ling Chen, Bao Yang  
  _2026-02-01_ · https://arxiv.org/abs/2602.01057v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged in computer vision as a promising rendering technique. By adapting the principles of Elliptical Weighted Average (EWA) splatting to a modern differentiable pipeline, 3DGS enables real-time, high-quality novel view synthesis. Building upon this, R2-Gaussian extended the 3DGS paradigm to tomographic reconstruction by rectifying integration bias, achieving state-of-the-art performance in computed tomography (CT). To enable differentiability, R2-Gaussian adopts a local affine approximation: each 3D Gaussian is locally mapped to a 2D Gaussian on the detector and composed via alpha blending to form projections. However, the affine approximation can degrade reconstruction quantitative accuracy and complicate the incorporation of nonlinear geometric corrections. To address these limitations, we propose a tomographic reconstruction framework based on 3D Gaussian ray tracing. Our approach provides two key advantages over splatting-based models: (i) it computes the line integral through 3D Gaussian primitives analytically, avoiding the local affine collapse and thus yielding a more physically consistent forward projection model; and (ii) the ray-tracing formulation gives explicit control over ray origins and directions, which facilitates the precise application of nonlinear geometric corrections, e.g., arc-correction used in positron emission tomography (PET). These properties extend the applicability of Gaussian-based reconstruction to a wider range of realistic tomography systems while improving projection accuracy.

  </details>


