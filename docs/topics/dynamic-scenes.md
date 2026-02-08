# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-08 07:05 UTC_

Total papers shown: **3**


---

- **MambaVF: State Space Model for Efficient Video Fusion**  
  Zixiang Zhao, Yukun Cui, Lilun Deng, Haowen Bai, Haotong Qin, Tao Feng, Konrad Schindler  
  _2026-02-05_ · https://arxiv.org/abs/2602.06017v1  
  <details><summary>Abstract</summary>

  Video fusion is a fundamental technique in various video processing tasks. However, existing video fusion methods heavily rely on optical flow estimation and feature warping, resulting in severe computational overhead and limited scalability. This paper presents MambaVF, an efficient video fusion framework based on state space models (SSMs) that performs temporal modeling without explicit motion estimation. First, by reformulating video fusion as a sequential state update process, MambaVF captures long-range temporal dependencies with linear complexity while significantly reducing computation and memory costs. Second, MambaVF proposes a lightweight SSM-based fusion module that replaces conventional flow-guided alignment via a spatio-temporal bidirectional scanning mechanism. This module enables efficient information aggregation across frames. Extensive experiments across multiple benchmarks demonstrate that our MambaVF achieves state-of-the-art performance in multi-exposure, multi-focus, infrared-visible, and medical video fusion tasks. We highlight that MambaVF enjoys high efficiency, reducing up to 92.25% of parameters and 88.79% of computational FLOPs and a 2.1x speedup compared to existing methods. Project page: https://mambavf.github.io

  </details>



- **Geometric Observability Index: An Operator-Theoretic Framework for Per-Feature Sensitivity, Weak Observability, and Dynamic Effects in SE(3) Pose Estimation**  
  Joe-Mei Feng, Sheng-Wei Yu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05582v1  
  <details><summary>Abstract</summary>

  We present a unified operator-theoretic framework for analyzing per-feature sensitivity in camera pose estimation on the Lie group SE(3). Classical sensitivity tools - conditioning analyses, Euclidean perturbation arguments, and Fisher information bounds - do not explain how individual image features influence the pose estimate, nor why dynamic or inconsistent observations can disproportionately distort modern SLAM and structure-from-motion systems. To address this gap, we extend influence function theory to matrix Lie groups and derive an intrinsic perturbation operator for left-trivialized M-estimators on SE(3). The resulting Geometric Observability Index (GOI) quantifies the contribution of a single measurement through the curvature operator and the Lie algebraic structure of the observable subspace. GOI admits a spectral decomposition along the principal directions of the observable curvature, revealing a direct correspondence between weak observability and amplified sensitivity. In the population regime, GOI coincides with the Fisher information geometry on SE(3), yielding a single-measurement analogue of the Cramer-Rao bound. The same spectral mechanism explains classical degeneracies such as pure rotation and vanishing parallax, as well as dynamic feature amplification along weak curvature directions. Overall, GOI provides a geometrically consistent description of measurement influence that unifies conditioning analysis, Fisher information geometry, influence function theory, and dynamic scene detectability through the spectral geometry of the curvature operator. Because these quantities arise directly within Gauss-Newton pipelines, the curvature spectrum and GOI also yield lightweight, training-free diagnostic signals for identifying dynamic features and detecting weak observability configurations without modifying existing SLAM architectures.

  </details>



- **VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency**  
  Zhuang Xiong, Chen Zhang, Qingshan Xu, Wenbing Tao  
  _2026-02-05_ · https://arxiv.org/abs/2602.05508v1  
  <details><summary>Abstract</summary>

  Despite recent progress in calibration-free monocular SLAM via 3D vision foundation models, scale drift remains severe on long sequences. Motion-agnostic partitioning breaks contextual coherence and causes zero-motion drift, while conventional geometric alignment is computationally expensive. To address these issues, we propose VGGT-Motion, a calibration-free SLAM system for efficient and robust global consistency over kilometer-scale trajectories. Specifically, we first propose a motion-aware submap construction mechanism that uses optical flow to guide adaptive partitioning, prune static redundancy, and encapsulate turns for stable local geometry. We then design an anchor-driven direct Sim(3) registration strategy. By exploiting context-balanced anchors, it achieves search-free, pixel-wise dense alignment and efficient loop closure without costly feature matching. Finally, a lightweight submap-level pose graph optimization enforces global consistency with linear complexity, enabling scalable long-range operation. Experiments show that VGGT-Motion markedly improves trajectory accuracy and efficiency, achieving state-of-the-art performance in zero-shot, long-range calibration-free monocular SLAM.

  </details>


