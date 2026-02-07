# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-07 06:59 UTC_

Total papers shown: **5**


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



- **Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis**  
  Seong-Eun Hong, JaeYoung Seon, JuYeong Hwang, JongHwan Shin, HyeongYeop Kang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04292v1  
  <details><summary>Abstract</summary>

  Text-to-motion generation has advanced with diffusion models, yet existing systems often collapse complex multi-action prompts into a single embedding, leading to omissions, reordering, or unnatural transitions. In this work, we shift perspective by introducing a principled definition of an event as the smallest semantically self-contained action or state change in a text prompt that can be temporally aligned with a motion segment. Building on this definition, we propose Event-T2M, a diffusion-based framework that decomposes prompts into events, encodes each with a motion-aware retrieval model, and integrates them through event-based cross-attention in Conformer blocks. Existing benchmarks mix simple and multi-event prompts, making it unclear whether models that succeed on single actions generalize to multi-action cases. To address this, we construct HumanML3D-E, the first benchmark stratified by event count. Experiments on HumanML3D, KIT-ML, and HumanML3D-E show that Event-T2M matches state-of-the-art baselines on standard tests while outperforming them as event complexity increases. Human studies validate the plausibility of our event definition, the reliability of HumanML3D-E, and the superiority of Event-T2M in generating multi-event motions that preserve order and naturalness close to ground-truth. These results establish event-level conditioning as a generalizable principle for advancing text-to-motion generation beyond single-action prompts.

  </details>



- **SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization**  
  Lifan Wu, Ruijie Zhu, Yubo Ai, Tianzhu Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04271v1  
  <details><summary>Abstract</summary>

  4D generation has made remarkable progress in synthesizing dynamic 3D objects from input text, images, or videos. However, existing methods often represent motion as an implicit deformation field, which limits direct control and editability. To address this issue, we propose SkeletonGaussian, a novel framework for generating editable dynamic 3D Gaussians from monocular video input. Our approach introduces a hierarchical articulated representation that decomposes motion into sparse rigid motion explicitly driven by a skeleton and fine-grained non-rigid motion. Concretely, we extract a robust skeleton and drive rigid motion via linear blend skinning, followed by a hexplane-based refinement for non-rigid deformations, enhancing interpretability and editability. Experimental results demonstrate that SkeletonGaussian surpasses existing methods in generation quality while enabling intuitive motion editing, establishing a new paradigm for editable 4D generation. Project page: https://wusar.github.io/projects/skeletongaussian/

  </details>


