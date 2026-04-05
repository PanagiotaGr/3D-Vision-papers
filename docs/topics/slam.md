# SLAM & Localization

_Updated: 2026-04-05 07:21 UTC_

Total papers shown: **7**


---

- **Cross-Modal Visuo-Tactile Object Perception**  
  Anirvan Dutta, Simone Tasciotti, Claudia Cusseddu, Ang Li, Panayiota Poirazi, Julijana Gjorgjieva, Etienne Burdet, Patrick van der Smagt, Mohsen Kaboli  
  _2026-04-02_ · https://arxiv.org/abs/2604.02108v1  
  <details><summary>Abstract</summary>

  Estimating physical properties is critical for safe and efficient autonomous robotic manipulation, particularly during contact-rich interactions. In such settings, vision and tactile sensing provide complementary information about object geometry, pose, inertia, stiffness, and contact dynamics, such as stick-slip behavior. However, these properties are only indirectly observable and cannot always be modeled precisely (e.g., deformation in non-rigid objects coupled with nonlinear contact friction), making the estimation problem inherently complex and requiring sustained exploitation of visuo-tactile sensory information during action. Existing visuo-tactile perception frameworks have primarily emphasized forceful sensor fusion or static cross-modal alignment, with limited consideration of how uncertainty and beliefs about object properties evolve over time. Inspired by human multi-sensory perception and active inference, we propose the Cross-Modal Latent Filter (CMLF) to learn a structured, causal latent state-space of physical object properties. CMLF supports bidirectional transfer of cross-modal priors between vision and touch and integrates sensory evidence through a Bayesian inference process that evolves over time. Real-world robotic experiments demonstrate that CMLF improves the efficiency and robustness of latent physical properties estimation under uncertainty compared to baseline approaches. Beyond performance gains, the model exhibits perceptual coupling phenomena analogous to those observed in humans, including susceptibility to cross-modal illusions and similar trajectories in learning cross-sensory associations. Together, these results constitutes a significant step toward generalizable, robust and physically consistent cross-modal integration for robotic multi-sensory perception.

  </details>



- **HyVGGT-VO: Tightly Coupled Hybrid Dense Visual Odometry with Feed-Forward Models**  
  Junxiang Pan, Lipu Zhou, Baojie Chen  
  _2026-04-02_ · https://arxiv.org/abs/2604.02107v1  
  <details><summary>Abstract</summary>

  Dense visual odometry (VO), which provides pose estimation and dense 3D reconstruction, serves as the cornerstone for applications ranging from robotics to augmented reality. Recently, feed-forward models have demonstrated remarkable capabilities in dense mapping. However, when these models are used in dense visual SLAM systems, their heavy computational burden restricts them to yielding sparse pose outputs at keyframes while still failing to achieve real-time pose estimation. In contrast, traditional sparse methods provide high computational efficiency and high-frequency pose outputs, but lack the capability for dense reconstruction. To address these limitations, we propose HyVGGT-VO, a novel framework that combines the computational efficiency of sparse VO with the dense reconstruction capabilities of feed-forward models. To the best of our knowledge, this is the first work to tightly couple a traditional VO framework with VGGT, a state-of-the-art feed-forward model. Specifically, we design an adaptive hybrid tracking frontend that dynamically switches between traditional optical flow and the VGGT tracking head to ensure robustness. Furthermore, we introduce a hierarchical optimization framework that jointly refines VO poses and the scale of VGGT predictions to ensure global scale consistency. Our approach achieves an approximately 5x processing speedup compared to existing VGGT-based methods, while reducing the average trajectory error by 85% on the indoor EuRoC dataset and 12% on the outdoor KITTI benchmark. Our code will be publicly available upon acceptance. Project page: https://geneta2580.github.io/HyVGGT-VO.io.

  </details>



- **GS^2: Graph-based Spatial Distribution Optimization for Compact 3D Gaussian Splatting**  
  Xianben Yang, Tao Wang, Yuxuan Li, Yi Jin, Haibin Ling  
  _2026-04-02_ · https://arxiv.org/abs/2604.01884v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated breakthrough performance in novel view synthesis and real-time rendering. Nevertheless, its practicality is constrained by the high memory cost due to a huge number of Gaussian points. Many pruning-based 3DGS variants have been proposed for memory saving, but often compromise spatial consistency and may lead to rendering artifacts. To address this issue, we propose graph-based spatial distribution optimization for compact 3D Gaussian Splatting (GS\textasciicircum2), which enhances reconstruction quality by optimizing the spatial distribution of Gaussian points. Specifically, we introduce an evidence lower bound (ELBO)-based adaptive densification strategy that automatically controls the densification process. In addition, an opacity-aware progressive pruning strategy is proposed to further reduce memory consumption by dynamically removing low-opacity Gaussian points. Furthermore, we propose a graph-based feature encoding module to adjust the spatial distribution via feature-guided point shifting. Extensive experiments validate that GS\textasciicircum2 achieves a compact Gaussian representation while delivering superior rendering quality. Compared with 3DGS, it achieves higher PSNR with only about 12.5\% Gaussian points. Furthermore, it outperforms all compared baselines in both rendering quality and memory efficiency.

  </details>



- **PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency**  
  Leezy Han, Seunggyu Kim, Dongseok Shim, Hyeonbeom Lee  
  _2026-04-02_ · https://arxiv.org/abs/2604.01791v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation (MDE) has been widely adopted in the perception systems of autonomous vehicles and mobile robots. However, existing approaches often struggle to maintain temporal consistency in depth estimation across consecutive frames. This inconsistency not only causes jitter but can also lead to estimation failures when the depth range changes abruptly. To address these challenges, this paper proposes a consistency-aware monocular depth estimation framework that leverages wheel odometry from a mobile robot to achieve stable and coherent depth predictions over time. Specifically, we estimate camera pose and sparse depth from triangulation using optical flow between consecutive frames. The sparse depth estimates are used to update a recursive Bayesian estimate of the metric scale, which is then applied to rescale the relative depth predicted by a pre-trained depth estimation foundation model. The proposed method is evaluated on the KITTI, TartanAir, MS2, and our own dataset, demonstrating robust and accurate depth estimation performance.

  </details>



- **Unifying UAV Cross-View Geo-Localization via 3D Geometric Perception**  
  Haoyuan Li, Wen Yang, Fang Xu, Hong Tan, Haijian Zhang, Shengyang Li, Gui-Song Xia  
  _2026-04-02_ · https://arxiv.org/abs/2604.01747v1  
  <details><summary>Abstract</summary>

  Cross-view geo-localization for Unmanned Aerial Vehicles (UAVs) operating in GNSS-denied environments remains challenging due to the severe geometric discrepancy between oblique UAV imagery and orthogonal satellite maps. Most existing methods address this problem through a decoupled pipeline of place retrieval and pose estimation, implicitly treating perspective distortion as appearance noise rather than an explicit geometric transformation. In this work, we propose a geometry-aware UAV geo-localization framework that explicitly models the 3D scene geometry to unify coarse place recognition and fine-grained pose estimation within a single inference pipeline. Our approach reconstructs a local 3D scene from multi-view UAV image sequences using a Visual Geometry Grounded Transformer (VGGT), and renders a virtual Bird's-Eye View (BEV) representation that orthorectifies the UAV perspective to align with satellite imagery. This BEV serves as a geometric intermediary that enables robust cross-view retrieval and provides spatial priors for accurate 3 Degrees of Freedom (3-DoF) pose regression. To efficiently handle multiple location hypotheses, we introduce a Satellite-wise Attention Block that isolates the interaction between each satellite candidate and the reconstructed UAV scene, preventing inter-candidate interference while maintaining linear computational complexity. In addition, we release a recalibrated version of the University-1652 dataset with precise coordinate annotations and spatial overlap analysis, enabling rigorous evaluation of end-to-end localization accuracy. Extensive experiments on the refined University-1652 benchmark and SUES-200 demonstrate that our method significantly outperforms state-of-the-art baselines, achieving robust meter-level localization accuracy and improved generalization in complex urban environments.

  </details>



- **Setup-Independent Full Projector Compensation**  
  Haibo Li, Qingyue Deng, Jijiang Li, Haibin Ling, Bingyao Huang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01736v1  
  <details><summary>Abstract</summary>

  Projector compensation seeks to correct geometric and photometric distortions that occur when images are projected onto nonplanar or textured surfaces. However, most existing methods are highly setup-dependent, requiring fine-tuning or retraining whenever the surface, lighting, or projector-camera pose changes. Progress has been limited by two key challenges: (1) the absence of large, diverse training datasets and (2) existing geometric correction models are typically constrained by specific spatial setups; without further retraining or fine-tuning, they often fail to generalize directly to novel geometric configurations. We introduce SIComp, the first Setup-Independent framework for full projector Compensation, capable of generalizing to unseen setups without fine-tuning or retraining. To enable this, we construct a large-scale real-world dataset spanning 277 distinct projector-camera setups. SIComp adopts a co-adaptive design that decouples geometry and photometry: A carefully tailored optical flow module performs online geometric correction, while a novel photometric network handles photometric compensation. To further enhance robustness under varying illumination, we integrate intensity-varying surface priors into the network design. Extensive experiments demonstrate that SIComp consistently produces high-quality compensation across diverse unseen setups, substantially outperforming existing methods in terms of generalization ability and establishing the first generalizable solution to projector compensation. The code and dataset are available on our project page: https://hai-bo-li.github.io/SIComp/

  </details>



- **Hi-LOAM: Hierarchical Implicit Neural Fields for LiDAR Odometry and Mapping**  
  Zhiliu Yang, Jianyuan Zhang, Lianhui Zhao, Jinyu Dai, Zhu Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01720v1  
  <details><summary>Abstract</summary>

  LiDAR Odometry and Mapping (LOAM) is a pivotal technique for embodied-AI applications such as autonomous driving and robot navigation. Most existing LOAM frameworks are either contingent on the supervision signal, or lack of the reconstruction fidelity, which are deficient in depicting details of large-scale complex scenes. To overcome these limitations, we propose a multi-scale implicit neural localization and mapping framework using LiDAR sensor, called Hi-LOAM. Hi-LOAM receives LiDAR point cloud as the input data modality, learns and stores hierarchical latent features in multiple levels of hash tables based on an octree structure, then these multi-scale latent features are decoded into signed distance value through shallow Multilayer Perceptrons (MLPs) in the mapping procedure. For pose estimation procedure, we rely on a correspondence-free, scan-to-implicit matching paradigm to estimate optimal pose and register current scan into the submap. The entire training process is conducted in a self-supervised manner, which waives the model pre-training and manifests its generalizability when applied to diverse environments. Extensive experiments on multiple real-world and synthetic datasets demonstrate the superior performance, in terms of the effectiveness and generalization capabilities, of our Hi-LOAM compared to existing state-of-the-art methods.

  </details>


