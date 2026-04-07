# SLAM & Localization

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **10**


---

- **Multi-Modal Sensor Fusion using Hybrid Attention for Autonomous Driving**  
  Mayank Mayank, Bharanidhar Duraisamy, Florian Geiß, Abhinav Valada  
  _2026-04-06_ · https://arxiv.org/abs/2604.04797v1  
  <details><summary>Abstract</summary>

  Accurate 3D object detection for autonomous driving requires complementary sensors. Cameras provide dense semantics but unreliable depth, while millimeter-wave radar offers precise range and velocity measurements with sparse geometry. We propose MMF-BEV, a radar-camera BEV fusion framework that leverages deformable attention for cross-modal feature alignment on the View-of-Delft (VoD) 4D radar dataset [1]. MMF-BEV builds a BEVDepth [2] camera branch and a RadarBEVNet [3] radar branch, each enhanced with Deformable Self-Attention, and fuses them via a Deformable Cross-Attention module. We evaluate three configurations: camera-only, radar-only, and hybrid fusion. A sensor contribution analysis quantifies per-distance modality weighting, providing interpretable evidence of sensor complementarity. A two-stage training strategy - pre-training the camera branch with depth supervision, then jointly training radar and fusion modules stabilizes learning. Experiments on VoD show that MMF-BEV consistently outperforms unimodal baselines and achieves competitive results against prior fusion methods across all object classes in both the full annotated area and near-range Region of Interest.

  </details>



- **ZeD-MAP: Bundle Adjustment Guided Zero-Shot Depth Maps for Real-Time Aerial Imaging**  
  Selim Ahmet Iz, Francesco Nex, Norman Kerle, Henry Meissner, Ralf Berger  
  _2026-04-06_ · https://arxiv.org/abs/2604.04667v1  
  <details><summary>Abstract</summary>

  Real-time depth reconstruction from ultra-high-resolution UAV imagery is essential for time-critical geospatial tasks such as disaster response, yet remains challenging due to wide-baseline parallax, large image sizes, low-texture or specular surfaces, occlusions, and strict computational constraints. Recent zero-shot diffusion models offer fast per-image dense predictions without task-specific retraining, and require fewer labelled datasets than transformer-based predictors while avoiding the rigid capture geometry requirement of classical multi-view stereo. However, their probabilistic inference prevents reliable metric accuracy and temporal consistency across sequential frames and overlapping tiles. We present ZeD-MAP, a cluster-level framework that converts a test-time diffusion depth model into a metrically consistent, SLAM-like mapping pipeline by integrating incremental cluster-based bundle adjustment (BA). Streamed UAV frames are grouped into overlapping clusters; periodic BA produces metrically consistent poses and sparse 3D tie-points, which are reprojected into selected frames and used as metric guidance for diffusion-based depth estimation. Validation on ground-marker flights captured at approximately 50 m altitude (GSD is approximately 0.85 cm/px, corresponding to 2,650 square meters ground coverage per frame) with the DLR Modular Aerial Camera System (MACS) shows that our method achieves sub-meter accuracy, with approximately 0.87 m error in the horizontal (XY) plane and 0.12 m in the vertical (Z) direction, while maintaining per-image runtimes between 1.47 and 4.91 seconds. Results are subject to minor noise from manual point-cloud annotation. These findings show that BA-based metric guidance provides consistency comparable to classical photogrammetric methods while significantly accelerating processing, enabling real-time 3D map generation.

  </details>



- **WaterSplat-SLAM: Photorealistic Monocular SLAM in Underwater Environment**  
  Kangxu Wang, Shaofeng Zou, Chenxing Jiang, Yixiang Dai, Siang Chen, Shaojie Shen, Guijin Wang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04642v1  
  <details><summary>Abstract</summary>

  Underwater monocular SLAM is a challenging problem with applications from autonomous underwater vehicles to marine archaeology. However, existing underwater SLAM methods struggle to produce maps with high-fidelity rendering. In this paper, we propose WaterSplat-SLAM, a novel monocular underwater SLAM system that achieves robust pose estimation and photorealistic dense mapping. Specifically, we couple semantic medium filtering into two-view 3D reconstruction prior to enable underwater-adapted camera tracking and depth estimation. Furthermore, we present a semantic-guided rendering and adaptive map management strategy with an online medium-aware Gaussian map, modeling underwater environment in a photorealistic and compact manner. Experiments on multiple underwater datasets demonstrate that WaterSplat-SLAM achieves robust camera tracking and high-fidelity rendering in underwater environments.

  </details>



- **Relational Epipolar Graphs for Robust Relative Camera Pose Estimation**  
  Prateeth Rao, Sachit Rao  
  _2026-04-06_ · https://arxiv.org/abs/2604.04554v1  
  <details><summary>Abstract</summary>

  A key component of Visual Simultaneous Localization and Mapping (VSLAM) is estimating relative camera poses using matched keypoints. Accurate estimation is challenged by noisy correspondences. Classical methods rely on stochastic hypothesis sampling and iterative estimation, while learning-based methods often lack explicit geometric structure. In this work, we reformulate relative pose estimation as a relational inference problem over epipolar correspondence graphs, where matched keypoints are nodes and nearby ones are connected by edges. Graph operations such as pruning, message passing, and pooling estimate a quaternion rotation, translation vector, and the Essential Matrix (EM). Minimizing a loss comprising (i) $\mathcal{L}_2$ differences with ground truth (GT), (ii) Frobenius norm between estimated and GT EMs, (iii) singular value differences, (iv) heading angle differences, and (v) scale differences, yields the relative pose between image pairs. The dense detector-free method LoFTR is used for matching. Experiments on indoor and outdoor benchmarks show improved robustness to dense noise and large baseline variation compared to classical and learning-guided approaches, highlighting the effectiveness of global relational consensus.

  </details>



- **MPTF-Net: Multi-view Pyramid Transformer Fusion Network for LiDAR-based Place Recognition**  
  Shuyuan Li, Zihang Wang, Xieyuanli Chen, Wenkai Zhu, Xiaoteng Fang, Peizhou Ni, Junhao Yang, Dong Kong  
  _2026-04-06_ · https://arxiv.org/abs/2604.04513v1  
  <details><summary>Abstract</summary>

  LiDAR-based place recognition (LPR) is essential for global localization and loop-closure detection in large-scale SLAM systems. Existing methods typically construct global descriptors from Range Images or BEV representations for matching. BEV is widely adopted due to its explicit 2D spatial layout encoding and efficient retrieval. However, conventional BEV representations rely on simple statistical aggregation, which fails to capture fine-grained geometric structures, leading to performance degradation in complex or repetitive environments. To address this, we propose MPTF-Net, a novel multi-view multi-scale pyramid Transformer fusion network. Our core contribution is a multi-channel NDT-based BEV encoding that explicitly models local geometric complexity and intensity distributions via Normal Distribution Transform, providing a noise-resilient structural prior. To effectively integrate these features, we develop a customized pyramid Transformer module that captures cross-view interactive correlations between Range Image Views (RIV) and NDT-BEV at multiple spatial scales. Extensive experiments on the nuScenes, KITTI and NCLT datasets demonstrate that MPTF-Net achieves state-of-the-art performance, specifically attaining a Recall@1 of 96.31\% on the nuScenes Boston split while maintaining an inference latency of only 10.02 ms, making it highly suitable for real-time autonomous unmanned systems.

  </details>



- **DINO-VO: Learning Where to Focus for Enhanced State Estimation**  
  Qi Chen, Guanghao Li, Sijia Hu, Xin Gao, Junpeng Ma, Xiangyang Xue, Jian Pu  
  _2026-04-05_ · https://arxiv.org/abs/2604.04055v1  
  <details><summary>Abstract</summary>

  We present DINO Patch Visual Odometry (DINO-VO), an end-to-end monocular visual odometry system with strong scene generalization. Current Visual Odometry (VO) systems often rely on heuristic feature extraction strategies, which can degrade accuracy and robustness, particularly in large-scale outdoor environments. DINO-VO addresses these limitations by incorporating a differentiable adaptive patch selector into the end-to-end pipeline, improving the quality of extracted patches and enhancing generalization across diverse datasets. Additionally, our system integrates a multi-task feature extraction module with a differentiable bundle adjustment (BA) module that leverages inverse depth priors, enabling the system to learn and utilize appearance and geometric information effectively. This integration bridges the gap between feature learning and state estimation. Extensive experiments on the TartanAir, KITTI, Euroc, and TUM datasets demonstrate that DINO-VO exhibits strong generalization across synthetic, indoor, and outdoor environments, achieving state-of-the-art tracking accuracy.

  </details>



- **Learning 3D Reconstruction with Priors in Test Time**  
  Lei Zhou, Haoyu Wu, Akshat Dave, Dimitris Samaras  
  _2026-04-04_ · https://arxiv.org/abs/2604.03878v1  
  <details><summary>Abstract</summary>

  We introduce a test-time framework for multiview Transformers (MVTs) that incorporates priors (e.g., camera poses, intrinsics, and depth) to improve 3D tasks without retraining or modifying pre-trained image-only networks. Rather than feeding priors into the architecture, we cast them as constraints on the predictions and optimize the network at inference time. The optimization loss consists of a self-supervised objective and prior penalty terms. The self-supervised objective captures the compatibility among multi-view predictions and is implemented using photometric or geometric loss between renderings from other views and each view itself. Any available priors are converted into penalty terms on the corresponding output modalities. Across a series of 3D vision benchmarks, including point map estimation and camera pose estimation, our method consistently improves performance over base MVTs by a large margin. On the ETH3D, 7-Scenes, and NRGBD datasets, our method reduces the point-map distance error by more than half compared with the base image-only models. Our method also outperforms retrained prior-aware feed-forward methods, demonstrating the effectiveness of our test-time constrained optimization (TCO) framework for incorporating priors into 3D vision tasks.

  </details>



- **InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset**  
  Felix Stillger, Lukas Hahn, Frederik Hasecke, Tobias Meisen  
  _2026-04-04_ · https://arxiv.org/abs/2604.03814v1  
  <details><summary>Abstract</summary>

  Camera extrinsic calibration is a fundamental task in computer vision. However, precise relative pose estimation in constrained, highly distorted environments, such as in-cabin automotive monitoring (ICAM), remains challenging. We present InCaRPose, a Transformer-based architecture designed for robust relative pose prediction between image pairs, which can be used for camera extrinsic calibration. By leveraging frozen backbone features such as DINOv3 and a Transformer-based decoder, our model effectively captures the geometric relationship between a reference and a target view. Unlike traditional methods, our approach achieves absolute metric-scale translation within the physically plausible adjustment range of in-cabin camera mounts in a single inference step, which is critical for ICAM, where accurate real-world distances are required for safety-relevant perception. We specifically address the challenges of highly distorted fisheye cameras in automotive interiors by training exclusively on synthetic data. Our model is capable of generalization to real-world cabin environments without relying on the exact same camera intrinsics and additionally achieves competitive performance on the public 7-Scenes dataset. Despite having limited training data, InCaRPose maintains high precision in both rotation and translation, even with a ViT-Small backbone. This enables real-time performance for time-critical inference, such as driver monitoring in supervised autonomous driving. We release our real-world In-Cabin-Pose test dataset consisting of highly distorted vehicle-interior images and our code at https://github.com/felixstillger/InCaRPose.

  </details>



- **CT-VoxelMap: Efficient Continuous-Time LiDAR-Inertial Odometry with Probabilistic Adaptive Voxel Mapping**  
  Lei Zhao, Xingyi Li, Tianchen Deng, Chuan Cao, Han Zhang, Weidong Chen  
  _2026-04-04_ · https://arxiv.org/abs/2604.03747v1  
  <details><summary>Abstract</summary>

  Maintaining stable and accurate localization during fast motion or on rough terrain remains highly challenging for mobile robots with onboard resources. Currently, multi-sensor fusion methods based on continuous-time representation offer a potential and effective solution to this challenge. Among these, spline-based methods provide an efficient and intuitive approach for continuous-time representation. Previous continuous-time odometry works based on B-splines either treat control points as variables to be estimated or perform estimation in quaternion space, which introduces complexity in deriving analytical Jacobians and often overlooks the fitting error between the spline and the true trajectory over time. To address these issues, we first propose representing the increments of control points on matrix Lie groups as variables to be estimated. Leveraging the feature of the cumulative form of B-splines, we derive a more compact formulation that yields simpler analytical Jacobians without requiring additional boundary condition considerations. Second, we utilize forward propagation information from IMU measurements to estimate fitting errors online and further introduce a hybrid feature-based voxel map management strategy, enhancing system accuracy and robustness. Finally, we propose a re-estimation policy that significantly improves system computational efficiency and robustness. The proposed method is evaluated on multiple challenging public datasets, demonstrating superior performance on most sequences. Detailed ablation studies are conducted to analyze the impact of each module on the overall pose estimation system.

  </details>



- **SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation**  
  Guiyu Zhang, Yabo Chen, Xunzhi Xiang, Junchao Huang, Zhongyu Wang, Li Jiang  
  _2026-04-04_ · https://arxiv.org/abs/2604.03723v1  
  <details><summary>Abstract</summary>

  Controlling both camera motion and object dynamics is essential for coherent and expressive video generation, yet current methods typically handle only one motion type or rely on ambiguous 2D cues that entangle camera-induced parallax with true object movement. We present SymphoMotion, a unified motion-control framework that jointly governs camera trajectories and object dynamics within a single model. SymphoMotion features a Camera Trajectory Control mechanism that integrates explicit camera paths with geometry-aware cues to ensure stable, structurally consistent viewpoint transitions, and an Object Dynamics Control mechanism that combines 2D visual guidance with 3D trajectory embeddings to enable depth-aware, spatially coherent object manipulation. To support large-scale training and evaluation, we further construct RealCOD-25K, a comprehensive real-world dataset containing paired camera poses and object-level 3D trajectories across diverse indoor and outdoor scenes, addressing a key data gap in unified motion control. Extensive experiments and user studies show that SymphoMotion significantly outperforms existing methods in visual fidelity, camera controllability, and object-motion accuracy, establishing a new benchmark for unified motion control in video generation.Codes and data are publicly available at https://grenoble-zhang.github.io/SymphoMotion/.

  </details>


