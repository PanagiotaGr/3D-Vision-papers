# SLAM & Localization

_Updated: 2026-04-08 07:51 UTC_

Total papers shown: **12**


---

- **Graph-PiT: Enhancing Structural Coherence in Part-Based Image Synthesis via Graph Priors**  
  Junbin Zhang, Meng Cao, Feng Tan, Yikai Lin, Yuexian Zou  
  _2026-04-07_ · https://arxiv.org/abs/2604.06074v1  
  <details><summary>Abstract</summary>

  Achieving fine-grained and structurally sound controllability is a cornerstone of advanced visual generation. Existing part-based frameworks treat user-provided parts as an unordered set and therefore ignore their intrinsic spatial and semantic relationships, which often results in compositions that lack structural integrity. To bridge this gap, we propose Graph-PiT, a framework that explicitly models the structural dependencies of visual components using a graph prior. Specifically, we represent visual parts as nodes and their spatial-semantic relationships as edges. At the heart of our method is a Hierarchical Graph Neural Network (HGNN) module that performs bidirectional message passing between coarse-grained part-level super-nodes and fine-grained IP+ token sub-nodes, refining part embeddings before they enter the generative pipeline. We also introduce a graph Laplacian smoothness loss and an edge-reconstruction loss so that adjacent parts acquire compatible, relation-aware embeddings. Quantitative experiments on controlled synthetic domains (character, product, indoor layout, and jigsaw), together with qualitative transfer to real web images, show that Graph-PiT improves structural coherence over vanilla PiT while remaining compatible with the original IP-Prior pipeline. Ablation experiments confirm that explicit relational reasoning is crucial for enforcing user-specified adjacency constraints. Our approach not only enhances the plausibility of generated concepts but also offers a scalable and interpretable mechanism for complex, multi-part image synthesis. The code is available at https://github.com/wolf-bailang/Graph-PiT.

  </details>



- **GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance**  
  Weiqi Zhang, Junsheng Zhou, Haotian Geng, Kanle Shi, Shenkun Xu, Yi Fang, Yu-Shen Liu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05721v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has demonstrated superior performance in rendering efficiency and quality, yet the generation of 3D Gaussians still remains a challenge without proper geometric priors. Existing methods have explored predicting point maps as geometric references for inferring Gaussian primitives, while the unreliable estimated geometries may lead to poor generations. In this work, we introduce GaussianGrow, a novel approach that generates 3D Gaussians by learning to grow them from easily accessible 3D point clouds, naturally enforcing geometric accuracy in Gaussian generation. Specifically, we design a text-guided Gaussian growing scheme that leverages a multi-view diffusion model to synthesize consistent appearances from input point clouds for supervision. To mitigate artifacts caused by fusing neighboring views, we constrain novel views generated at non-preset camera poses identified in overlapping regions across different views. For completing the hard-to-observe regions, we propose to iteratively detect the camera pose by observing the largest un-grown regions in point clouds and inpainting them by inpainting the rendered view with a pretrained 2D diffusion model. The process continues until complete Gaussians are generated. We extensively evaluate GaussianGrow on text-guided Gaussian generation from synthetic and even real-scanned point clouds. Project Page: https://weiqi-zhang.github.io/GaussianGrow

  </details>



- **LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios**  
  Xiang Zhang, Tengfei Wang, Fang Xu, Xin Wang, Zongqian Zhan  
  _2026-04-07_ · https://arxiv.org/abs/2604.05402v1  
  <details><summary>Abstract</summary>

  Visual localization in large-scale UAV scenarios is a critical capability for autonomous systems, yet it remains challenging due to geometric complexity and environmental variations. While 3D Gaussian Splatting (3DGS) has emerged as a promising scene representation, existing 3DGS-based visual localization methods struggle with robust pose initialization and sensitivity to rendering artifacts in large-scale settings. To address these limitations, we propose LSGS-Loc, a novel visual localization pipeline tailored for large-scale 3DGS scenes. Specifically, we introduce a scale-aware pose initialization strategy that combines scene-agnostic relative pose estimation with explicit 3DGS scale constraints, enabling geometrically grounded localization without scene-specific training. Furthermore, in the pose refinement, to mitigate the impact of reconstruction artifacts such as blur and floaters, we develop a Laplacian-based reliability masking mechanism that guides photometric refinement toward high-quality regions. Extensive experiments on large-scale UAV benchmarks demonstrate that our method achieves state-of-the-art accuracy and robustness for unordered image queries, significantly outperforming existing 3DGS-based approaches. Code is available at: https://github.com/xzhang-z/LSGS-Loc

  </details>



- **AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation**  
  Yijie Deng, Shuaihang Yuan, Yi Fang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05351v1  
  <details><summary>Abstract</summary>

  Image Goal Navigation (ImageNav) is evaluated by a coarse success criterion, the agent must stop within 1m of the target, which is sufficient for finding objects but falls short for downstream tasks such as grasping that require precise positioning. We introduce AnyImageNav, a training-free system that pushes ImageNav toward this more demanding setting. Our key insight is that the goal image can be treated as a geometric query: any photo of an object, a hallway, or a room corner can be registered to the agent's observations via dense pixel-level correspondences, enabling recovery of the exact 6-DoF camera pose. Our method realizes this through a semantic-to-geometric cascade: a semantic relevance signal guides exploration and acts as a proximity gate, invoking a 3D multi-view foundation model only when the current view is highly relevant to the goal image; the model then self-certifies its registration in a loop for an accurate recovered pose. Our method sets state-of-the-art navigation success rates on Gibson (93.1%) and HM3D (82.6%), and achieves pose recovery that prior methods do not provide: a position error of 0.27m and heading error of 3.41 degrees on Gibson, and 0.21m / 1.23 degrees on HM3D, a 5-10x improvement over adapted baselines.

  </details>



- **Coverage Optimization for Camera View Selection**  
  Timothy Chen, Adam Dai, Maximilian Adang, Grace Gao, Mac Schwager  
  _2026-04-06_ · https://arxiv.org/abs/2604.05259v1  
  <details><summary>Abstract</summary>

  What makes a good viewpoint? The quality of the data used to learn 3D reconstructions is crucial for enabling efficient and accurate scene modeling. We study the active view selection problem and develop a principled analysis that yields a simple and interpretable criterion for selecting informative camera poses. Our key insight is that informative views can be obtained by minimizing a tractable approximation of the Fisher Information Gain, which reduces to favoring viewpoints that cover geometry that has been insufficiently observed by past cameras. This leads to a lightweight coverage-based view selection metric that avoids expensive transmittance estimation and is robust to noise and training dynamics. We call this metric COVER (Camera Optimization for View Exploration and Reconstruction). We integrate our method into the Nerfstudio framework and evaluate it on real datasets within fixed and embodied data acquisition scenarios. Across multiple datasets and radiance-field baselines, our method consistently improves reconstruction quality compared to state-of-the-art active view selection methods. Additional visualizations and our Nerfstudio package can be found at https://chengine.github.io/nbv_gym/.

  </details>



- **Synchronous Observer Design for Landmark-Inertial SLAM with Magnetometer and Intermittent GNSS Measurements**  
  Arkadeep Saha, Pieter van Goor, Ravi Banavar  
  _2026-04-06_ · https://arxiv.org/abs/2604.05156v1  
  <details><summary>Abstract</summary>

  In Landmark-Inertial Simultaneous Localisation and Mapping (LI-SLAM), the positions of landmarks in the environment and the robot's pose relative to these landmarks are estimated using landmark position measurements, and measurements from the Inertial Measurement Unit (IMU). However, the robot and landmark positions in the inertial frame, and the yaw of the robot, are not observable in LI-SLAM. This paper proposes a nonlinear observer for LI-SLAM that overcomes the observability constraints with the addition of intermittent GNSS position and magnetometer measurements. The full-state error dynamics of the proposed observer is shown to be both almost-globally asymptotically stable and locally exponentially stable, and this is validated using simulations.

  </details>



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


