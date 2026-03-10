# SLAM & Localization

_Updated: 2026-03-10 07:06 UTC_

Total papers shown: **10**


---

- **Improving Continual Learning for Gaussian Splatting based Environments Reconstruction on Commercial Off-the-Shelf Edge Devices**  
  Ivan Zaino, Matteo Risso, Daniele Jahier Pagliari, Miguel de Prado, Toon Van de Maele, Alessio Burrello  
  _2026-03-09_ · https://arxiv.org/abs/2603.08499v1  
  <details><summary>Abstract</summary>

  Novel view synthesis (NVS) is increasingly relevant for edge robotics, where compact and incrementally updatable 3D scene models are needed for SLAM, navigation, and inspection under tight memory and latency budgets. Variational Bayesian Gaussian Splatting (VBGS) enables replay-free continual updates for the 3DGS algorithm by maintaining a probabilistic scene model, but its high-precision computations and large intermediate tensors make on-device training impractical. We present a precision-adaptive optimization framework that enables VBGS training on resource-constrained hardware without altering its variational formulation. We (i) profile VBGS to identify memory/latency hotspots, (ii) fuse memory-dominant kernels to reduce materialized intermediate tensors, and (iii) automatically assign operation-level precisions via a mixed-precision search with bounded relative error. Across the Blender, Habitat, and Replica datasets, our optimised pipeline reduces peak memory from 9.44 GB to 1.11 GB and training time from ~234 min to ~61 min on an A5000 GPU, while preserving (and in some cases improving) reconstruction quality of the state-of-the-art VBGS baseline. We also enable for the first time NVS training on a commercial embedded platform, the Jetson Orin Nano, reducing per-frame latency by 19x compared to 3DGS.

  </details>



- **Adaptive Entropy-Driven Sensor Selection in a Camera-LiDAR Particle Filter for Single-Vessel Tracking**  
  Andrei Starodubov, Yaqub Aris Prabowo, Andreas Hadjipieris, Ioannis Kyriakides, Roberto Galeazzi  
  _2026-03-09_ · https://arxiv.org/abs/2603.08457v1  
  <details><summary>Abstract</summary>

  Robust single-vessel tracking from fixed coastal platforms is hindered by modality-specific degradations: cameras suffer from illumination and visual clutter, while LiDAR performance drops with range and intermittent returns. We present a heterogeneous multi-sensor fusion particle-filter tracker that incorporates an information-gain (entropy-reduction) adaptive sensing policy to select the most informative configuration at each fusion time bin. The approach is validated in a real maritime deployment at the CMMI Smart Marina Testbed (Ayia Napa Marina, Cyprus), using a shore-mounted 3D LiDAR and an elevated fixed camera to track a rigid inflatable boat with onboard GNSS ground truth. We compare LiDAR-only, camera-only, all-sensors, and adaptive configurations. Results show LiDAR dominates near-field accuracy, the camera sustains longer-range coverage when LiDAR becomes unavailable, and the adaptive policy achieves a favorable accuracy-continuity trade-off by switching modalities based on information gain. By avoiding continuous multi-stream processing, the adaptive configuration provides a practical baseline for resilient and resource-aware maritime surveillance.

  </details>



- **FoMo: A Multi-Season Dataset for Robot Navigation in Forêt Montmorency**  
  Matěj Boxan, Gabriel Jeanson, Alexander Krawciw, Effie Daum, Xinyuan Qiao, Sven Lilge, Timothy D. Barfoot, François Pomerleau  
  _2026-03-09_ · https://arxiv.org/abs/2603.08433v1  
  <details><summary>Abstract</summary>

  The Forêt Montmorency (FoMo) dataset is a comprehensive multi-season data collection, recorded over the span of one year in a boreal forest. Featuring a unique combination of on- and off-pavement environments with significant environmental changes, the dataset challenges established odometry and SLAM pipelines. Some highlights of the data include the accumulation of snow exceeding 1 m, significant vegetation growth in front of sensors, and operations at the traction limits of the platform. In total, the FoMo dataset includes over 64 km of six diverse trajectories, repeated during 12 deployments throughout the year. The dataset features data from one rotating and one hybrid solid-state lidar, a Frequency Modulated Continuous Wave (FMCW) radar, full-HD images from a stereo camera and a wide lens monocular camera, as well as data from two IMUs. Ground Truth is calculated by post-processing three GNSS receivers mounted on the Uncrewed Ground Vehicle (UGV) and a static GNSS base station. Additional metadata, such as one measurement per minute from an on-site weather station, camera calibration intrinsics, and vehicle power consumption, is available for all sequences. To highlight the relevance of the dataset, we performed a preliminary evaluation of the robustness of a lidar-inertial, radar-gyro, and a visual-inertial localization and mapping techniques to seasonal changes. We show that seasonal changes have serious effects on the re-localization capabilities of the state-of-the-art methods. The dataset and development kit are available at https://fomo.norlab.ulaval.ca.

  </details>



- **Perception-Aware Communication-Free Multi-UAV Coordination in the Wild**  
  Manuel Boldrer, Michal Kamler, Afzal Ahmad, Martin Saska  
  _2026-03-09_ · https://arxiv.org/abs/2603.08379v1  
  <details><summary>Abstract</summary>

  We present a communication-free method for safe multi-robot coordination in complex environments such as forests with dense canopy cover, where GNSS is unavailable. Our approach relies on an onboard anisotropic 3D LiDAR sensor used for SLAM as well as for detecting obstacles and neighboring robots. We develop a novel perception-aware 3D navigation framework that enables robots to safely and effectively progress toward a goal region despite limited sensor field-of-view. The approach is evaluated through extensive simulations across diverse scenarios and validated in real-world field experiments, demonstrating its scalability, robustness, and reliability.

  </details>



- **Edged USLAM: Edge-Aware Event-Based SLAM with Learning-Based Depth Priors**  
  Şebnem Sarıözkan, Hürkan Şahin, Olaya Álvarez-Tuñón, Erdal Kayacan  
  _2026-03-09_ · https://arxiv.org/abs/2603.08150v1  
  <details><summary>Abstract</summary>

  Conventional visual simultaneous localization and mapping (SLAM) algorithms often fail under rapid motion, low illumination, or abrupt lighting transitions due to motion blur and limited dynamic range. Event cameras mitigate these issues with high temporal resolution and high dynamic range (HDR), but their sparse, asynchronous outputs complicate feature extraction and integration with other sensors; e.g. inertial measurement units (IMUs) and standard cameras. We present Edged USLAM, a hybrid visual-inertial system that extends Ultimate SLAM (USLAM) with an edge-aware front-end and a lightweight depth module. The frontend enhances event frames for robust feature tracking and nonlinear motion compensation, while the depth module provides coarse, region-of-interest (ROI)-based scene depth to improve motion compensation and scale consistency. Evaluations across public benchmarks and real-world unmanned air vehicle (UAV) flights demonstrate that performance varies significantly by scenario. For instance, event-only methods like point-line event-based visual-inertial odometry (PL-EVIO) or learning-based pipelines such as deep event-based visual odometry (DEVO) excel in highly aggressive or extreme HDR conditions. In contrast, Edged USLAM provides superior stability and minimal drift in slow or structured trajectories, ensuring consistently accurate localization on real flights under challenging illumination. These findings highlight the complementary strengths of event-only, learning-based, and hybrid approaches, while positioning Edged USLAM as a robust solution for diverse aerial navigation tasks.

  </details>



- **Speed3R: Sparse Feed-forward 3D Reconstruction Models**  
  Weining Ren, Xiao Tan, Kai Han  
  _2026-03-09_ · https://arxiv.org/abs/2603.08055v1  
  <details><summary>Abstract</summary>

  While recent feed-forward 3D reconstruction models accelerate 3D reconstruction by jointly inferring dense geometry and camera poses in a single pass, their reliance on dense attention imposes a quadratic complexity, creating a prohibitive computational bottleneck that severely limits inference speed. To resolve this, we introduce Speed3R, an end-to-end trainable model inspired by the core principle of Structure-from-Motion: that a sparse set of keypoints is sufficient for robust pose estimation. Speed3R features a dual-branch attention mechanism where a compression branch creates a coarse contextual prior to guide a selection branch, which performs fine-grained attention only on the most informative image tokens. This strategy mimics the efficiency of traditional keypoint matching, achieving a remarkable 12.4x inference speedup on 1000-view sequences, while introducing a minimal, controlled trade-off in geometric accuracy. Validated on standard benchmarks with both VGGT and $π^3$ backbones, our method delivers high-quality reconstructions at a fraction of computational cost, paving the way for efficient large-scale scene modeling.

  </details>



- **$L^3$:Scene-agnostic Visual Localization in the Wild**  
  Yu Zhang, Muhua Zhu, Yifei Xue, Tie Ji, Yizhen Lao  
  _2026-03-09_ · https://arxiv.org/abs/2603.07937v1  
  <details><summary>Abstract</summary>

  Standard visual localization methods typically require offline pre-processing of scenes to obtain 3D structural information for better performance. This inevitably introduces additional computational and time costs, as well as the overhead of storing scene representations. Can we visually localize in a wild scene without any off-line preprocessing step? In this paper, we leverage the online inference capabilities of feed-forward 3D reconstruction networks to propose a novel map-free visual localization framework $L^3$. Specifically, by performing direct online 3D reconstruction on RGB images, followed by two-stage metric scale recovery and pose refinement based on 2D-3D correspondences, $L^3$ achieves high accuracy without the need to pre-build or store any offline scene representations. Extensive experiments demonstrate $L^3$ not only that the performance is comparable to state-of-the-art solutions on various benchmarks, but also that it exhibits significantly superior robustness in sparse scenes (fewer reference images per scene).

  </details>



- **RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving**  
  Zhangshuo Qi, Jingyi Xu, Luqi Cheng, Shichen Wen, Guangming Xiong  
  _2026-03-09_ · https://arxiv.org/abs/2603.07920v1  
  <details><summary>Abstract</summary>

  All-weather autonomy is critical for autonomous driving, which necessitates reliable localization across diverse scenarios. While LiDAR place recognition is widely deployed for this task, its performance degrades in adverse weather. Conversely, radar-based methods, though weather-resilient, are hindered by the general unavailability of radar maps. To bridge this gap, radar-to-LiDAR place recognition, which localizes radar scans within existing LiDAR maps, has garnered increasing interest. However, extracting discriminative and generalizable features shared between modalities remains challenging, compounded by the scarcity of large-scale paired training data and the signal heterogeneity across radar types. In this work, we propose RLPR, a robust radar-to-LiDAR place recognition framework compatible with single-chip, scanning, and 4D radars. We first design a dual-stream network to extract structural features that abstract away from sensor-specific signal properties (e.g., Doppler or RCS). Subsequently, motivated by our task-specific asymmetry observation between radar and LiDAR, we introduce a two-stage asymmetric cross-modal alignment (TACMA) strategy, which leverages the pre-trained radar branch as a discriminative anchor to guide the alignment process. Experiments on four datasets demonstrate that RLPR achieves state-of-the-art recognition accuracy with strong zero-shot generalization capabilities.

  </details>



- **FrameVGGT: Frame Evidence Rolling Memory for streaming VGGT**  
  Zhisong Xu, Takeshi Oishi  
  _2026-03-08_ · https://arxiv.org/abs/2603.07690v1  
  <details><summary>Abstract</summary>

  Streaming Visual Geometry Transformers such as StreamVGGT enable strong online 3D perception but suffer from unbounded KV-cache growth, which limits deployment over long streams. We revisit bounded-memory streaming from the perspective of geometric support. In geometry-driven reasoning, memory quality depends not only on how many tokens are retained, but also on whether the retained memory still preserves sufficiently coherent local support. This suggests that token-level retention may become less suitable under fixed budgets, as it can thin the evidence available within each contributing frame and make subsequent fusion more sensitive to weakly aligned history. Motivated by this observation, we propose FrameVGGT, a frame-driven rolling explicit-memory framework that treats each frame's incremental KV contribution as a coherent evidence block. FrameVGGT summarizes each block into a compact prototype and maintains a fixed-capacity mid-term bank of complementary frame blocks under strict budgets, with an optional lightweight anchor tier for rare prolonged degradation. Across long-sequence 3D reconstruction, video depth estimation, and camera pose benchmarks, FrameVGGT achieves favorable accuracy--memory trade-offs under bounded memory, while maintaining more stable geometry over long streams.

  </details>



- **QdaVPR: A novel query-based domain-agnostic model for visual place recognition**  
  Shanshan Wan, Lai Kang, Yingmei Wei, Tianrui Shen, Haixuan Wang, Chao Zuo  
  _2026-03-08_ · https://arxiv.org/abs/2603.07414v1  
  <details><summary>Abstract</summary>

  Visual place recognition (VPR) aiming at predicting the location of an image based solely on its visual features is a fundamental task in robotics and autonomous systems. Domain variation remains one of the main challenges in VPR and is relatively unexplored. Existing VPR models attempt to achieve domain agnosticism either by training on large-scale datasets that inherently contain some domain variations, or by being specifically adapted to particular target domains. In practice, the former lacks explicit domain supervision, while the latter generalizes poorly to unseen domain shifts. This paper proposes a novel query-based domain-agnostic VPR model called QdaVPR. First, a dual-level adversarial learning framework is designed to encourage domain invariance for both the query features forming the global descriptor and the image features from which these query features are derived. Then, a triplet supervision based on query combinations is designed to enhance the discriminative power of the global descriptors. To support the learning process, we augment a large-scale VPR dataset using style transfer methods, generating various synthetic domains with corresponding domain labels as auxiliary supervision. Extensive experiments show that QdaVPR achieves state-of-the-art performance on multiple VPR benchmarks with significant domain variations. Specifically, it attains the best Recall@1 and Recall@10 on nearly all test scenarios: 93.5%/98.6% on Nordland (seasonal changes), 97.5%/99.0% on Tokyo24/7 (day-night transitions), and the highest Recall@1 across almost all weather conditions on the SVOX dataset. Our code will be released at https://github.com/shuimushan/QdaVPR.

  </details>


