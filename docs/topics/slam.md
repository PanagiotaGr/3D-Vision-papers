# SLAM & Localization

_Updated: 2026-04-01 07:51 UTC_

Total papers shown: **13**


---

- **Semantic Zone-Based Map Management for Stable AI-Integrated Mobile Robots**  
  Huichang Yun, Seungho Yoo  
  _2026-03-31_ · https://arxiv.org/abs/2603.29627v1  
  <details><summary>Abstract</summary>

  Recent advances in large AI models (VLMs and LLMs) and joint use of the 3D dense maps, enable mobile robots to provide more powerful and interactive services grounded in rich spatial context. However, deploying both heavy AI models and dense maps on edge robots is challenging under strict memory budgets. When the memory budget is exceeded, required keyframes may not be loaded in time, which can degrade the stability of position estimation and interfering model performance. We proposes a semantic zone-based map management approach to stabilize dense-map utilization under memory constraints. We associate keyframes with semantic indoor regions (e.g., rooms and corridors) and keyframe management at the semantic zone level prioritizes spatially relevant map content while respecting memory constraints. This reduces keyframe loading and unloading frequency and memory usage. We evaluate the proposed approach in large-scale simulated indoor environments and on an NVIDIA Jetson Orin Nano under concurrent SLAM-VLM execution. With Qwen3.5:0.8b, the proposed method improves throughput by 3.3 tokens/s and reduces latency by 21.7% relative to a geometric map-management strategy. Furthermore, while the geometric strategy suffers from out-of-memory failures and stalled execution under memory pressure, the proposed method eliminates both issues, preserving localization stability and enabling robust VLM operation. These results demonstrate that the proposed approach enables efficient dense map utilization for memory constrained, AI-integrated mobile robots. Code is available at: https://github.com/huichangs/rtabmap/tree/segment

  </details>



- **Communication Outage-Resistant UUV State Estimation: A Variational History Distillation Approach**  
  Shuyue Li, Miguel López-Benítez, Eng Gee Lim, Fei Ma, Qian Dong, Mengze Cao, Limin Yu, Xiaohui Qin  
  _2026-03-31_ · https://arxiv.org/abs/2603.29512v1  
  <details><summary>Abstract</summary>

  The reliable operation of Unmanned Underwater Vehicle (UUV) clusters is highly dependent on continuous acoustic communication. However, this communication method is highly susceptible to intermittent interruptions. When communication outages occur, standard state estimators such as the Unscented Kalman Filter (UKF) will be forced to make open-loop predictions. If the environment contains unmodeled dynamic factors, such as unknown ocean currents, this estimation error will grow rapidly, which may eventually lead to mission failure. To address this critical issue, this paper proposes a Variational History Distillation (VHD) approach. VHD regards trajectory prediction as an approximate Bayesian reasoning process, which links a standard motion model based on physics with a pattern extracted directly from the past trajectory of the UUV. This is achieved by synthesizing ``virtual measurements'' distilled from historical trajectories. Recognizing that the reliability of extrapolated historical trends degrades over extended prediction horizons, an adaptive confidence mechanism is introduced. This mechanism allows the filter to gradually reduce the trust of virtual measurements as the communication outage time is extended. Extensive Monte Carlo simulations in a high-fidelity environment demonstrate that the proposed method achieves a 91\% reduction in prediction Root Mean Square Error (RMSE), reducing the error from approximately 170 m to 15 m during a 40-second communication outage. These results demonstrate that VHD can maintain robust state estimation performance even under complete communication loss.

  </details>



- **All-in-One Augmented Reality Guided Head and Neck Tumor Resection**  
  Yue Yang, Matthieu Chabanas, Carrie Reale, Annie Benson, Jason Slagle, Matthew Weinger, Michael Topf, Jie Ying Wu  
  _2026-03-31_ · https://arxiv.org/abs/2603.29495v1  
  <details><summary>Abstract</summary>

  Positive margins are common in head and neck squamous cell carcinoma, yet intraoperative re-resection is often imprecise because margin locations are typically communicated verbally from pathology. We present an all-in-one augmented reality (AR) system that relocalizes positive margins from a resected specimen to the resection bed and visualizes them in situ using HoloLens 2 depth sensing and fully automated markerless surface registration. In a silicone phantom study with six medical trainees, markerless registration achieved target registration errors comparable to a marker-based baseline (median 1.8 mm vs. 1.7 mm; maximum < 4 mm). In a margin relocalization task, AR guidance reduced error from verbal guidance (median 14.2 mm) to a few millimeters (median 3.2 mm), with all AR localizations within 5 mm error. These results support the feasibility of markerless AR margin guidance for more precise intraoperative re-excision.

  </details>



- **Interacting Multiple Model Proprioceptive Odometry for Legged Robots**  
  Wanlei Li, Zichang Chen, Shilei Li, Xiaogang Xiong, Yunjiang Lou  
  _2026-03-31_ · https://arxiv.org/abs/2603.29383v1  
  <details><summary>Abstract</summary>

  State estimation for legged robots remains challenging because legged odometry generally suffers from limited observability and therefore depends critically on measurement constraints to suppress drift. When exteroceptive sensors are unreliable or degraded, such constraints are mainly derived from proprioceptive measurements, particularly contact-related leg kinematics information. However, most existing proprioceptive odometry methods rely on an idealized point-contact assumption, which is often violated during real locomotion. Consequently, the effectiveness of proprioceptive constraints may be significantly reduced, resulting in degraded estimation accuracy. To address these limitations, we propose an interacting multiple model (IMM)-based proprioceptive odometry framework for legged robots. By incorporating multiple contact hypotheses within a unified probabilistic framework, the proposed method enables online mode switching and probabilistic fusion under varying contact conditions. Extensive simulations and real-world experiments demonstrate that the proposed method achieves superior pose estimation accuracy over state-of-the-art methods while maintaining comparable computational efficiency.

  </details>



- **StereoVGGT: A Training-Free Visual Geometry Transformer for Stereo Vision**  
  Ziyang Chen, Yansong Qu, You Shen, Xuan Cheng, Liujuan Cao  
  _2026-03-31_ · https://arxiv.org/abs/2603.29368v1  
  <details><summary>Abstract</summary>

  Driven by the advancement of 3D devices, stereo vision tasks including stereo matching and stereo conversion have emerged as a critical research frontier. Contemporary stereo vision backbones typically rely on either monocular depth estimation (MDE) models or visual foundation models (VFMs). Crucially, these models are predominantly pretrained without explicit supervision of camera poses. Given that such geometric knowledge is indispensable for stereo vision, the absence of explicit spatial constraints constitutes a significant performance bottleneck for existing architectures. Recognizing that the Visual Geometry Grounded Transformer (VGGT) operates as a foundation model pretrained on extensive 3D priors, including camera poses, we investigate its potential as a robust backbone for stereo vision tasks. Nevertheless, empirical results indicate that its direct application to stereo vision yields suboptimal performance. We observe that VGGT suffers from a more significant degradation of geometric details during feature extraction. Such characteristics conflict with the requirements of binocular stereo vision, thereby constraining its efficacy for relative tasks. To bridge this gap, we propose StereoVGGT, a feature backbone specifically tailored for stereo vision. By leveraging the frozen VGGT and introducing a training-free feature adjustment pipeline, we mitigate geometric degradation and harness the latent camera calibration knowledge embedded within the model. StereoVGGT-based stereo matching network achieved the $1^{st}$ rank among all published methods on the KITTI benchmark, validating that StereoVGGT serves as a highly effective backbone for stereo vision.

  </details>



- **MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting**  
  Haoran Zhou, Gim Hee Lee  
  _2026-03-31_ · https://arxiv.org/abs/2603.29296v1  
  <details><summary>Abstract</summary>

  Realistic reconstruction of dynamic 4D scenes from monocular videos is essential for understanding the physical world. Despite recent progress in neural rendering, existing methods often struggle to recover accurate 3D geometry and temporally consistent motion in complex environments. To address these challenges, we propose MotionScale, a 4D Gaussian Splatting framework that scales efficiently to large scenes and extended sequences while maintaining high-fidelity structural and motion coherence. At the core of our approach is a scalable motion field parameterized by cluster-centric basis transformations that adaptively expand to capture diverse and evolving motion patterns. To ensure robust reconstruction over long durations, we introduce a progressive optimization strategy comprising two decoupled propagation stages: 1) A background extension stage that adapts to newly visible regions, refines camera poses, and explicitly models transient shadows; 2) A foreground propagation stage that enforces motion consistency through a specialized three-stage refinement process. Extensive experiments on challenging real-world benchmarks demonstrate that MotionScale significantly outperforms state-of-the-art methods in both reconstruction quality and temporal stability. Project page: https://hrzhou2.github.io/motion-scale-web/.

  </details>



- **M2H-MX: Multi-Task Dense Visual Perception for Real-Time Monocular Spatial Understanding**  
  U. V. B. L. Udugama, George Vosselman, Francesco Nex  
  _2026-03-31_ · https://arxiv.org/abs/2603.29236v1  
  <details><summary>Abstract</summary>

  Monocular cameras are attractive for robotic perception due to their low cost and ease of deployment, yet achieving reliable real-time spatial understanding from a single image stream remains challenging. While recent multi-task dense prediction models have improved per-pixel depth and semantic estimation, translating these advances into stable monocular mapping systems is still non-trivial. This paper presents M2H-MX, a real-time multi-task perception model for monocular spatial understanding. The model preserves multi-scale feature representations while introducing register-gated global context and controlled cross-task interaction in a lightweight decoder, enabling depth and semantic predictions to reinforce each other under strict latency constraints. Its outputs integrate directly into an unmodified monocular SLAM pipeline through a compact perception-to-mapping interface. We evaluate both dense prediction accuracy and in-the-loop system performance. On NYUDv2, M2H-MX-L achieves state-of-the-art results, improving semantic mIoU by 6.6% and reducing depth RMSE by 9.4% over representative multi-task baselines. When deployed in a real-time monocular mapping system on ScanNet, M2H-MX reduces average trajectory error by 60.7% compared to a strong monocular SLAM baseline while producing cleaner metric-semantic maps. These results demonstrate that modern multi-task dense prediction can be reliably deployed for real-time monocular spatial perception in robotic systems.

  </details>



- **Efficient Camera Pose Augmentation for View Generalization in Robotic Policy Learning**  
  Sen Wang, Huaiyi Dong, Jingyi Tian, Jiayi Li, Zhuo Yang, Tongtong Cao, Anlin Chen, Shuang Wu, Le Wang, Sanping Zhou  
  _2026-03-31_ · https://arxiv.org/abs/2603.29192v1  
  <details><summary>Abstract</summary>

  Prevailing 2D-centric visuomotor policies exhibit a pronounced deficiency in novel view generalization, as their reliance on static observations hinders consistent action mapping across unseen views. In response, we introduce GenSplat, a feed-forward 3D Gaussian Splatting framework that facilitates view-generalized policy learning through novel view rendering. GenSplat employs a permutation-equivariant architecture to reconstruct high-fidelity 3D scenes from sparse, uncalibrated inputs in a single forward pass. To ensure structural integrity, we design a 3D-prior distillation strategy that regularizes the 3DGS optimization, preventing the geometric collapse typical of purely photometric supervision. By rendering diverse synthetic views from these stable 3D representations, we systematically augment the observational manifold during training. This augmentation forces the policy to ground its decisions in underlying 3D structures, thereby ensuring robust execution under severe spatial perturbations where baselines severely degrade.

  </details>



- **Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting**  
  Huaqi Tao, Bingxi Liu, Guangcheng Chen, Fulin Tang, Li He, Hong Zhang  
  _2026-03-31_ · https://arxiv.org/abs/2603.29185v1  
  <details><summary>Abstract</summary>

  Visual relocalization is a fundamental task in the field of 3D computer vision, estimating a camera's pose when it revisits a previously known scene. While point-based hierarchical relocalization methods have shown strong scalability and efficiency, they are often limited by sparse image observations and weak feature matching. In this work, we propose SplatHLoc, a novel hierarchical visual relocalization framework that uses Feature Gaussian Splatting as the scene representation. To address the sparsity of database images, we propose an adaptive viewpoint retrieval method that synthesizes virtual candidates with viewpoints more closely aligned with the query, thereby improving the accuracy of initial pose estimation. For feature matching, we observe that Gaussian-rendered features and those extracted directly from images exhibit different strengths across the two-stage matching process: the former performs better in the coarse stage, while the latter proves more effective in the fine stage. Therefore, we introduce a hybrid feature matching strategy, enabling more accurate and efficient pose estimation. Extensive experiments on both indoor and outdoor datasets show that SplatHLoc enhances the robustness of visual relocalization, setting a new state-of-the-art.

  </details>



- **Fisheye3R: Adapting Unified 3D Feed-Forward Foundation Models to Fisheye Lenses**  
  Ruxiao Duan, Erin Hong, Dongxu Zhao, Eric Turner, Alex Wong, Yunwen Zhou  
  _2026-03-30_ · https://arxiv.org/abs/2603.28896v1  
  <details><summary>Abstract</summary>

  Feed-forward foundation models for multi-view 3-dimensional (3D) reconstruction have been trained on large-scale datasets of perspective images; when tested on wide field-of-view images, e.g., from a fisheye camera, their performance degrades. Their error arises from changes in spatial positions of pixels due to a non-linear projection model that maps 3D points onto the 2D image plane. While one may surmise that training on fisheye images would resolve this problem, there are far fewer fisheye images with ground truth than perspective images, which limit generalization. To enable inference on imagery exhibiting high radial distortion, we propose Fisheye3R, a novel adaptation framework that extends these multi-view 3D reconstruction foundation models to natively accommodate fisheye inputs without performance regression on perspective images. To address the scarcity of fisheye images and ground truth, we introduce flexible learning schemes that support self-supervised adaptation using only unlabeled perspective images and supervised adaptation without any fisheye training data. Extensive experiments across three foundation models, including VGGT, $π^3$, and MapAnything, demonstrate that our approach consistently improves camera pose, depth, point map, and field-of-view estimation on fisheye images.

  </details>



- **TerraSky3D: Multi-View Reconstructions of European Landmarks in 4K**  
  Mattia D'Urso, Yuxi Hu, Christian Sormann, Mattia Rossi, Friedrich Fraundorfer  
  _2026-03-30_ · https://arxiv.org/abs/2603.28287v1  
  <details><summary>Abstract</summary>

  Despite the growing need for data of more and more sophisticated 3D reconstruction pipelines, we can still observe a scarcity of suitable public datasets. Existing 3D datasets are either low resolution, limited to a small amount of scenes, based on images of varying quality because retrieved from the internet, or limited to specific capturing scenarios. Motivated by this lack of suitable 3D datasets, we captured TerraSky3D, a high-resolution large-scale 3D reconstruction dataset comprising 50,000 images divided into 150 ground, aerial, and mixed scenes. The dataset focuses on European landmarks and comes with curated calibration data, camera poses, and depth maps. TerraSky3D tries to answer the need for challenging dataset that can be used to train and evaluate 3D reconstruction-related pipelines.

  </details>



- **Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal**  
  Kazuma Ikeda, Ryosei Hara, Rokuto Nagata, Ozora Sako. Zihao Ding, Takahiro Kado, Ibuki Fujioka, Taro Beppu, Mariko Isogawa, Kentaro Yoshioka  
  _2026-03-30_ · https://arxiv.org/abs/2603.28224v1  
  <details><summary>Abstract</summary>

  LiDAR has become an essential sensing modality in autonomous driving, robotics, and smart-city applications. However, ghost points (or ghosts), which are false reflections caused by multi-path laser returns from glass and reflective surfaces, severely degrade 3D mapping and localization accuracy. Prior ghost removal relies on geometric consistency in dense point clouds, failing on mobile LiDAR's sparse, dynamic data. We address this by exploiting full-waveform LiDAR (FWL), which captures complete temporal intensity profiles rather than just peak distances, providing crucial cues for distinguishing ghosts from genuine reflections in mobile scenarios. As this is a new task, we present Ghost-FWL, the first and largest annotated mobile FWL dataset for ghost detection and removal. Ghost-FWL comprises 24K frames across 10 diverse scenes with 7.5 billion peak-level annotations, which is 100x larger than existing annotated FWL datasets. Benefiting from this large-scale dataset, we establish a FWL-based baseline model for ghost detection and propose FWL-MAE, a masked autoencoder for efficient self-supervised representation learning on FWL data. Experiments show that our baseline outperforms existing methods in ghost removal accuracy, and our ghost removal further enhances downstream tasks such as LiDAR-based SLAM (66% trajectory error reduction) and 3D object detection (50x false positive reduction). The dataset and code is publicly available and can be accessed via the project page: https://keio-csg.github.io/Ghost-FWL

  </details>



- **A Classification of Heterogeneity in Uncrewed Vehicle Swarms and the Effects of Its Inclusion on Overall Swarm Resilience**  
  Abhishek Joshi, Abhishek Phadke, Tianxing Chu, F. Antonio Medrano  
  _2026-03-30_ · https://arxiv.org/abs/2603.28831v1  
  <details><summary>Abstract</summary>

  Combining different types of agents in uncrewed vehicle (UV) swarms has emerged as an approach to enhance mission resilience and operational capabilities across a wide range of applications. This study offers a systematic framework for grouping different types of swarms based on three main factors: agent nature (behavior and function), hardware structure (physical configuration and sensing capabilities), and operational space (domain of operation). A literature review indicates that strategic heterogeneity significantly improves swarm performance. Operational challenges, including communication architecture constraints, energy-aware coordination strategies, and control system integration, are also discussed. The analysis shows that heterogeneous swarms are more resilient because they can leverage diverse capabilities, adapt roles on the fly, and integrate data from multidimensional sensor feeds. Some important factors to consider when implementing are sim-to-real-world transfer for learned policies, standardized evaluation metrics, and control architectures that can work together. Learning-based coordination, GPS (Global Positioning System)-denied multi-robot SLAM (Simultaneous Localization and Mapping), and domain-specific commercial deployments collectively demonstrate that heterogeneous swarm technology is moving closer to readiness for high-value applications. This study offers a single taxonomy and evidence-based observations on methods for designing mission-ready heterogeneous swarms that balance complexity and increased capability.

  </details>


