# SLAM & Localization

_Updated: 2026-02-05 07:14 UTC_

Total papers shown: **14**


---

- **How to rewrite the stars: Mapping your orchard over time through constellations of fruits**  
  Gonçalo P. Matos, Carlos Santiago, João P. Costeira, Ricardo L. Saldanha, Ernesto M. Morgado  
  _2026-02-04_ · https://arxiv.org/abs/2602.04722v1  
  <details><summary>Abstract</summary>

  Following crop growth through the vegetative cycle allows farmers to predict fruit setting and yield in early stages, but it is a laborious and non-scalable task if performed by a human who has to manually measure fruit sizes with a caliper or dendrometers. In recent years, computer vision has been used to automate several tasks in precision agriculture, such as detecting and counting fruits, and estimating their size. However, the fundamental problem of matching the exact same fruits from one video, collected on a given date, to the fruits visible in another video, collected on a later date, which is needed to track fruits' growth through time, remains to be solved. Few attempts were made, but they either assume that the camera always starts from the same known position and that there are sufficiently distinct features to match, or they used other sources of data like GPS. Here we propose a new paradigm to tackle this problem, based on constellations of 3D centroids, and introduce a descriptor for very sparse 3D point clouds that can be used to match fruits across videos. Matching constellations instead of individual fruits is key to deal with non-rigidity, occlusions and challenging imagery with few distinct visual features to track. The results show that the proposed method can be successfully used to match fruits across videos and through time, and also to build an orchard map and later use it to locate the camera pose in 6DoF, thus providing a method for autonomous navigation of robots in the orchard and for selective fruit picking, for example.

  </details>



- **Radar-Inertial Odometry For Computationally Constrained Aerial Navigation**  
  Jan Michalczyk  
  _2026-02-04_ · https://arxiv.org/abs/2602.04631v1  
  <details><summary>Abstract</summary>

  Recently, the progress in the radar sensing technology consisting in the miniaturization of the packages and increase in measuring precision has drawn the interest of the robotics research community. Indeed, a crucial task enabling autonomy in robotics is to precisely determine the pose of the robot in space. To fulfill this task sensor fusion algorithms are often used, in which data from one or several exteroceptive sensors like, for example, LiDAR, camera, laser ranging sensor or GNSS are fused together with the Inertial Measurement Unit (IMU) measurements to obtain an estimate of the navigation states of the robot. Nonetheless, owing to their particular sensing principles, some exteroceptive sensors are often incapacitated in extreme environmental conditions, like extreme illumination or presence of fine particles in the environment like smoke or fog. Radars are largely immune to aforementioned factors thanks to the characteristics of electromagnetic waves they use. In this thesis, we present Radar-Inertial Odometry (RIO) algorithms to fuse the information from IMU and radar in order to estimate the navigation states of a (Uncrewed Aerial Vehicle) UAV capable of running on a portable resource-constrained embedded computer in real-time and making use of inexpensive, consumer-grade sensors. We present novel RIO approaches relying on the multi-state tightly-coupled Extended Kalman Filter (EKF) and Factor Graphs (FG) fusing instantaneous velocities of and distances to 3D points delivered by a lightweight, low-cost, off-the-shelf Frequency Modulated Continuous Wave (FMCW) radar with IMU readings. We also show a novel way to exploit advances in deep learning to retrieve 3D point correspondences in sparse and noisy radar point clouds.

  </details>



- **S-MUSt3R: Sliding Multi-view 3D Reconstruction**  
  Leonid Antsfeld, Boris Chidlovskii, Yohann Cabon, Vincent Leroy, Jerome Revaud  
  _2026-02-04_ · https://arxiv.org/abs/2602.04517v1  
  <details><summary>Abstract</summary>

  The recent paradigm shift in 3D vision led to the rise of foundation models with remarkable capabilities in 3D perception from uncalibrated images. However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations. This work proposes S-MUSt3R, a simple and efficient pipeline that extends the limits of foundation models for monocular 3D reconstruction. Our approach addresses the scalability bottleneck of foundation models through a simple strategy of sequence segmentation followed by segment alignment and lightweight loop closure optimization. Without model retraining, we benefit from remarkable 3D reconstruction capacities of MUSt3R model and achieve trajectory and reconstruction performance comparable to traditional methods with more complex architecture. We evaluate S-MUSt3R on TUM, 7-Scenes and proprietary robot navigation datasets and show that S-MUSt3R runs successfully on long RGB sequences and produces accurate and consistent 3D reconstruction. Our results highlight the potential of leveraging the MUSt3R model for scalable monocular 3D scene in real-world settings, with an important advantage of making predictions directly in the metric space.

  </details>



- **TrajVG: 3D Trajectory-Coupled Visual Geometry Learning**  
  Xingyu Miao, Weiguang Zhao, Tao Lu, Linning Yu, Mulin Yu, Yang Long, Jiangmiao Pang, Junting Dong  
  _2026-02-04_ · https://arxiv.org/abs/2602.04439v1  
  <details><summary>Abstract</summary>

  Feed-forward multi-frame 3D reconstruction models often degrade on videos with object motion. Global-reference becomes ambiguous under multiple motions, while the local pointmap relies heavily on estimated relative poses and can drift, causing cross-frame misalignment and duplicated structures. We propose TrajVG, a reconstruction framework that makes cross-frame 3D correspondence an explicit prediction by estimating camera-coordinate 3D trajectories. We couple sparse trajectories, per-frame local point maps, and relative camera poses with geometric consistency objectives: (i) bidirectional trajectory-pointmap consistency with controlled gradient flow, and (ii) a pose consistency objective driven by static track anchors that suppresses gradients from dynamic regions. To scale training to in-the-wild videos where 3D trajectory labels are scarce, we reformulate the same coupling constraints into self-supervised objectives using only pseudo 2D tracks, enabling unified training with mixed supervision. Extensive experiments across 3D tracking, pose estimation, pointmap reconstruction, and video depth show that TrajVG surpasses the current feedforward performance baseline.

  </details>



- **Quantile Transfer for Reliable Operating Point Selection in Visual Place Recognition**  
  Dhyey Manish Rajani, Michael Milford, Tobias Fischer  
  _2026-02-04_ · https://arxiv.org/abs/2602.04401v1  
  <details><summary>Abstract</summary>

  Visual Place Recognition (VPR) is a key component for localisation in GNSS-denied environments, but its performance critically depends on selecting an image matching threshold (operating point) that balances precision and recall. Thresholds are typically hand-tuned offline for a specific environment and fixed during deployment, leading to degraded performance under environmental change. We propose a method that, given a user-defined precision requirement, automatically selects the operating point of a VPR system to maximise recall. The method uses a small calibration traversal with known correspondences and transfers thresholds to deployment via quantile normalisation of similarity score distributions. This quantile transfer ensures that thresholds remain stable across calibration sizes and query subsets, making the method robust to sampling variability. Experiments with multiple state-of-the-art VPR techniques and datasets show that the proposed approach consistently outperforms the state-of-the-art, delivering up to 25% higher recall in high-precision operating regimes. The method eliminates manual tuning by adapting to new environments and generalising across operating conditions. Our code will be released upon acceptance.

  </details>



- **Beyond Static Cropping: Layer-Adaptive Visual Localization and Decoding Enhancement**  
  Zipeng Zhu, Zhanghao Hu, Qinglin Zhu, Yuxi Hong, Yijun Liu, Jingyong Su, Yulan He, Lin Gui  
  _2026-02-04_ · https://arxiv.org/abs/2602.04304v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (LVLMs) have advanced rapidly by aligning visual patches with the text embedding space, but a fixed visual-token budget forces images to be resized to a uniform pretraining resolution, often erasing fine-grained details and causing hallucinations via over-reliance on language priors. Recent attention-guided enhancement (e.g., cropping or region-focused attention allocation) alleviates this, yet it commonly hinges on a static "magic layer" empirically chosen on simple recognition benchmarks and thus may not transfer to complex reasoning tasks. In contrast to this static assumption, we propose a dynamic perspective on visual grounding. Through a layer-wise sensitivity analysis, we demonstrate that visual grounding is a dynamic process: while simple object recognition tasks rely on middle layers, complex visual search and reasoning tasks require visual information to be reactivated at deeper layers. Based on this observation, we introduce Visual Activation by Query (VAQ), a metric that identifies the layer whose attention map is most relevant to query-specific visual grounding by measuring attention sensitivity to the input query. Building on VAQ, we further propose LASER (Layer-adaptive Attention-guided Selective visual and decoding Enhancement for Reasoning), a training-free inference procedure that adaptively selects task-appropriate layers for visual localization and question answering. Experiments across diverse VQA benchmarks show that LASER significantly improves VQA accuracy across tasks with varying levels of complexity.

  </details>



- **Control and State Estimation of Vehicle-Mounted Aerial Systems in GPS-Denied, Non-Inertial Environments**  
  Riming Xu, Obadah Wali, Yasmine Marani, Eric Feron  
  _2026-02-03_ · https://arxiv.org/abs/2602.04057v1  
  <details><summary>Abstract</summary>

  We present a robust control and estimation framework for quadrotors operating in Global Navigation Satellite System(GNSS)-denied, non-inertial environments where inertial sensors such as Inertial Measurement Units (IMUs) become unreliable due to platform-induced accelerations. In such settings, conventional estimators fail to distinguish whether the measured accelerations arise from the quadrotor itself or from the non-inertial platform, leading to drift and control degradation. Unlike conventional approaches that depend heavily on IMU and GNSS, our method relies exclusively on external position measurements combined with a Extended Kalman Filter with Unknown Inputs (EKF-UI) to account for platform motion. The estimator is paired with a cascaded PID controller for full 3D tracking. To isolate estimator performance from localization errors, all tests are conducted using high-precision motion capture systems. Experimental results in a moving-cart testbed validate our approach under both translational in X-axis and Y-axis dissonance. Compared to standard EKF, the proposed method significantly improves stability and trajectory tracking without requiring inertial feedback, enabling practical deployment on moving platforms such as trucks or elevators.

  </details>



- **DADP: Domain Adaptive Diffusion Policy**  
  Pengcheng Wang, Qinghang Liu, Haotian Lin, Yiheng Li, Guojian Zhan, Masayoshi Tomizuka, Yixiao Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.04037v1  
  <details><summary>Abstract</summary>

  Learning domain adaptive policies that can generalize to unseen transition dynamics, remains a fundamental challenge in learning-based control. Substantial progress has been made through domain representation learning to capture domain-specific information, thus enabling domain-aware decision making. We analyze the process of learning domain representations through dynamical prediction and find that selecting contexts adjacent to the current step causes the learned representations to entangle static domain information with varying dynamical properties. Such mixture can confuse the conditioned policy, thereby constraining zero-shot adaptation. To tackle the challenge, we propose DADP (Domain Adaptive Diffusion Policy), which achieves robust adaptation through unsupervised disentanglement and domain-aware diffusion injection. First, we introduce Lagged Context Dynamical Prediction, a strategy that conditions future state estimation on a historical offset context; by increasing this temporal gap, we unsupervisedly disentangle static domain representations by filtering out transient properties. Second, we integrate the learned domain representations directly into the generative process by biasing the prior distribution and reformulating the diffusion target. Extensive experiments on challenging benchmarks across locomotion and manipulation demonstrate the superior performance, and the generalizability of DADP over prior methods. More visualization results are available on the https://outsider86.github.io/DomainAdaptiveDiffusionPolicy/.

  </details>



- **Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios**  
  Kuo-Yi Chao, Ralph Rasshofer, Alois Christian Knoll  
  _2026-02-03_ · https://arxiv.org/abs/2602.03908v1  
  <details><summary>Abstract</summary>

  Accurate vehicle localization is a critical challenge in urban environments where GPS signals are often unreliable. This paper presents a cooperative multi-sensor and multi-modal localization approach to address this issue by fusing data from vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) systems. Our approach integrates cooperative data with a point cloud registration-based simultaneous localization and mapping (SLAM) algorithm. The system processes point clouds generated from diverse sensor modalities, including vehicle-mounted LiDAR and stereo cameras, as well as sensors deployed at intersections. By leveraging shared data from infrastructure, our method significantly improves localization accuracy and robustness in complex, GPS-noisy urban scenarios.

  </details>



- **Z3D: Zero-Shot 3D Visual Grounding from Images**  
  Nikita Drozdov, Andrey Lemeshko, Nikita Gavrilov, Anton Konushin, Danila Rukhovich, Maksim Kolodiazhnyi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03361v1  
  <details><summary>Abstract</summary>

  3D visual grounding (3DVG) aims to localize objects in a 3D scene based on natural language queries. In this work, we explore zero-shot 3DVG from multi-view images alone, without requiring any geometric supervision or object priors. We introduce Z3D, a universal grounding pipeline that flexibly operates on multi-view images while optionally incorporating camera poses and depth maps. We identify key bottlenecks in prior zero-shot methods causing significant performance degradation and address them with (i) a state-of-the-art zero-shot 3D instance segmentation method to generate high-quality 3D bounding box proposals and (ii) advanced reasoning via prompt-based segmentation, which utilizes full capabilities of modern VLMs. Extensive experiments on the ScanRefer and Nr3D benchmarks demonstrate that our approach achieves state-of-the-art performance among zero-shot methods. Code is available at https://github.com/col14m/z3d .

  </details>



- **Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization**  
  Manuel Hofer, Markus Steinberger, Thomas Köhler  
  _2026-02-03_ · https://arxiv.org/abs/2602.03327v1  
  <details><summary>Abstract</summary>

  Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.

  </details>



- **LEVIO: Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices**  
  Jonas Kühne, Christian Vogt, Michele Magno, Luca Benini  
  _2026-02-03_ · https://arxiv.org/abs/2602.03294v1  
  <details><summary>Abstract</summary>

  Accurate, infrastructure-less sensor systems for motion tracking are essential for mobile robotics and augmented reality (AR) applications. The most popular state-of-the-art visual-inertial odometry (VIO) systems, however, are too computationally demanding for resource-constrained hardware, such as micro-drones and smart glasses. This work presents LEVIO, a fully featured VIO pipeline optimized for ultra-low-power compute platforms, allowing six-degrees-of-freedom (DoF) real-time sensing. LEVIO incorporates established VIO components such as Oriented FAST and Rotated BRIEF (ORB) feature tracking and bundle adjustment, while emphasizing a computationally efficient architecture with parallelization and low memory usage to suit embedded microcontrollers and low-power systems-on-chip (SoCs). The paper proposes and details the algorithmic design choices and the hardware-software co-optimization approach, and presents real-time performance on resource-constrained hardware. LEVIO is validated on a parallel-processing ultra-low-power RISC-V SoC, achieving 20 FPS while consuming less than 100 mW, and benchmarked against public VIO datasets, offering a compelling balance between efficiency and accuracy. To facilitate reproducibility and adoption, the complete implementation is released as open-source.

  </details>



- **LaVPR: Benchmarking Language and Vision for Place Recognition**  
  Ofer Idan, Dan Badur, Yosi Keller, Yoli Shavit  
  _2026-02-03_ · https://arxiv.org/abs/2602.03253v1  
  <details><summary>Abstract</summary>

  Visual Place Recognition (VPR) often fails under extreme environmental changes and perceptual aliasing. Furthermore, standard systems cannot perform "blind" localization from verbal descriptions alone, a capability needed for applications such as emergency response. To address these challenges, we introduce LaVPR, a large-scale benchmark that extends existing VPR datasets with over 650,000 rich natural-language descriptions. Using LaVPR, we investigate two paradigms: Multi-Modal Fusion for enhanced robustness and Cross-Modal Retrieval for language-based localization. Our results show that language descriptions yield consistent gains in visually degraded conditions, with the most significant impact on smaller backbones. Notably, adding language allows compact models to rival the performance of much larger vision-only architectures. For cross-modal retrieval, we establish a baseline using Low-Rank Adaptation (LoRA) and Multi-Similarity loss, which substantially outperforms standard contrastive methods across vision-language models. Ultimately, LaVPR enables a new class of localization systems that are both resilient to real-world stochasticity and practical for resource-constrained deployment. Our dataset and code are available at https://github.com/oferidan1/LaVPR.

  </details>



- **From Single Scan to Sequential Consistency: A New Paradigm for LIDAR Relocalization**  
  Minghang Zhu, Zhijing Wang, Yuxin Guo, Wen Li, Sheng Ao, Cheng Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03198v1  
  <details><summary>Abstract</summary>

  LiDAR relocalization aims to estimate the global 6-DoF pose of a sensor in the environment. However, existing regression-based approaches are prone to dynamic or ambiguous scenarios, as they either solely rely on single-frame inference or neglect the spatio-temporal consistency across scans. In this paper, we propose TempLoc, a new LiDAR relocalization framework that enhances the robustness of localization by effectively modeling sequential consistency. Specifically, a Global Coordinate Estimation module is first introduced to predict point-wise global coordinates and associated uncertainties for each LiDAR scan. A Prior Coordinate Generation module is then presented to estimate inter-frame point correspondences by the attention mechanism. Lastly, an Uncertainty-Guided Coordinate Fusion module is deployed to integrate both predictions of point correspondence in an end-to-end fashion, yielding a more temporally consistent and accurate global 6-DoF pose. Experimental results on the NCLT and Oxford Robot-Car benchmarks show that our TempLoc outperforms stateof-the-art methods by a large margin, demonstrating the effectiveness of temporal-aware correspondence modeling in LiDAR relocalization. Our code will be released soon.

  </details>


