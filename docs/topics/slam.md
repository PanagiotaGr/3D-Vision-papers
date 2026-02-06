# SLAM & Localization

_Updated: 2026-02-06 07:11 UTC_

Total papers shown: **15**


---

- **Predicting Camera Pose from Perspective Descriptions for Spatial Reasoning**  
  Xuejun Zhang, Aditi Tiwari, Zhenhailong Wang, Heng Ji  
  _2026-02-05_ · https://arxiv.org/abs/2602.06041v1  
  <details><summary>Abstract</summary>

  Multi-image spatial reasoning remains challenging for current multimodal large language models (MLLMs). While single-view perception is inherently 2D, reasoning over multiple views requires building a coherent scene understanding across viewpoints. In particular, we study perspective taking, where a model must build a coherent 3D understanding from multi-view observations and use it to reason from a new, language-specified viewpoint. We introduce CAMCUE, a pose-aware multi-image framework that uses camera pose as an explicit geometric anchor for cross-view fusion and novel-view reasoning. CAMCUE injects per-view pose into visual tokens, grounds natural-language viewpoint descriptions to a target camera pose, and synthesizes a pose-conditioned imagined target view to support answering. To support this setting, we curate CAMCUE-DATA with 27,668 training and 508 test instances pairing multi-view images and poses with diverse target-viewpoint descriptions and perspective-shift questions. We also include human-annotated viewpoint descriptions in the test split to evaluate generalization to human language. CAMCUE improves overall accuracy by 9.06% and predicts target poses from natural-language viewpoint descriptions with over 90% rotation accuracy within 20° and translation accuracy within a 0.5 error threshold. This direct grounding avoids expensive test-time search-and-match, reducing inference time from 256.6s to 1.45s per example and enabling fast, interactive use in real-world scenarios.

  </details>



- **NVS-HO: A Benchmark for Novel View Synthesis of Handheld Objects**  
  Musawar Ali, Manuel Carranza-García, Nicola Fioraio, Samuele Salti, Luigi Di Stefano  
  _2026-02-05_ · https://arxiv.org/abs/2602.05822v1  
  <details><summary>Abstract</summary>

  We propose NVS-HO, the first benchmark designed for novel view synthesis of handheld objects in real-world environments using only RGB inputs. Each object is recorded in two complementary RGB sequences: (1) a handheld sequence, where the object is manipulated in front of a static camera, and (2) a board sequence, where the object is fixed on a ChArUco board to provide accurate camera poses via marker detection. The goal of NVS-HO is to learn a NVS model that captures the full appearance of an object from (1), whereas (2) provides the ground-truth images used for evaluation. To establish baselines, we consider both a classical SfM pipeline and a state-of-the-art pre-trained feed-forward neural network (VGGT) as pose estimators, and train NVS models based on NeRF and Gaussian Splatting. Our experiments reveal significant performance gaps in current methods under unconstrained handheld conditions, highlighting the need for more robust approaches. NVS-HO thus offers a challenging real-world benchmark to drive progress in RGB-based novel view synthesis of handheld objects.

  </details>



- **Geometric Observability Index: An Operator-Theoretic Framework for Per-Feature Sensitivity, Weak Observability, and Dynamic Effects in SE(3) Pose Estimation**  
  Joe-Mei Feng, Sheng-Wei Yu  
  _2026-02-05_ · https://arxiv.org/abs/2602.05582v1  
  <details><summary>Abstract</summary>

  We present a unified operator-theoretic framework for analyzing per-feature sensitivity in camera pose estimation on the Lie group SE(3). Classical sensitivity tools - conditioning analyses, Euclidean perturbation arguments, and Fisher information bounds - do not explain how individual image features influence the pose estimate, nor why dynamic or inconsistent observations can disproportionately distort modern SLAM and structure-from-motion systems. To address this gap, we extend influence function theory to matrix Lie groups and derive an intrinsic perturbation operator for left-trivialized M-estimators on SE(3). The resulting Geometric Observability Index (GOI) quantifies the contribution of a single measurement through the curvature operator and the Lie algebraic structure of the observable subspace. GOI admits a spectral decomposition along the principal directions of the observable curvature, revealing a direct correspondence between weak observability and amplified sensitivity. In the population regime, GOI coincides with the Fisher information geometry on SE(3), yielding a single-measurement analogue of the Cramer-Rao bound. The same spectral mechanism explains classical degeneracies such as pure rotation and vanishing parallax, as well as dynamic feature amplification along weak curvature directions. Overall, GOI provides a geometrically consistent description of measurement influence that unifies conditioning analysis, Fisher information geometry, influence function theory, and dynamic scene detectability through the spectral geometry of the curvature operator. Because these quantities arise directly within Gauss-Newton pipelines, the curvature spectrum and GOI also yield lightweight, training-free diagnostic signals for identifying dynamic features and detecting weak observability configurations without modifying existing SLAM architectures.

  </details>



- **A Comparative Study of 3D Person Detection: Sensor Modalities and Robustness in Diverse Indoor and Outdoor Environments**  
  Malaz Tamim, Andrea Matic-Flierl, Karsten Roscher  
  _2026-02-05_ · https://arxiv.org/abs/2602.05538v1  
  <details><summary>Abstract</summary>

  Accurate 3D person detection is critical for safety in applications such as robotics, industrial monitoring, and surveillance. This work presents a systematic evaluation of 3D person detection using camera-only, LiDAR-only, and camera-LiDAR fusion. While most existing research focuses on autonomous driving, we explore detection performance and robustness in diverse indoor and outdoor scenes using the JRDB dataset. We compare three representative models - BEVDepth (camera), PointPillars (LiDAR), and DAL (camera-LiDAR fusion) - and analyze their behavior under varying occlusion and distance levels. Our results show that the fusion-based approach consistently outperforms single-modality models, particularly in challenging scenarios. We further investigate robustness against sensor corruptions and misalignments, revealing that while DAL offers improved resilience, it remains sensitive to sensor misalignment and certain LiDAR-based corruptions. In contrast, the camera-based BEVDepth model showed the lowest performance and was most affected by occlusion, distance, and noise. Our findings highlight the importance of utilizing sensor fusion for enhanced 3D person detection, while also underscoring the need for ongoing research to address the vulnerabilities inherent in these systems.

  </details>



- **VGGT-Motion: Motion-Aware Calibration-Free Monocular SLAM for Long-Range Consistency**  
  Zhuang Xiong, Chen Zhang, Qingshan Xu, Wenbing Tao  
  _2026-02-05_ · https://arxiv.org/abs/2602.05508v1  
  <details><summary>Abstract</summary>

  Despite recent progress in calibration-free monocular SLAM via 3D vision foundation models, scale drift remains severe on long sequences. Motion-agnostic partitioning breaks contextual coherence and causes zero-motion drift, while conventional geometric alignment is computationally expensive. To address these issues, we propose VGGT-Motion, a calibration-free SLAM system for efficient and robust global consistency over kilometer-scale trajectories. Specifically, we first propose a motion-aware submap construction mechanism that uses optical flow to guide adaptive partitioning, prune static redundancy, and encapsulate turns for stable local geometry. We then design an anchor-driven direct Sim(3) registration strategy. By exploiting context-balanced anchors, it achieves search-free, pixel-wise dense alignment and efficient loop closure without costly feature matching. Finally, a lightweight submap-level pose graph optimization enforces global consistency with linear complexity, enabling scalable long-range operation. Experiments show that VGGT-Motion markedly improves trajectory accuracy and efficiency, achieving state-of-the-art performance in zero-shot, long-range calibration-free monocular SLAM.

  </details>



- **Feature points evaluation on omnidirectional vision with a photorealistic fisheye sequence -- A report on experiments done in 2014**  
  Julien Moreau, S. Ambellouis, Yassine Ruichek  
  _2026-02-05_ · https://arxiv.org/abs/2602.05487v1  
  <details><summary>Abstract</summary>

  What is this report: This is a scientific report, contributing with a detailed bibliography, a dataset which we will call now PFSeq for ''Photorealistic Fisheye Sequence'' and make available at https://doi.org/10. 57745/DYIVVU, and comprehensive experiments. This work should be considered as a draft, and has been done during my PhD thesis ''Construction of 3D models from fisheye video data-Application to the localisation in urban area'' in 2014 [Mor16]. These results have never been published. The aim was to find the best features detector and descriptor for fisheye images, in the context of selfcalibration, with cameras mounted on the top of a car and aiming at the zenith (to proceed then fisheye visual odometry and stereovision in urban scenes). We face a chicken and egg problem, because we can not take advantage of an accurate projection model for an optimal features detection and description, and we rightly need good features to perform the calibration (i.e. to compute the accurate projection model of the camera). What is not this report: It does not contribute with new features algorithm. It does not compare standard features algorithms to algorithms designed for omnidirectional images (unfortunately). It has not been peer-reviewed. Discussions have been translated and enhanced but the experiments have not been run again and the report has not been updated accordingly to the evolution of the state-of-the-art (read this as a 2014 report).

  </details>



- **NeVStereo: A NeRF-Driven NVS-Stereo Architecture for High-Fidelity 3D Tasks**  
  Pengcheng Chen, Yue Hu, Wenhao Li, Nicole M Gunderson, Andrew Feng, Zhenglong Sun, Peter Beerel, Eric J Seibel  
  _2026-02-05_ · https://arxiv.org/abs/2602.05423v1  
  <details><summary>Abstract</summary>

  In modern dense 3D reconstruction, feed-forward systems (e.g., VGGT, pi3) focus on end-to-end matching and geometry prediction but do not explicitly output the novel view synthesis (NVS). Neural rendering-based approaches offer high-fidelity NVS and detailed geometry from posed images, yet they typically assume fixed camera poses and can be sensitive to pose errors. As a result, it remains non-trivial to obtain a single framework that can offer accurate poses, reliable depth, high-quality rendering, and accurate 3D surfaces from casually captured views. We present NeVStereo, a NeRF-driven NVS-stereo architecture that aims to jointly deliver camera poses, multi-view depth, novel view synthesis, and surface reconstruction from multi-view RGB-only inputs. NeVStereo combines NeRF-based NVS for stereo-friendly renderings, confidence-guided multi-view depth estimation, NeRF-coupled bundle adjustment for pose refinement, and an iterative refinement stage that updates both depth and the radiance field to improve geometric consistency. This design mitigated the common NeRF-based issues such as surface stacking, artifacts, and pose-depth coupling. Across indoor, outdoor, tabletop, and aerial benchmarks, our experiments indicate that NeVStereo achieves consistently strong zero-shot performance, with up to 36% lower depth error, 10.4% improved pose accuracy, 4.5% higher NVS fidelity, and state-of-the-art mesh quality (F1 91.93%, Chamfer 4.35 mm) compared to existing prestigious methods.

  </details>



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
  Xingyu Miao, Weiguang Zhao, Tao Lu, Linning Xu, Mulin Yu, Yang Long, Jiangmiao Pang, Junting Dong  
  _2026-02-04_ · https://arxiv.org/abs/2602.04439v2  
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


