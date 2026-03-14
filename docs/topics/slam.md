# SLAM & Localization

_Updated: 2026-03-14 07:04 UTC_

Total papers shown: **9**


---

- **Decentralized Cooperative Localization for Multi-Robot Systems with Asynchronous Sensor Fusion**  
  Nivand Khosravi, Niusha Khosravi, Mohammad Bozorg, Masoud S. Bahraini  
  _2026-03-12_ · https://arxiv.org/abs/2603.12075v1  
  <details><summary>Abstract</summary>

  Decentralized cooperative localization (DCL) is a promising approach for nonholonomic mobile robots operating in GPS-denied environments with limited communication infrastructure. This paper presents a DCL framework in which each robot performs localization locally using an Extended Kalman Filter, while sharing measurement information during update stages only when communication links are available and companion robots are successfully detected by LiDAR. The framework preserves cross-correlation consistency among robot state estimates while handling asynchronous sensor data with heterogeneous sampling rates and accommodating accelerations during dynamic maneuvers. Unlike methods that require pre-aligned coordinate systems, the proposed approach allows robots to initialize with arbitrary reference-frame orientations and achieves automatic alignment through transformation matrices in both the prediction and update stages. To improve robustness in feature-sparse environments, we introduce a dual-landmark evaluation framework that exploits both static environmental features and mobile robots as dynamic landmarks. The proposed framework enables reliable detection and feature extraction during sharp turns, while prediction accuracy is improved through information sharing from mutual observations. Experimental results in both Gazebo simulation and real-world basement environments show that DCL outperforms centralized cooperative localization (CCL), achieving a 34% reduction in RMSE, while the dual-landmark variant yields an improvement of 56%. These results demonstrate the applicability of DCL to challenging domains such as enclosed spaces, underwater environments, and feature-sparse terrains where conventional localization methods are ineffective.

  </details>



- **Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos**  
  Shuo Sun, Unal Artan, Malcolm Mielle, Achim J. Lilienthaland, Martin Magnusson  
  _2026-03-12_ · https://arxiv.org/abs/2603.12064v1  
  <details><summary>Abstract</summary>

  We address the challenging problem of dense dynamic scene reconstruction and camera pose estimation from multiple freely moving cameras -- a setting that arises naturally when multiple observers capture a shared event. Prior approaches either handle only single-camera input or require rigidly mounted, pre-calibrated camera rigs, limiting their practical applicability. We propose a two-stage optimization framework that decouples the task into robust camera tracking and dense depth refinement. In the first stage, we extend single-camera visual SLAM to the multi-camera setting by constructing a spatiotemporal connection graph that exploits both intra-camera temporal continuity and inter-camera spatial overlap, enabling consistent scale and robust tracking. To ensure robustness under limited overlap, we introduce a wide-baseline initialization strategy using feed-forward reconstruction models. In the second stage, we refine depth and camera poses by optimizing dense inter- and intra-camera consistency using wide-baseline optical flow. Additionally, we introduce MultiCamRobolab, a new real-world dataset with ground-truth poses from a motion capture system. Finally, we demonstrate that our method significantly outperforms state-of-the-art feed-forward models on both synthetic and real-world benchmarks, while requiring less memory.

  </details>



- **Pano360: Perspective to Panoramic Vision with Geometric Consistency**  
  Zhengdong Zhu, Weiyi Xue, Zuyuan Yang, Wenlve Zhou, Zhiheng Zhou  
  _2026-03-12_ · https://arxiv.org/abs/2603.12013v1  
  <details><summary>Abstract</summary>

  Prior panorama stitching approaches heavily rely on pairwise feature correspondences and are unable to leverage geometric consistency across multiple views. This leads to severe distortion and misalignment, especially in challenging scenes with weak textures, large parallax, and repetitive patterns. Given that multi-view geometric correspondences can be directly constructed in 3D space, making them more accurate and globally consistent, we extend the 2D alignment task to the 3D photogrammetric space. We adopt a novel transformer-based architecture to achieve 3D awareness and aggregate global information across all views. It directly utilizes camera poses to guide image warping for global alignment in 3D space and employs a multi-feature joint optimization strategy to compute the seams. Additionally, to establish an evaluation benchmark and train our network, we constructed a large-scale dataset of real-world scenes. Extensive experiments show that our method significantly outperforms existing alternatives in alignment accuracy and perceptual quality.

  </details>



- **A Hybrid Neural-Assisted Unscented Kalman Filter for Unmanned Ground Vehicle Navigation**  
  Gal Versano, Itzik Klein  
  _2026-03-12_ · https://arxiv.org/abs/2603.11649v1  
  <details><summary>Abstract</summary>

  Modern autonomous navigation for unmanned ground vehicles relies on different estimators to fuse inertial sensors and GNSS measurements. However, the constant noise covariance matrices often struggle to account for dynamic real-world conditions. In this work we propose a hybrid estimation framework that bridges classical state estimation foundations with modern deep learning approaches. Instead of altering the fundamental unscented Kalman filter equations, a dedicated deep neural network is developed to predict the process and measurement noise uncertainty directly from raw inertial and GNSS measurements. We present a sim2real approach, with training performed only on simulative data. In this manner, we offer perfect ground truth data and relieves the burden of extensive data recordings. To evaluate our proposed approach and examine its generalization capabilities, we employed a 160-minutes test set from three datasets each with different types of vehicles (off-road vehicle, passenger car, and mobile robot), inertial sensors, road surface, and environmental conditions. We demonstrate across the three datasets a position improvement of $12.7\%$ compared to the adaptive model-based approach. Thus, offering a scalable and a more robust solution for unmanned ground vehicles navigation tasks.

  </details>



- **D-SLAMSpoof: An Environment-Agnostic LiDAR Spoofing Attack using Dynamic Point Cloud Injection**  
  Rokuto Nagata, Kenji Koide, Kazuma Ikeda, Ozora Sako, Kentaro Yoshioka  
  _2026-03-11_ · https://arxiv.org/abs/2603.11365v1  
  <details><summary>Abstract</summary>

  In this work, we introduce Dynamic SLAMSpoof (D-SLAMSpoof), a novel attack that compromises LiDAR SLAM even in feature-rich environments. The attack leverages LiDAR spoofing, which injects spurious measurements into LiDAR scans through external laser interference. By designing both spatial injection shapes and temporally coordinated dynamic injection patterns guided by scan-matching principles, D-SLAMSpoof significantly improves attack success rates in real-world, feature-rich environments such as urban areas and indoor spaces, where conventional LiDAR spoofing methods often fail. Furthermore, we propose a practical defense method, ISD-SLAM, that relies solely on inertial dead reckoning signals commonly available in autonomous systems. We demonstrate that ISD-SLAM accurately detects LiDAR spoofing attacks, including D-SLAMSpoof, and effectively mitigates the resulting position drift. Our findings expose inherent vulnerabilities in LiDAR-based SLAM and introduce the first practical defense against LiDAR-based SLAM spoofing using only standard onboard sensors, providing critical insights for improving the security and reliability of autonomous systems.

  </details>



- **MirrorDrift: Actuated Mirror-Based Attacks on LiDAR SLAM**  
  Rokuto Nagata, Kenji Koide, Kazuma Ikeda, Ozora Sako, Shion Horie, Kentaro Yoshioka  
  _2026-03-11_ · https://arxiv.org/abs/2603.11364v1  
  <details><summary>Abstract</summary>

  LiDAR SLAM provides high-accuracy localization but is fragile to point-cloud corruption because scan matching assumes geometric consistency. Prior physical attacks on LiDAR SLAM largely rely on LiDAR spoofing via external signal injection, which requires sensor-specific timing knowledge and is increasingly mitigated by modern defense mechanisms such as timing obfuscation and injection rejection. In this work, we show that specular reflection offers an injection-free alternative and demonstrate an attack, MirrorDrift, that uses an actuated planar mirror to cause ghost points in LiDAR scans and systematically bias scan-matching correspondences. MirrorDrift optimizes mirror placement, alignment, and actuation. In simulation, it increases the average pose error (APE) by 6.1x over random placement, degrading three SLAM systems to 2.29-3.31 m mean APE. In real-world experiments on a modern LiDAR with state-of-the-art interference mitigation, it induces localization errors of up to 6.03 m. To the best of our knowledge, this is the first successful SLAM-targeted attack against production-grade secure LiDARs.

  </details>



- **InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction**  
  Dingqiang Ye, Jiacong Xu, Jianglu Ping, Yuxiang Guo, Chao Fan, Vishal M. Patel  
  _2026-03-11_ · https://arxiv.org/abs/2603.11298v1  
  <details><summary>Abstract</summary>

  High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.

  </details>



- **GGPT: Geometry Grounded Point Transformer**  
  Yutong Chen, Yiming Wang, Xucong Zhang, Sergey Prokudin, Siyu Tang  
  _2026-03-11_ · https://arxiv.org/abs/2603.11174v1  
  <details><summary>Abstract</summary>

  Recent feed-forward networks have achieved remarkable progress in sparse-view 3D reconstruction by predicting dense point maps directly from RGB images. However, they often suffer from geometric inconsistencies and limited fine-grained accuracy due to the absence of explicit multi-view constraints. We introduce the Geometry-Grounded Point Transformer (GGPT), a framework that augments feed-forward reconstruction with reliable sparse geometric guidance. We first propose an improved Structure-from-Motion pipeline based on dense feature matching and lightweight geometric optimisation to efficiently estimate accurate camera poses and partial 3D point clouds from sparse input views. Building on this foundation, we propose a geometry-guided 3D point transformer that refines dense point maps under explicit partial-geometry supervision using an optimised guidance encoding. Extensive experiments demonstrate that our method provides a principled mechanism for integrating geometric priors with dense feed-forward predictions, producing reconstructions that are both geometrically consistent and spatially complete, recovering fine structures and filling gaps in textureless areas. Trained solely on ScanNet++ with VGGT predictions, GGPT generalises across architectures and datasets, substantially outperforming state-of-the-art feed-forward 3D reconstruction models in both in-domain and out-of-domain settings.

  </details>



- **Semantic Landmark Particle Filter for Robot Localisation in Vineyards**  
  Rajitha de Silva, Jonathan Cox, James R. Heselden, Marija Popović, Cesar Cadena, Riccardo Polvara  
  _2026-03-11_ · https://arxiv.org/abs/2603.10847v1  
  <details><summary>Abstract</summary>

  Reliable localisation in vineyards is hindered by row-level perceptual aliasing: parallel crop rows produce nearly identical LiDAR observations, causing geometry-only and vision-based SLAM systems to converge towards incorrect corridors, particularly during headland transitions. We present a Semantic Landmark Particle Filter (SLPF) that integrates trunk and pole landmark detections with 2D LiDAR within a probabilistic localisation framework. Detected trunks are converted into semantic walls, forming structural row boundaries embedded in the measurement model to improve discrimination between adjacent rows. GNSS is incorporated as a lightweight prior that stabilises localisation when semantic observations are sparse. Field experiments in a 10-row vineyard demonstrate consistent improvements over geometry-only (AMCL), vision-based (RTAB-Map), and GNSS baselines. Compared to AMCL, SLPF reduces Absolute Pose Error by 22% and 65% across two traversal directions; relative to a NoisyGNSS baseline, APE decreases by 65% and 61%. Row correctness improves from 0.67 to 0.73, while mean cross-track error decreases from 1.40 m to 1.26 m. These results show that embedding row-level structural semantics within the measurement model enables robust localisation in highly repetitive outdoor agricultural environments.

  </details>


