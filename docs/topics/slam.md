# SLAM & Localization

_Updated: 2026-03-15 07:12 UTC_

Total papers shown: **4**


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


