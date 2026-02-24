# SLAM & Localization

_Updated: 2026-02-24 07:14 UTC_

Total papers shown: **4**


---

- **VGGT-MPR: VGGT-Enhanced Multimodal Place Recognition in Autonomous Driving Environments**  
  Jingyi Xu, Zhangshuo Qi, Zhongmiao Yan, Xuyu Gao, Qianyun Jiao, Songpengcheng Xia, Xieyuanli Chen, Ling Pei  
  _2026-02-23_ · https://arxiv.org/abs/2602.19735v1  
  <details><summary>Abstract</summary>

  In autonomous driving, robust place recognition is critical for global localization and loop closure detection. While inter-modality fusion of camera and LiDAR data in multimodal place recognition (MPR) has shown promise in overcoming the limitations of unimodal counterparts, existing MPR methods basically attend to hand-crafted fusion strategies and heavily parameterized backbones that require costly retraining. To address this, we propose VGGT-MPR, a multimodal place recognition framework that adopts the Visual Geometry Grounded Transformer (VGGT) as a unified geometric engine for both global retrieval and re-ranking. In the global retrieval stage, VGGT extracts geometrically-rich visual embeddings through prior depth-aware and point map supervision, and densifies sparse LiDAR point clouds with predicted depth maps to improve structural representation. This enhances the discriminative ability of fused multimodal features and produces global descriptors for fast retrieval. Beyond global retrieval, we design a training-free re-ranking mechanism that exploits VGGT's cross-view keypoint-tracking capability. By combining mask-guided keypoint extraction with confidence-aware correspondence scoring, our proposed re-ranking mechanism effectively refines retrieval results without additional parameter optimization. Extensive experiments on large-scale autonomous driving benchmarks and our self-collected data demonstrate that VGGT-MPR achieves state-of-the-art performance, exhibiting strong robustness to severe environmental changes, viewpoint shifts, and occlusions. Our code and data will be made publicly available.

  </details>



- **Denoising Particle Filters: Learning State Estimation with Single-Step Objectives**  
  Lennart Röstel, Berthold Bäuml  
  _2026-02-23_ · https://arxiv.org/abs/2602.19651v1  
  <details><summary>Abstract</summary>

  Learning-based methods commonly treat state estimation in robotics as a sequence modeling problem. While this paradigm can be effective at maximizing end-to-end performance, models are often difficult to interpret and expensive to train, since training requires unrolling sequences of predictions in time. As an alternative to end-to-end trained state estimation, we propose a novel particle filtering algorithm in which models are trained from individual state transitions, fully exploiting the Markov property in robotic systems. In this framework, measurement models are learned implicitly by minimizing a denoising score matching objective. At inference, the learned denoiser is used alongside a (learned) dynamics model to approximately solve the Bayesian filtering equation at each time step, effectively guiding predicted states toward the data manifold informed by measurements. We evaluate the proposed method on challenging robotic state estimation tasks in simulation, demonstrating competitive performance compared to tuned end-to-end trained baselines. Importantly, our method offers the desirable composability of classical filtering algorithms, allowing prior information and external sensor models to be incorporated without retraining.

  </details>



- **OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness**  
  Phuc D. A. Nguyen, Anh N. Nhu, Ming C. Lin  
  _2026-02-22_ · https://arxiv.org/abs/2602.19035v1  
  <details><summary>Abstract</summary>

  We introduce OpenVO, a novel framework for Open-world Visual Odometry (VO) with temporal awareness under limited input conditions. OpenVO effectively estimates real-world-scale ego-motion from monocular dashcam footage with varying observation rates and uncalibrated cameras, enabling robust trajectory dataset construction from rare driving events recorded in dashcam. Existing VO methods are trained on fixed observation frequency (e.g., 10Hz or 12Hz), completely overlooking temporal dynamics information. Many prior methods also require calibrated cameras with known intrinsic parameters. Consequently, their performance degrades when (1) deployed under unseen observation frequencies or (2) applied to uncalibrated cameras. These significantly limit their generalizability to many downstream tasks, such as extracting trajectories from dashcam footage. To address these challenges, OpenVO (1) explicitly encodes temporal dynamics information within a two-frame pose regression framework and (2) leverages 3D geometric priors derived from foundation models. We validate our method on three major autonomous-driving benchmarks - KITTI, nuScenes, and Argoverse 2 - achieving more than 20 performance improvement over state-of-the-art approaches. Under varying observation rate settings, our method is significantly more robust, achieving 46%-92% lower errors across all metrics. These results demonstrate the versatility of OpenVO for real-world 3D reconstruction and diverse downstream applications.

  </details>



- **Marginalized Bundle Adjustment: Multi-View Camera Pose from Monocular Depth Estimates**  
  Shengjie Zhu, Ahmed Abdelkader, Mark J. Matthews, Xiaoming Liu, Wen-Sheng Chu  
  _2026-02-21_ · https://arxiv.org/abs/2602.18906v1  
  <details><summary>Abstract</summary>

  Structure-from-Motion (SfM) is a fundamental 3D vision task for recovering camera parameters and scene geometry from multi-view images. While recent deep learning advances enable accurate Monocular Depth Estimation (MDE) from single images without depending on camera motion, integrating MDE into SfM remains a challenge. Unlike conventional triangulated sparse point clouds, MDE produces dense depth maps with significantly higher error variance. Inspired by modern RANSAC estimators, we propose Marginalized Bundle Adjustment (MBA) to mitigate MDE error variance leveraging its density. With MBA, we show that MDE depth maps are sufficiently accurate to yield SoTA or competitive results in SfM and camera relocalization tasks. Through extensive evaluations, we demonstrate consistently robust performance across varying scales, ranging from few-frame setups to large multi-view systems with thousands of images. Our method highlights the significant potential of MDE in multi-view 3D vision.

  </details>


