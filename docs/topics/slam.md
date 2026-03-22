# SLAM & Localization

_Updated: 2026-03-22 07:06 UTC_

Total papers shown: **5**


---

- **DROID-SLAM in the Wild**  
  Moyang Li, Zihan Zhu, Marc Pollefeys, Daniel Barath  
  _2026-03-19_ · https://arxiv.org/abs/2603.19076v1  
  <details><summary>Abstract</summary>

  We present a robust, real-time RGB SLAM system that handles dynamic environments by leveraging differentiable Uncertainty-aware Bundle Adjustment. Traditional SLAM methods typically assume static scenes, leading to tracking failures in the presence of motion. Recent dynamic SLAM approaches attempt to address this challenge using predefined dynamic priors or uncertainty-aware mapping, but they remain limited when confronted with unknown dynamic objects or highly cluttered scenes where geometric mapping becomes unreliable. In contrast, our method estimates per-pixel uncertainty by exploiting multi-view visual feature inconsistency, enabling robust tracking and reconstruction even in real-world environments. The proposed system achieves state-of-the-art camera poses and scene geometry in cluttered dynamic scenarios while running in real time at around 10 FPS. Code and datasets are available at https://github.com/MoyangLi00/DROID-W.git.

  </details>



- **Measuring 3D Spatial Geometric Consistency in Dynamic Generated Videos**  
  Weijia Dou, Wenzhao Zheng, Weiliang Chen, Yu Zheng, Jie Zhou, Jiwen Lu  
  _2026-03-19_ · https://arxiv.org/abs/2603.19048v1  
  <details><summary>Abstract</summary>

  Recent generative models can produce high-fidelity videos, yet they often exhibit 3D spatial geometric inconsistencies. Existing evaluation methods fail to accurately characterize these inconsistencies: fidelity-centric metrics like FVD are insensitive to geometric distortions, while consistency-focused benchmarks often penalize valid foreground dynamics. To address this gap, we introduce SGC, a metric for evaluating 3D \textbf{S}patial \textbf{G}eometric \textbf{C}onsistency in dynamically generated videos. We quantify geometric consistency by measuring the divergence among multiple camera poses estimated from distinct local regions. Our approach first separates static from dynamic regions, then partitions the static background into spatially coherent sub-regions. We predict depth for each pixel, estimate a local camera pose for each subregion, and compute the divergence among these poses to quantify geometric consistency. Experiments on real and generative videos demonstrate that SGC robustly quantifies geometric inconsistencies, effectively identifying critical failures missed by existing metrics.

  </details>



- **SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction**  
  Vsevolod Skorokhodov, Chenghao Xu, Shuo Sun, Olga Fink, Malcolm Mielle  
  _2026-03-19_ · https://arxiv.org/abs/2603.18774v1  
  <details><summary>Abstract</summary>

  Foundational feed-forward visual geometry models enable accurate and efficient camera pose estimation and scene reconstruction by learning strong scene priors from massive RGB datasets. However, their effectiveness drops when applied to mixed sensing modalities, such as RGB-thermal (RGB-T) images. We observe that while a visual geometry grounded transformer pretrained on RGB data generalizes well to thermal-only reconstruction, it struggles to align RGB and thermal modalities when processed jointly. To address this, we propose SEAR, a simple yet efficient fine-tuning strategy that adapts a pretrained geometry transformer to multimodal RGB-T inputs. Despite being trained on a relatively small RGB-T dataset, our approach significantly outperforms state-of-the-art methods for 3D reconstruction and camera pose estimation, achieving significant improvements over all metrics (e.g., over 29\% in AUC@30) and delivering higher detail and consistency between modalities with negligible overhead in inference time compared to the original pretrained model. Notably, SEAR enables reliable multimodal pose estimation and reconstruction even under challenging conditions, such as low lighting and dense smoke. We validate our architecture through extensive ablation studies, demonstrating how the model aligns both modalities. Additionally, we introduce a new dataset featuring RGB and thermal sequences captured at different times, viewpoints, and illumination conditions, providing a robust benchmark for future work in multimodal 3D scene reconstruction. Code and models are publicly available at https://www.github.com/Schindler-EPFL-Lab/SEAR.

  </details>



- **ROFT-VINS: Robust Feature Tracking-based Visual-Inertial State Estimation for Harsh Environment**  
  Sanghyun Park, Soohee Han  
  _2026-03-19_ · https://arxiv.org/abs/2603.18746v1  
  <details><summary>Abstract</summary>

  SLAM (Simultaneous Localization and Mapping) and Odometry are important systems for estimating the position of mobile devices, such as robots and cars, utilizing one or more sensors. Particularly in camera-based SLAM or Odometry, effectively tracking visual features is important as it significantly impacts system performance. In this paper, we propose a method that leverages deep learning to robustly track visual features in monocular camera images. This method operates reliably even in textureless environments and situations with rapid lighting changes. Additionally, we evaluate the performance of our proposed method by integrating it into VINS-Fusion (Monocular-Inertial), a commonly used Visual-Inertial Odometry (VIO) system.

  </details>



- **Benchmarking Visual Feature Representations for LiDAR-Inertial-Visual Odometry Under Challenging Conditions**  
  Eunseon Choi, Junwoo Hong, Daehan Lee, Sanghyun Park, Hyunyoung Jo, Sunyoung Kim, Changho Kang, Seongsam Kim, Yonghan Jung, Jungwook Park, et al.  
  _2026-03-19_ · https://arxiv.org/abs/2603.18589v1  
  <details><summary>Abstract</summary>

  Accurate localization in autonomous driving is critical for successful missions including environmental mapping and survivor searches. In visually challenging environments, including low-light conditions, overexposure, illumination changes, and high parallax, the performance of conventional visual odometry methods significantly degrade undermining robust robotic navigation. Researchers have recently proposed LiDAR-inertial-visual odometry (LIVO) frameworks, that integrate LiDAR, IMU, and camera sensors, to address these challenges. This paper extends the FAST-LIVO2-based framework by introducing a hybrid approach that integrates direct photometric methods with descriptor-based feature matching. For the descriptor-based feature matching, this work proposes pairs of ORB with the Hamming distance, SuperPoint with SuperGlue, SuperPoint with LightGlue, and XFeat with the mutual nearest neighbor. The proposed configurations are benchmarked by accuracy, computational cost, and feature tracking stability, enabling a quantitative comparison of the adaptability and applicability of visual descriptors. The experimental results reveal that the proposed hybrid approach outperforms the conventional sparse-direct method. Although the sparse-direct method often fails to converge in regions where photometric inconsistency arises due to illumination changes, the proposed approach still maintains robust performance under the same conditions. Furthermore, the hybrid approach with learning-based descriptors enables robust and reliable visual state estimation across challenging environments.

  </details>


