# SLAM & Localization

_Updated: 2026-03-21 07:01 UTC_

Total papers shown: **10**


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



- **FILT3R: Latent State Adaptive Kalman Filter for Streaming 3D Reconstruction**  
  Seonghyun Jin, Jong Chul Ye  
  _2026-03-19_ · https://arxiv.org/abs/2603.18493v1  
  <details><summary>Abstract</summary>

  Streaming 3D reconstruction maintains a persistent latent state that is updated online from incoming frames, enabling constant-memory inference. A key failure mode is the state update rule: aggressive overwrites forget useful history, while conservative updates fail to track new evidence, and both behaviors become unstable beyond the training horizon. To address this challenge, we propose FILT3R, a training-free latent filtering layer that casts recurrent state updates as stochastic state estimation in token space. FILT3R maintains a per-token variance and computes a Kalman-style gain that adaptively balances memory retention against new observations. Process noise -- governing how much the latent state is expected to change between frames -- is estimated online from EMA-normalized temporal drift of candidate tokens. Using extensive experiments, we demonstrate that FILT3R yields an interpretable, plug-in update rule that generalizes common overwrite and gating policies as special cases. Specifically, we show that gains shrink in stable regimes as uncertainty contracts with accumulated evidence, and rise when genuine scene change increases process uncertainty, improving long-horizon stability for depth, pose, and 3D reconstruction, compared to the existing methods. Code will be released at https://github.com/jinotter3/FILT3R.

  </details>



- **Proprioceptive-only State Estimation for Legged Robots with Set-Coverage Measurements of Learned Dynamics**  
  Abhijeet M. Kulkarni, Ioannis Poulakakis, Guoquan Huang  
  _2026-03-18_ · https://arxiv.org/abs/2603.18308v1  
  <details><summary>Abstract</summary>

  Proprioceptive-only state estimation is attractive for legged robots since it is computationally cheaper and is unaffected by perceptually degraded conditions. The history of joint-level measurements contains rich information that can be used to infer the dynamics of the system and subsequently produce navigational measurements. Recent approaches produce these estimates with learned measurement models and fuse with IMU data, under a Gaussian noise assumption. However, this assumption can easily break down with limited training data and render the estimates inconsistent and potentially divergent. In this work, we propose a proprioceptive-only state estimation framework for legged robots that characterizes the measurement noise using set-coverage statements that do not assume any distribution. We develop a practical and computationally inexpensive method to use these set-coverage measurements with a Gaussian filter in a systematic way. We validate the approach in both simulation and two real-world quadrupedal datasets. Comparison with the Gaussian baselines shows that our proposed method remains consistent and is not prone to drift under real noise scenarios.

  </details>



- **Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting**  
  Guillem Casadesus Vila, Adam Dai, Grace Gao  
  _2026-03-18_ · https://arxiv.org/abs/2603.18218v1  
  <details><summary>Abstract</summary>

  Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.

  </details>



- **Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models**  
  Kevin Qu, Haozhe Qi, Mihai Dusmanu, Mahdi Rad, Rui Wang, Marc Pollefeys  
  _2026-03-18_ · https://arxiv.org/abs/2603.18002v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have made impressive progress in connecting vision and language, but they still struggle with spatial understanding and viewpoint-aware reasoning. Recent efforts aim to augment the input representations with geometric cues rather than explicitly teaching models to reason in 3D space. We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input. Inspired by human spatial cognition, Loc3R-VLM relies on two joint objectives: global layout reconstruction to build a holistic representation of the scene structure, and explicit situation modeling to anchor egocentric perspective. These objectives provide direct spatial supervision that grounds both perception and language in a 3D context. To ensure geometric consistency and metric-scale alignment, we leverage lightweight camera pose priors extracted from a pre-trained 3D foundation model. Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding. Project page: https://kevinqu7.github.io/loc3r-vlm

  </details>



- **PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery**  
  Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, Yujiao Shi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17571v1  
  <details><summary>Abstract</summary>

  Panoramic imagery offers a full 360° field of view and is increasingly common in consumer devices. However, it introduces non-pinhole distortions that challenge joint pose estimation and 3D reconstruction. Existing feed-forward models, built for perspective cameras, generalize poorly to this setting. We propose PanoVGGT, a permutation-equivariant Transformer framework that jointly predicts camera poses, depth maps, and 3D point clouds from one or multiple panoramas in a single forward pass. The model incorporates spherical-aware positional embeddings and a panorama-specific three-axis SO(3) rotation augmentation, enabling effective geometric reasoning in the spherical domain. To resolve inherent global-frame ambiguity, we further introduce a stochastic anchoring strategy during training. In addition, we contribute PanoCity, a large-scale outdoor panoramic dataset with dense depth and 6-DoF pose annotations. Extensive experiments on PanoCity and standard benchmarks demonstrate that PanoVGGT achieves competitive accuracy, strong robustness, and improved cross-domain generalization. Code and dataset will be released.

  </details>


