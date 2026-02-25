# SLAM & Localization

_Updated: 2026-02-25 07:15 UTC_

Total papers shown: **8**


---

- **LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments**  
  Zeyu Jiang, Kuan Xu, Changhao Chen  
  _2026-02-24_ · https://arxiv.org/abs/2602.20925v1  
  <details><summary>Abstract</summary>

  Thermal cameras offer strong potential for robot perception under challenging illumination and weather conditions. However, thermal Simultaneous Localization and Mapping (SLAM) remains difficult due to unreliable feature extraction, unstable motion tracking, and inconsistent global pose and map construction, particularly in dynamic large-scale outdoor environments. To address these challenges, we propose LST-SLAM, a novel large-scale stereo thermal SLAM system that achieves robust performance in complex, dynamic scenes. Our approach combines self-supervised thermal feature learning, stereo dual-level motion tracking, and geometric pose optimization. We also introduce a semantic-geometric hybrid constraint that suppresses potentially dynamic features lacking strong inter-frame geometric consistency. Furthermore, we develop an online incremental bag-of-words model for loop closure detection, coupled with global pose optimization to mitigate accumulated drift. Extensive experiments on kilometer-scale dynamic thermal datasets show that LST-SLAM significantly outperforms recent representative SLAM systems, including AirSLAM and DROID-SLAM, in both robustness and accuracy.

  </details>



- **RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction**  
  Yangfan Zhao, Hanwei Zhang, Ke Huang, Qiufeng Wang, Zhenzhou Shao, Dengyu Wu  
  _2026-02-24_ · https://arxiv.org/abs/2602.20807v1  
  <details><summary>Abstract</summary>

  Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io

  </details>



- **From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection**  
  Yepeng Liu, Hao Li, Liwen Yang, Fangzhen Li, Xudi Ge, Yuliang Gu, kuang Gao, Bing Wang, Guang Chen, Hangjun Ye, et al.  
  _2026-02-24_ · https://arxiv.org/abs/2602.20630v1  
  <details><summary>Abstract</summary>

  Keypoint-based matching is a fundamental component of modern 3D vision systems, such as Structure-from-Motion (SfM) and SLAM. Most existing learning-based methods are trained on image pairs, a paradigm that fails to explicitly optimize for the long-term trackability of keypoints across sequences under challenging viewpoint and illumination changes. In this paper, we reframe keypoint detection as a sequential decision-making problem. We introduce TraqPoint, a novel, end-to-end Reinforcement Learning (RL) framework designed to optimize the \textbf{Tra}ck-\textbf{q}uality (Traq) of keypoints directly on image sequences. Our core innovation is a track-aware reward mechanism that jointly encourages the consistency and distinctiveness of keypoints across multiple views, guided by a policy gradient method. Extensive evaluations on sparse matching benchmarks, including relative pose estimation and 3D reconstruction, demonstrate that TraqPoint significantly outperforms some state-of-the-art (SOTA) keypoint detection and description methods.

  </details>



- **Object-Scene-Camera Decomposition and Recomposition for Data-Efficient Monocular 3D Object Detection**  
  Zhaonian Kuang, Rui Ding, Meng Yang, Xinhu Zheng, Gang Hua  
  _2026-02-24_ · https://arxiv.org/abs/2602.20627v1  
  <details><summary>Abstract</summary>

  Monocular 3D object detection (M3OD) is intrinsically ill-posed, hence training a high-performance deep learning based M3OD model requires a humongous amount of labeled data with complicated visual variation from diverse scenes, variety of objects and camera poses.However, we observe that, due to strong human bias, the three independent entities, i.e., object, scene, and camera pose, are always tightly entangled when an image is captured to construct training data. More specifically, specific 3D objects are always captured in particular scenes with fixed camera poses, and hence lacks necessary diversity. Such tight entanglement induces the challenging issues of insufficient utilization and overfitting to uniform training data. To mitigate this, we propose an online object-scene-camera decomposition and recomposition data manipulation scheme to more efficiently exploit the training data. We first fully decompose training images into textured 3D object point models and background scenes in an efficient computation and storage manner. We then continuously recompose new training images in each epoch by inserting the 3D objects into the freespace of the background scenes, and rendering them with perturbed camera poses from textured 3D point representation. In this way, the refreshed training data in all epochs can cover the full spectrum of independent object, scene, and camera pose combinations. This scheme can serve as a plug-and-play component to boost M3OD models, working flexibly with both fully and sparsely supervised settings. In the sparsely-supervised setting, objects closest to the ego-camera for all instances are sparsely annotated. We then can flexibly increase the annotated objects to control annotation cost. For validation, our method is widely applied to five representative M3OD models and evaluated on both the KITTI and the more complicated Waymo datasets.

  </details>



- **Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques**  
  Christos Maikos, Georgios Angelidis, Georgios Th. Papadopoulos  
  _2026-02-23_ · https://arxiv.org/abs/2602.20342v1  
  <details><summary>Abstract</summary>

  In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.

  </details>



- **GSNR: Graph Smooth Null-Space Representation for Inverse Problems**  
  Romario Gualdrón-Hurtado, Roman Jacome, Rafael S. Suarez, Henry Arguello  
  _2026-02-23_ · https://arxiv.org/abs/2602.20328v1  
  <details><summary>Abstract</summary>

  Inverse problems in imaging are ill-posed, leading to infinitely many solutions consistent with the measurements due to the non-trivial null-space of the sensing matrix. Common image priors promote solutions on the general image manifold, such as sparsity, smoothness, or score function. However, as these priors do not constrain the null-space component, they can bias the reconstruction. Thus, we aim to incorporate meaningful null-space information in the reconstruction framework. Inspired by smooth image representation on graphs, we propose Graph-Smooth Null-Space Representation (GSNR), a mechanism that imposes structure only into the invisible component. Particularly, given a graph Laplacian, we construct a null-restricted Laplacian that encodes similarity between neighboring pixels in the null-space signal, and we design a low-dimensional projection matrix from the $p$-smoothest spectral graph modes (lowest graph frequencies). This approach has strong theoretical and practical implications: i) improved convergence via a null-only graph regularizer, ii) better coverage, how much null-space variance is captured by $p$ modes, and iii) high predictability, how well these modes can be inferred from the measurements. GSNR is incorporated into well-known inverse problem solvers, e.g., PnP, DIP, and diffusion solvers, in four scenarios: image deblurring, compressed sensing, demosaicing, and image super-resolution, providing consistent improvement of up to 4.3 dB over baseline formulations and up to 1 dB compared with end-to-end learned models in terms of PSNR.

  </details>



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


