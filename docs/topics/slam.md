# SLAM & Localization

_Updated: 2026-03-31 07:42 UTC_

Total papers shown: **5**


---

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



- **GS3LAM: Gaussian Semantic Splatting SLAM**  
  Linfei Li, Lin Zhang, Zhong Wang, Ying Shen  
  _2026-03-29_ · https://arxiv.org/abs/2603.27781v1  
  <details><summary>Abstract</summary>

  Recently, the multi-modal fusion of RGB, depth, and semantics has shown great potential in dense Simultaneous Localization and Mapping (SLAM). However, a prerequisite for generating consistent semantic maps is the availability of dense, efficient, and scalable scene representations. Existing semantic SLAM systems based on explicit representations are often limited by resolution and an inability to predict unknown areas. Conversely, implicit representations typically rely on time-consuming ray tracing, failing to meet real-time requirements. Fortunately, 3D Gaussian Splatting (3DGS) has emerged as a promising representation that combines the efficiency of point-based methods with the continuity of geometric structures. To this end, we propose GS3LAM, a Gaussian Semantic Splatting SLAM framework that processes multimodal data to render consistent, dense semantic maps in real-time. GS3LAM models the scene as a Semantic Gaussian Field (SG-Field) and jointly optimizes camera poses and the field via multimodal error constraints. Furthermore, a Depth-adaptive Scale Regularization (DSR) scheme is introduced to resolve misalignments between scale-invariant Gaussians and geometric surfaces. To mitigate catastrophic forgetting, we propose a Random Sampling-based Keyframe Mapping (RSKM) strategy, which demonstrates superior performance over common local covisibility optimization methods. Extensive experiments on benchmark datasets show that GS3LAM achieves increased tracking robustness, superior rendering quality, and enhanced semantic precision compared to state-of-the-art methods. Source code is available at https://github.com/lif314/GS3LAM.

  </details>



- **RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization**  
  Junwei Zheng, Ruize Dai, Ruiping Liu, Zichao Zeng, Yufan Chen, Fangjinhua Wang, Kunyu Peng, Kailun Yang, Jiaming Zhang, Rainer Stiefelhagen  
  _2026-03-29_ · https://arxiv.org/abs/2603.27758v1  
  <details><summary>Abstract</summary>

  Metric Cross-View Geo-Localization (MCVGL) aims to estimate the 3-DoF camera pose (position and heading) by matching ground and satellite images. In this work, instead of pinhole and satellite images, we study robust MCVGL using holistic panoramas and OpenStreetMap (OSM). To this end, we establish a large-scale MCVGL benchmark dataset, CV-RHO, with over 2.7M images under different weather and lighting conditions, as well as sensor noise. Furthermore, we propose a model termed RHO with a two-branch Pin-Pan architecture for accurate visual localization. A Split-Undistort-Merge (SUM) module is introduced to address the panoramic distortion, and a Position-Orientation Fusion (POF) mechanism is designed to enhance the localization accuracy. Extensive experiments prove the value of our CV-RHO dataset and the effectiveness of the RHO model, with a significant performance gain up to 20% compared with the state-of-the-art baselines. Project page: https://github.com/InSAI-Lab/RHO.

  </details>



- **LiDAR for Crowd Management: Applications, Benefits, and Future Directions**  
  Abdullah Khanfor, Chaima Zaghouani, Hakim Ghazzai, Ahmad Alsharoa, Gianluca Setti  
  _2026-03-29_ · https://arxiv.org/abs/2603.27663v1  
  <details><summary>Abstract</summary>

  Light Detection and Ranging (LiDAR) technology offers significant advantages for effective crowd management. This article presents LiDAR technology and highlights its primary advantages over other monitoring technologies, including enhanced privacy, performance in various weather conditions, and precise 3D mapping. We present a general taxonomy of four key tasks in crowd management: crowd detection, counting, tracking, and behavior classification, with illustrative examples of LiDAR applications for each task. We identify challenges and open research directions, including the scarcity of dedicated datasets, sensor fusion requirements, artificial intelligence integration, and processing needs for LiDAR point clouds. This article offers actionable insights for developing crowd management solutions tailored to public safety applications.

  </details>


