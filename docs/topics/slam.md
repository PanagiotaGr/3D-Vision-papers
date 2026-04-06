# SLAM & Localization

_Updated: 2026-04-06 08:00 UTC_

Total papers shown: **3**


---

- **An Open-Source LiDAR and Monocular Off-Road Autonomous Navigation Stack**  
  Rémi Marsal, Quentin Picard, Adrien Poiré, Sébastien Kerbourc'h, Thibault Toralba, Clément Yver, Alexandre Chapoutot, David Filliat  
  _2026-04-03_ · https://arxiv.org/abs/2604.03096v1  
  <details><summary>Abstract</summary>

  Off-road autonomous navigation demands reliable 3D perception for robust obstacle detection in challenging unstructured terrain. While LiDAR is accurate, it is costly and power-intensive. Monocular depth estimation using foundation models offers a lightweight alternative, but its integration into outdoor navigation stacks remains underexplored. We present an open-source off-road navigation stack supporting both LiDAR and monocular 3D perception without task-specific training. For the monocular setup, we combine zero-shot depth prediction (Depth Anything V2) with metric depth rescaling using sparse SLAM measurements (VINS-Mono). Two key enhancements improve robustness: edge-masking to reduce obstacle hallucination and temporal smoothing to mitigate the impact of SLAM instability. The resulting point cloud is used to generate a robot-centric 2.5D elevation map for costmap-based planning. Evaluated in photorealistic simulations (Isaac Sim) and real-world unstructured environments, the monocular configuration matches high-resolution LiDAR performance in most scenarios, demonstrating that foundation-model-based monocular depth estimation is a viable LiDAR alternative for robust off-road navigation. By open-sourcing the navigation stack and the simulation environment, we provide a complete pipeline for off-road navigation as well as a reproducible benchmark. Code available at https://github.com/LARIAD/Offroad-Nav.

  </details>



- **Flash-Mono: Feed-Forward Accelerated Gaussian Splatting Monocular SLAM**  
  Zicheng Zhang, Ke Wu, Xiangting Meng, Keyu Liu, Jieru Zhao, Wenchao Ding  
  _2026-04-03_ · https://arxiv.org/abs/2604.03092v1  
  <details><summary>Abstract</summary>

  Monocular 3D Gaussian Splatting SLAM suffers from critical limitations in time efficiency, geometric accuracy, and multi-view consistency. These issues stem from the time-consuming $\textit{Train-from-Scratch}$ optimization and the lack of inter-frame scale consistency from single-frame geometry priors. We contend that a feed-forward paradigm, leveraging multi-frame context to predict Gaussian attributes directly, is crucial for addressing these challenges. We present Flash-Mono, a system composed of three core modules: a feed-forward prediction frontend, a 2D Gaussian Splatting mapping backend, and an efficient hidden-state-based loop closure module. We trained a recurrent feed-forward frontend model that progressively aggregates multi-frame visual features into a hidden state via cross attention and jointly predicts camera poses and per-pixel Gaussian properties. By directly predicting Gaussian attributes, our method bypasses the burdensome per-frame optimization required in optimization-based GS-SLAM, achieving a $\textbf{10x}$ speedup while ensuring high-quality rendering. The power of our recurrent architecture extends beyond efficient prediction. The hidden states act as compact submap descriptors, facilitating efficient loop closure and global $\mathrm{Sim}(3)$ optimization to mitigate the long-standing challenge of drift. For enhanced geometric fidelity, we replace conventional 3D Gaussian ellipsoids with 2D Gaussian surfels. Extensive experiments demonstrate that Flash-Mono achieves state-of-the-art performance in both tracking and mapping quality, highlighting its potential for embodied perception and real-time reconstruction applications. Project page: https://victkk.github.io/flash-mono.

  </details>



- **An Asynchronous Two-Speed Kalman Filter for Real-Time UUV Cooperative Navigation Under Acoustic Delays**  
  Shuyue Li, Miguel López-Benítez, Eng Gee Lim, Fei Ma, Qian Dong, Mengze Cao, Limin Yu, Xiaohui Qin  
  _2026-04-03_ · https://arxiv.org/abs/2604.02878v1  
  <details><summary>Abstract</summary>

  In GNSS-denied underwater environments, individual unmanned underwater vehicles (UUVs) suffer from unbounded dead-reckoning drift, making collaborative navigation crucial for accurate state estimation. However, the severe communication delay inherent in underwater acoustic channels poses serious challenges to real-time state estimation. Traditional filters, such as Extended Kalman Filters (EKF) or Unscented Kalman Filters (UKF), usually block the main control loop while waiting for delayed data, or completely discard Out-of-Sequence Measurements (OOSM), resulting in serious drift. To address this, we propose an Asynchronous Two-Speed Kalman Filter (TSKF) enhanced by a novel projection mechanism, which we term Variational History Distillation (VHD). The proposed architecture decouples the estimation process into two parallel threads: a fast-rate thread that utilizes Gaussian Process (GP) compensated dead reckoning to guarantee high-frequency real-time control, and a slow-rate thread dedicated to processing asynchronously delayed collaborative information. By introducing a finite-length State Buffer, the algorithm applies delayed measurements (t-T) to their corresponding historical states, and utilizes a VHD-based projection to fast-forward the correction to the current time without computationally heavy recalculations. Simulation results demonstrate that the proposed TSKF maintains trajectory Root Mean Square Error (RMSE) comparable to computationally intensive batch-optimization methods under severe delays (up to 30 s). Executing in sub-millisecond time, it significantly outperforms standard EKF/UKF. The results demonstrate an effective control, communication, and computing (3C) co-design that significantly enhances the resilience of autonomous marine automation systems.

  </details>


