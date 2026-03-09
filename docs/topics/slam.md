# SLAM & Localization

_Updated: 2026-03-09 07:16 UTC_

Total papers shown: **2**


---

- **CFEAR-Teach-and-Repeat: Fast and Accurate Radar-only Localization**  
  Maximilian Hilger, Daniel Adolfsson, Ralf Becker, Henrik Andreasson, Achim J. Lilienthal  
  _2026-03-06_ · https://arxiv.org/abs/2603.06501v1  
  <details><summary>Abstract</summary>

  Reliable localization in prior maps is essential for autonomous navigation, particularly under adverse weather, where optical sensors may fail. We present CFEAR-TR, a teach-and-repeat localization pipeline using a single spinning radar, which is designed for easily deployable, lightweight, and robust navigation in adverse conditions. Our method localizes by jointly aligning live scans to both stored scans from the teach mapping pass, and to a sliding window of recent live keyframes. This ensures accurate and robust pose estimation across different seasons and weather phenomena. Radar scans are represented using a sparse set of oriented surface points, computed from Doppler-compensated measurements. The map is stored in a pose graph that is traversed during localization. Experiments on the held-out test sequences from the Boreas dataset show that CFEAR-TR can localize with an accuracy as low as 0.117 m and 0.096°, corresponding to improvements of up to 63% over the previous state of the art, while running efficiently at 29 Hz. These results substantially narrow the gap to lidar-level localization, particularly in heading estimation. We make the C++ implementation of our work available to the community.

  </details>



- **KISS-IMU: Self-supervised Inertial Odometry with Motion-balanced Learning and Uncertainty-aware Inference**  
  Jiwon Choi, Hogyun Kim, Geonmo Yang, Juhui Lee, Younggun Cho  
  _2026-03-06_ · https://arxiv.org/abs/2603.06205v1  
  <details><summary>Abstract</summary>

  Inertial measurement units (IMUs), which provide high-frequency linear acceleration and angular velocity measurements, serve as fundamental sensing modalities in robotic systems. Recent advances in deep neural networks have led to remarkable progress in inertial odometry. However, the heavy reliance on ground truth data during training fundamentally limits scalability and generalization to unseen and diverse environments. We propose KISS-IMU, a novel self-supervised inertial odometry framework that eliminates ground truth dependency by leveraging simple LiDAR-based ICP registration and pose graph optimization as a supervisory signal. Our approach embodies two key principles: keeping the IMU stable through motion-aware balanced training and keeping the IMU strong through uncertainty-driven adaptive weighting during inference. To evaluate performance across diverse motion patterns and scenarios, we conducted comprehensive experiments on various real-world platforms, including quadruped robots. Importantly, we train only the IMU network in a self-supervised manner, with LiDAR serving solely as a lightweight supervisory signal rather than requiring additional learnable processes. This design enables the framework to ensure robustness without relying on joint multi-modal learning or ground truth supervision. The supplementary materials are available at https://sparolab.github.io/research/kiss_imu.

  </details>


