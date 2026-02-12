# SLAM & Localization

_Updated: 2026-02-12 07:16 UTC_

Total papers shown: **3**


---

- **Pitch Angle Control of a Magnetically Actuated Capsule Robot with Nonlinear FEA-based MPC and EKF Multisensory Fusion**  
  Chongxun Wang, Zikang Shen, Apoorav Rathore, Akanimoh Udombeh, Harrison Teng, Fangzhou Xia  
  _2026-02-11_ · https://arxiv.org/abs/2602.10610v1  
  <details><summary>Abstract</summary>

  Magnetically actuated capsule robots promise minimally invasive diagnosis and therapy in the gastrointestinal (GI) tract, but existing systems largely neglect control of capsule pitch, a degree of freedom critical for contact-rich interaction with inclined gastric walls. This paper presents a nonlinear, model-based framework for magnetic pitch control of an ingestible capsule robot actuated by a four-coil electromagnetic array. Angle-dependent magnetic forces and torques acting on embedded permanent magnets are characterized using three-dimensional finite-element simulations and embedded as lookup tables in a control-oriented rigid-body pitching model with rolling contact and actuator dynamics. A constrained model predictive controller (MPC) is designed to regulate pitch while respecting hardware-imposed current and slew-rate limits. Experiments on a compliant stomach-inspired surface demonstrate robust pitch reorientation from both horizontal and upright configurations, achieving about three to five times faster settling and reduced oscillatory motion than on-off control. Furthermore, an extended Kalman filter (EKF) fusing inertial sensing with intermittent visual measurements enables stable closed-loop control when the camera update rate is reduced from 30 Hz to 1 Hz, emulating clinically realistic imaging constraints. These results establish finite-element-informed MPC with sensor fusion as a scalable strategy for pitch regulation, controlled docking, and future multi-degree-of-freedom capsule locomotion.

  </details>



- **ReSPEC: A Framework for Online Multispectral Sensor Reconfiguration in Dynamic Environments**  
  Yanchen Liu, Yuang Fan, Minghui Zhao, Xiaofan Jiang  
  _2026-02-11_ · https://arxiv.org/abs/2602.10547v1  
  <details><summary>Abstract</summary>

  Multi-sensor fusion is central to robust robotic perception, yet most existing systems operate under static sensor configurations, collecting all modalities at fixed rates and fidelity regardless of their situational utility. This rigidity wastes bandwidth, computation, and energy, and prevents systems from prioritizing sensors under challenging conditions such as poor lighting or occlusion. Recent advances in reinforcement learning (RL) and modality-aware fusion suggest the potential for adaptive perception, but prior efforts have largely focused on re-weighting features at inference time, ignoring the physical cost of sensor data collection. We introduce a framework that unifies sensing, learning, and actuation into a closed reconfiguration loop. A task-specific detection backbone extracts multispectral features (e.g. RGB, IR, mmWave, depth) and produces quantitative contribution scores for each modality. These scores are passed to an RL agent, which dynamically adjusts sensor configurations, including sampling frequency, resolution, sensing range, and etc., in real time. Less informative sensors are down-sampled or deactivated, while critical sensors are sampled at higher fidelity as environmental conditions evolve. We implement and evaluate this framework on a mobile rover, showing that adaptive control reduces GPU load by 29.3\% with only a 5.3\% accuracy drop compared to a heuristic baseline. These results highlight the potential of resource-aware adaptive sensing for embedded robotic platforms.

  </details>



- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>


