# SLAM & Localization

_Updated: 2026-02-13 07:13 UTC_

Total papers shown: **8**


---

- **DiffPlace: Street View Generation via Place-Controllable Diffusion Model Enhancing Place Recognition**  
  Ji Li, Zhiwei Li, Shihao Li, Zhenjiang Yu, Boyang Wang, Haiou Liu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11875v1  
  <details><summary>Abstract</summary>

  Generative models have advanced significantly in realistic image synthesis, with diffusion models excelling in quality and stability. Recent multi-view diffusion models improve 3D-aware street view generation, but they struggle to produce place-aware and background-consistent urban scenes from text, BEV maps, and object bounding boxes. This limits their effectiveness in generating realistic samples for place recognition tasks. To address these challenges, we propose DiffPlace, a novel framework that introduces a place-ID controller to enable place-controllable multi-view image generation. The place-ID controller employs linear projection, perceiver transformer, and contrastive learning to map place-ID embeddings into a fixed CLIP space, allowing the model to synthesize images with consistent background buildings while flexibly modifying foreground objects and weather conditions. Extensive experiments, including quantitative comparisons and augmented training evaluations, demonstrate that DiffPlace outperforms existing methods in both generation quality and training support for visual place recognition. Our results highlight the potential of generative models in enhancing scene-level and place-aware synthesis, providing a valuable approach for improving place recognition in autonomous driving

  </details>



- **HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model**  
  Dongting Li, Xingyu Chen, Qianyang Wu, Bo Chen, Sikai Wu, Hanyu Wu, Guoyao Zhang, Liang Li, Mingliang Zhou, Diyun Xiang, et al.  
  _2026-02-12_ · https://arxiv.org/abs/2602.11758v1  
  <details><summary>Abstract</summary>

  Humanoid robots show promise for complex whole-body tasks in unstructured environments. Although Human-Object Interaction (HOI) has advanced, most methods focus on fully actuated objects rigidly coupled to the robot, ignoring underactuated objects with independent dynamics and non-holonomic constraints. These introduce control challenges from coupling forces and occlusions. We present HAIC, a unified framework for robust interaction across diverse object dynamics without external state estimation. Our key contribution is a dynamics predictor that estimates high-order object states (velocity, acceleration) solely from proprioceptive history. These predictions are projected onto static geometric priors to form a spatially grounded dynamic occupancy map, enabling the policy to infer collision boundaries and contact affordances in blind spots. We use asymmetric fine-tuning, where a world model continuously adapts to the student policy's exploration, ensuring robust state estimation under distribution shifts. Experiments on a humanoid robot show HAIC achieves high success rates in agile tasks (skateboarding, cart pushing/pulling under various loads) by proactively compensating for inertial perturbations, and also masters multi-object long-horizon tasks like carrying a box across varied terrain by predicting the dynamics of multiple objects.

  </details>



- **GSO-SLAM: Bidirectionally Coupled Gaussian Splatting and Direct Visual Odometry**  
  Jiung Yeon, Seongbo Ha, Hyeonwoo Yu  
  _2026-02-12_ · https://arxiv.org/abs/2602.11714v1  
  <details><summary>Abstract</summary>

  We propose GSO-SLAM, a real-time monocular dense SLAM system that leverages Gaussian scene representation. Unlike existing methods that couple tracking and mapping with a unified scene, incurring computational costs, or loosely integrate them with well-structured tracking frameworks, introducing redundancies, our method bidirectionally couples Visual Odometry (VO) and Gaussian Splatting (GS). Specifically, our approach formulates joint optimization within an Expectation-Maximization (EM) framework, enabling the simultaneous refinement of VO-derived semi-dense depth estimates and the GS representation without additional computational overhead. Moreover, we present Gaussian Splat Initialization, which utilizes image information, keyframe poses, and pixel associations from VO to produce close approximations to the final Gaussian scene, thereby eliminating the need for heuristic methods. Through extensive experiments, we validate the effectiveness of our method, showing that it not only operates in real time but also achieves state-of-the-art geometric/photometric fidelity of the reconstructed scene and tracking accuracy.

  </details>



- **Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation**  
  Penghui Ruan, Bojia Zi, Xianbiao Qi, Youze Huang, Rong Xiao, Pichao Wang, Jiannong Cao, Yuhui Shi  
  _2026-02-11_ · https://arxiv.org/abs/2602.11440v1  
  <details><summary>Abstract</summary>

  Object-level manipulation, relocating or reorienting objects in images or videos while preserving scene realism, is central to film post-production, AR, and creative editing. Yet existing methods struggle to jointly achieve three core goals: background preservation, geometric consistency under viewpoint shifts, and user-controllable transformations. Geometry-based approaches offer precise control but require explicit 3D reconstruction and generalize poorly; diffusion-based methods generalize better but lack fine-grained geometric control. We present Ctrl&Shift, an end-to-end diffusion framework to achieve geometry-consistent object manipulation without explicit 3D representations. Our key insight is to decompose manipulation into two stages, object removal and reference-guided inpainting under explicit camera pose control, and encode both within a unified diffusion process. To enable precise, disentangled control, we design a multi-task, multi-stage training strategy that separates background, identity, and pose signals across tasks. To improve generalization, we introduce a scalable real-world dataset construction pipeline that generates paired image and video samples with estimated relative camera poses. Extensive experiments demonstrate that Ctrl&Shift achieves state-of-the-art results in fidelity, viewpoint consistency, and controllability. To our knowledge, this is the first framework to unify fine-grained geometric control and real-world generalization for object manipulation, without relying on any explicit 3D modeling.

  </details>



- **Advancing Digital Twin Generation Through a Novel Simulation Framework and Quantitative Benchmarking**  
  Jacob Rubinstein, Avi Donaty, Don Engel  
  _2026-02-11_ · https://arxiv.org/abs/2602.11314v1  
  <details><summary>Abstract</summary>

  The generation of 3D models from real-world objects has often been accomplished through photogrammetry, i.e., by taking 2D photos from a variety of perspectives and then triangulating matched point-based features to create a textured mesh. Many design choices exist within this framework for the generation of digital twins, and differences between such approaches are largely judged qualitatively. Here, we present and test a novel pipeline for generating synthetic images from high-quality 3D models and programmatically generated camera poses. This enables a wide variety of repeatable, quantifiable experiments which can compare ground-truth knowledge of virtual camera parameters and of virtual objects against the reconstructed estimations of those perspectives and subjects.

  </details>



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


