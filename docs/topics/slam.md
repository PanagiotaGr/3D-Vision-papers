# SLAM & Localization

_Updated: 2026-03-17 07:17 UTC_

Total papers shown: **12**


---

- **Perception-Aware Autonomous Exploration in Feature-Limited Environments**  
  Moji Shi, Rajitha de Silva, Hang Yu, Riccardo Polvara, Marija Popović  
  _2026-03-16_ · https://arxiv.org/abs/2603.15605v1  
  <details><summary>Abstract</summary>

  Autonomous exploration in unknown environments typically relies on onboard state estimation for localisation and mapping. Existing exploration methods primarily maximise coverage efficiency, but often overlook that visual-inertial odometry (VIO) performance strongly depends on the availability of robust visual features. As a result, exploration policies can drive a robot into feature-sparse regions where tracking degrades, leading to odometry drift, corrupted maps, and mission failure. We propose a hierarchical perception-aware exploration framework for a stereo-equipped unmanned aerial vehicle (UAV) that explicitly couples exploration progress with feature observability. Our approach (i) associates each candidate frontier with an expected feature quality using a global feature map, and prioritises visually informative subgoals, and (ii) optimises a continuous yaw trajectory along the planned motion to maintain stable feature tracks. We evaluate our method in simulation across environments with varying texture levels and in real-world indoor experiments with largely textureless walls. Compared to baselines that ignore feature quality and/or do not optimise continuous yaw, our method maintains more reliable feature tracking, reduces odometry drift, and achieves on average 30\% higher coverage before the odometry error exceeds specified thresholds.

  </details>



- **On the Derivation of Tightly-Coupled LiDAR-Inertial Odometry with VoxelMap**  
  Zhihao Zhan  
  _2026-03-16_ · https://arxiv.org/abs/2603.15471v1  
  <details><summary>Abstract</summary>

  This note presents a concise mathematical formulation of tightly-coupled LiDAR-Inertial Odometry within an iterated error-state Kalman filter framework using a VoxelMap representation. Rather than proposing a new algorithm, it provides a clear and self-contained derivation that unifies the geometric modeling and probabilistic state estimation through consistent notation and explicit formulations. The document is intended to serve both as a technical reference and as an accessible entry point for a foundational understanding of the system architecture and estimation principles.

  </details>



- **A Novel Camera-to-Robot Calibration Method for Vision-Based Floor Measurements**  
  Jan Andre Rudolph, Dennis Haitz, Markus Ulrich  
  _2026-03-16_ · https://arxiv.org/abs/2603.15126v1  
  <details><summary>Abstract</summary>

  A novel hand-eye calibration method for ground-observing mobile robots is proposed. While cameras on mobile robots are com- mon, they are rarely used for ground-observing measurement tasks. Laser trackers are increasingly used in robotics for precise localization. A referencing plate is designed to combine the two measurement modalities of laser-tracker 3D metrology and camera- based 2D imaging. It incorporates reflector nests for pose acquisition using a laser tracker and a camera calibration target that is observed by the robot-mounted camera. The procedure comprises estimating the plate pose, the plate-camera pose, and the robot pose, followed by computing the robot-camera transformation. Experiments indicate sub-millimeter repeatability.

  </details>



- **Thermal Image Refinement with Depth Estimation using Recurrent Networks for Monocular ORB-SLAM3**  
  Hürkan Şahin, Huy Xuan Pham, Van Huyen Dang, Alper Yegenoglu, Erdal Kayacan  
  _2026-03-16_ · https://arxiv.org/abs/2603.14998v1  
  <details><summary>Abstract</summary>

  Autonomous navigation in GPS-denied and visually degraded environments remains challenging for unmanned aerial vehicles (UAVs). To this end, we investigate the use of a monocular thermal camera as a standalone sensor on a UAV platform for real-time depth estimation and simultaneous localization and mapping (SLAM). To extract depth information from thermal images, we propose a novel pipeline employing a lightweight supervised network with recurrent blocks (RBs) integrated to capture temporal dependencies, enabling more robust predictions. The network combines lightweight convolutional backbones with a thermal refinement network (T-RefNet) to refine raw thermal inputs and enhance feature visibility. The refined thermal images and predicted depth maps are integrated into ORB-SLAM3, enabling thermal-only localization. Unlike previous methods, the network is trained on a custom non-radiometric dataset, obviating the need for high-cost radiometric thermal cameras. Experimental results on datasets and UAV flights demonstrate competitive depth accuracy and robust SLAM performance under low-light conditions. On the radiometric VIVID++ (indoor-dark) dataset, our method achieves an absolute relative error of approximately 0.06, compared to baselines exceeding 0.11. In our non-radiometric indoor set, baseline errors remain above 0.24, whereas our approach remains below 0.10. Thermal-only ORB-SLAM3 maintains a mean trajectory error under 0.4 m.

  </details>



- **Voronoi-based Second-order Descriptor with Whitened Metric in LiDAR Place Recognition**  
  Jaein Kim, Hee Bin Yoo, Dong-Sig Han, Byoung-Tak Zhang  
  _2026-03-16_ · https://arxiv.org/abs/2603.14974v1  
  <details><summary>Abstract</summary>

  The pooling layer plays a vital role in aggregating local descriptors into the metrizable global descriptor in the LiDAR Place Recognition (LPR). In particular, the second-order pooling is capable of capturing higher-order interactions among local descriptors. However, its existing methods in the LPR adhere to conventional implementations and post-normalization, and incur the descriptor unsuitable for Euclidean distancing. Based on the recent interpretation that associates NetVLAD with the second-order statistics, we propose to integrate second-order pooling with the inductive bias from Voronoi cells. Our novel pooling method aggregates local descriptors to form the second-order matrix and whitens the global descriptor to implicitly measure the Mahalanobis distance while conserving the cluster property from Voronoi cells, addressing its numerical instability during learning with diverse techniques. We demonstrate its performance gains through the experiments conducted on the Oxford Robotcar and Wild-Places benchmarks and analyze the numerical effect of the proposed whitening algorithm.

  </details>



- **Intelligent Control of Differential Drive Robots Subject to Unmodeled Dynamics with EKF-based State Estimation**  
  Amos Alwala, Yuchen Hu, Gabriel da Silva Lima, Wallace Moreira Bessa  
  _2026-03-16_ · https://arxiv.org/abs/2603.14940v1  
  <details><summary>Abstract</summary>

  Reliable control and state estimation of differential drive robots (DDR) operating in dynamic and uncertain environments remains a challenge, particularly when system dynamics are partially unknown and sensor measurements are prone to degradation. This work introduces a unified control and state estimation framework that combines a Lyapunov-based nonlinear controller and Adaptive Neural Networks (ANN) with Extended Kalman Filter (EKF)-based multi-sensor fusion. The proposed controller leverages the universal approximation property of neural networks to model unknown nonlinearities in real time. An online adaptation scheme updates the weights of the radial basis function (RBF), the architecture chosen for the ANN. The learned dynamics are integrated into a feedback linearization (FBL) control law, for which theoretical guarantees of closed-loop stability and asymptotic convergence in a trajectory-tracking task are established through a Lyapunov-like stability analysis. To ensure robust state estimation, the EKF fuses inertial measurement unit (IMU) and odometry from monocular, 2D-LiDAR and wheel encoders. The fused state estimate drives the intelligent controller, ensuring consistent performance even under drift, wheel slip, sensor noise and failure. Gazebo simulations and real-world experiments are done using DDR, demonstrating the effectiveness of the approach in terms of improved velocity tracking performance with reduction in linear and angular velocity errors up to $53.91\%$ and $29.0\%$ in comparison to the baseline FBL.

  </details>



- **E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction**  
  Yunsoo Kim, Changki Sung, Dasol Hong, Hyun Myung  
  _2026-03-16_ · https://arxiv.org/abs/2603.14684v1  
  <details><summary>Abstract</summary>

  The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS). These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting. Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution. However, existing event-based NVS methods either assume known poses or rely on depth estimation models that are bounded by their initial observations, failing to generalize as the camera traverses previously unseen regions. We present E2EGS, a pose-free framework operating solely on event streams. Our key insight is that edge information provides rich structural cues essential for accurate trajectory estimation and high-quality NVS. To extract edges from noisy event streams, we exploit the distinct spatio-temporal characteristics of edges and non-edge regions. The event camera's movement induces consistent events along edges, while non-edge regions produce sparse noise. We leverage this through a patch-based temporal coherence analysis that measures local variance to extract edges while robustly suppressing noise. The extracted edges guide structure-aware Gaussian initialization and enable edge-weighted losses throughout initialization, tracking, and bundle adjustment. Extensive experiments on both synthetic and real datasets demonstrate that E2EGS achieves superior reconstruction quality and trajectory accuracy, establishing a fully pose-free paradigm for event-based 3D reconstruction.

  </details>



- **CyboRacket: A Perception-to-Action Framework for Humanoid Racket Sports**  
  Peng Ren, Chuan Qi, Haoyang Ge, Qiyuan Su, Xuguo He, Cong Huang, Pei Chi, Jiang Zhao, Kai Chen  
  _2026-03-15_ · https://arxiv.org/abs/2603.14605v1  
  <details><summary>Abstract</summary>

  Dynamic ball-interaction tasks remain challenging for robots because they require tight perception-action coupling under limited reaction time. This challenge is especially pronounced in humanoid racket sports, where successful interception depends on accurate visual tracking, trajectory prediction, coordinated stepping, and stable whole-body striking. Existing robotic racket-sport systems often rely on external motion capture for state estimation or on task-specific low-level controllers that must be retrained across tasks and platforms. We present CyboRacket, a hierarchical perception-to-action framework for humanoid racket sports that integrates onboard visual perception, physics-based trajectory prediction, and large-scale pre-trained whole-body control. The framework uses onboard cameras to track the incoming object, predicts its future trajectory, and converts the estimated interception state into target end-effector and base-motion commands for whole-body execution by SONIC on the Unitree G1 humanoid robot. We evaluate the proposed framework in a vision-based humanoid tennis-hitting task. Experimental results demonstrate real-time visual tracking, trajectory prediction, and successful striking using purely onboard sensing.

  </details>



- **Interp3R: Continuous-time 3D Geometry Estimation with Frames and Events**  
  Shuang Guo, Filbert Febryanto, Lei Sun, Guillermo Gallego  
  _2026-03-15_ · https://arxiv.org/abs/2603.14528v1  
  <details><summary>Abstract</summary>

  In recent years, 3D visual foundation models pioneered by pointmap-based approaches such as DUSt3R have attracted a lot of interest, achieving impressive accuracy and strong generalization across diverse scenes. However, these methods are inherently limited to recovering scene geometry only at the discrete time instants when images are captured, leaving the scene evolution during the blind time between consecutive frames largely unexplored. We introduce Interp3R, to the best of our knowledge the first method that enhances pointmap-based models to estimate depth and camera poses at arbitrary time instants. Interp3R leverages asynchronous event data to interpolate pointmaps produced by frame-based models, enabling temporally continuous geometric representations. Depth and camera poses are then jointly recovered by aligning the interpolated pointmaps together with those predicted by the underlying frame-based models into a consistent spatial framework. We train Interp3R exclusively on a synthetic dataset, yet demonstrate strong generalization across a wide range of synthetic and real-world benchmarks. Extensive experiments show that Interp3R outperforms by a considerable margin state-of-the-art baselines that follow a two-stage pipeline of 2D video frame interpolation followed by 3D geometry estimation.

  </details>



- **Towards Versatile Opti-Acoustic Sensor Fusion and Volumetric Mapping**  
  Ivana Collado-Gonzalez, John McConnell, Brendan Englot  
  _2026-03-15_ · https://arxiv.org/abs/2603.14457v1  
  <details><summary>Abstract</summary>

  Accurate 3D volumetric mapping is critical for autonomous underwater vehicles operating in obstacle-rich environments. Vision-based perception provides high-resolution data but fails in turbid conditions, while sonar is robust to lighting and turbidity but suffers from low resolution and elevation ambiguity. This paper presents a volumetric mapping framework that fuses a stereo sonar pair with a monocular camera to enable safe navigation under varying visibility conditions. Overlapping sonar fields of view resolve elevation ambiguity, producing fully defined 3D point clouds at each time step. The framework identifies regions of interest in camera images, associates them with corresponding sonar returns, and combines sonar range with camera-derived elevation cues to generate additional 3D points. Each 3D point is assigned a confidence value reflecting its reliability. These confidence-weighted points are fused using a Gaussian Process Volumetric Mapping framework that prioritizes the most reliable measurements. Experimental comparisons with other opti-acoustic and sonar-based approaches, along with field tests in a marina environment, demonstrate the method's effectiveness in capturing complex geometries and preserving critical information for robot navigation in both clear and turbid conditions. Our code is open-source to support community adoption.

  </details>



- **eNavi: Event-based Imitation Policies for Low-Light Indoor Mobile Robot Navigation**  
  Prithvi Jai Ramesh, Kaustav Chanda, Krishna Vinod, Joseph Raj Vishal, Yezhou Yang, Bharatesh Chakravarthi  
  _2026-03-15_ · https://arxiv.org/abs/2603.14397v1  
  <details><summary>Abstract</summary>

  Event cameras provide high dynamic range and microsecond-level temporal resolution, making them well-suited for indoor robot navigation, where conventional RGB cameras degrade under fast motion or low-light conditions. Despite advances in event-based perception spanning detection, SLAM, and pose estimation, there remains limited research on end-to-end control policies that exploit the asynchronous nature of event streams. To address this gap, we introduce a real-world indoor person-following dataset collected using a TurtleBot 2 robot, featuring synchronized raw event streams, RGB frames, and expert control actions across multiple indoor maps, trajectories under both normal and low-light conditions. We further build a multimodal data preprocessing pipeline that temporally aligns event and RGB observations while reconstructing ground-truth actions from odometry to support high-quality imitation learning. Building on this dataset, we propose a late-fusion RGB-Event navigation policy that combines dual MobileNet encoders with a transformer-based fusion module trained via behavioral cloning. A systematic evaluation of RGB-only, Event-only, and RGB-Event fusion models across 12 training variations ranging from single-path imitation to general multi-path imitation shows that policies incorporating event data, particularly the fusion model, achieve improved robustness and lower action prediction error, especially in unseen low-light conditions where RGB-only models fail. We release the dataset, synchronization pipeline, and trained models at https://eventbasedvision.github.io/eNavi/

  </details>



- **CamLit: Unified Video Diffusion with Explicit Camera and Lighting Control**  
  Zhiyi Kuang, Chengan He, Egor Zakharov, Yuxuan Xue, Shunsuke Saito, Olivier Maury, Timur Bagautdinov, Youyi Zheng, Giljoo Nam  
  _2026-03-15_ · https://arxiv.org/abs/2603.14241v1  
  <details><summary>Abstract</summary>

  We present CamLit, the first unified video diffusion model that jointly performs novel view synthesis (NVS) and relighting from a single input image. Given one reference image, a user-defined camera trajectory, and an environment map, CamLit synthesizes a video of the scene from new viewpoints under the specified illumination. Within a single generative process, our model produces temporally coherent and spatially aligned outputs, including relit novel-view frames and corresponding albedo frames, enabling high-quality control of both camera pose and lighting. Qualitative and quantitative experiments demonstrate that CamLit achieves high-fidelity outputs on par with state-of-the-art methods in both novel view synthesis and relighting, without sacrificing visual quality in either task. We show that a single generative model can effectively integrate camera and lighting control, simplifying the video generation pipeline while maintaining competitive performance and consistent realism.

  </details>


