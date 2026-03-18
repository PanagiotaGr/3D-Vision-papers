# SLAM & Localization

_Updated: 2026-03-18 07:16 UTC_

Total papers shown: **9**


---

- **WorldCam: Interactive Autoregressive 3D Gaming Worlds with Camera Pose as a Unifying Geometric Representation**  
  Jisu Nam, Yicong Hong, Chun-Hao Paul Huang, Feng Liu, JoungBin Lee, Jiyoung Kim, Siyoon Jin, Yunsung Lee, Jaeyoon Jung, Suhwan Choi, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16871v1  
  <details><summary>Abstract</summary>

  Recent advances in video diffusion transformers have enabled interactive gaming world models that allow users to explore generated environments over extended horizons. However, existing approaches struggle with precise action control and long-horizon 3D consistency. Most prior works treat user actions as abstract conditioning signals, overlooking the fundamental geometric coupling between actions and the 3D world, whereby actions induce relative camera motions that accumulate into a global camera pose within a 3D world. In this paper, we establish camera pose as a unifying geometric representation to jointly ground immediate action control and long-term 3D consistency. First, we define a physics-based continuous action space and represent user inputs in the Lie algebra to derive precise 6-DoF camera poses, which are injected into the generative model via a camera embedder to ensure accurate action alignment. Second, we use global camera poses as spatial indices to retrieve relevant past observations, enabling geometrically consistent revisiting of locations during long-horizon navigation. To support this research, we introduce a large-scale dataset comprising 3,000 minutes of authentic human gameplay annotated with camera trajectories and textual descriptions. Extensive experiments show that our approach substantially outperforms state-of-the-art interactive gaming world models in action controllability, long-horizon visual quality, and 3D spatial consistency.

  </details>



- **M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM**  
  Kerui Ren, Guanghao Li, Changjian Jiang, Yingxiang Xu, Tao Lu, Linning Xu, Junting Dong, Jiangmiao Pang, Mulin Yu, Bo Dai  
  _2026-03-17_ · https://arxiv.org/abs/2603.16844v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

  </details>



- **Reconciling distributed compliance with high-performance control in continuum soft robotics**  
  Vito Daniele Perfetta, Daniel Feliu Talegon, Ebrahim Shahabi, Cosimo Della Santina  
  _2026-03-17_ · https://arxiv.org/abs/2603.16630v1  
  <details><summary>Abstract</summary>

  High-performance closed-loop control of truly soft continuum manipulators has remained elusive. Experimental demonstrations have largely relied on sufficiently stiff, piecewise architectures in which each actuated segment behaves as a distributed yet effectively rigid element, while deformation modes beyond simple bending are suppressed. This strategy simplifies modeling and control, but sidesteps the intrinsic complexity of a fully compliant body and makes the system behave as a serial kinematic chain, much like a conventional articulated robot. An implicit conclusion has consequently emerged within the community: distributed softness and dynamic precision are incompatible. Here we show this trade-off is not fundamental. We present a highly compliant, fully continuum robotic arm - without hardware discretization or stiffness-based mode suppression - that achieves fast, precise task-space convergence under dynamic conditions. The platform integrates direct-drive actuation, a tendon routing scheme enabling coupled bending and twisting, and a structured nonlinear control architecture grounded in reduced-order strain modeling of underactuated systems. Modeling, actuation, and control are co-designed to preserve essential mechanical complexity while enabling high-bandwidth loop closure. Experiments demonstrate accurate, repeatable execution of dynamic Cartesian tasks, including fast positioning and interaction. The proposed system achieves the fastest reported task-execution speed among soft robots. At millimetric precision, execution speed increases nearly fourfold compared with prior approaches, while operating on a fully compliant continuum body. These results show that distributed compliance and high-performance dynamic control can coexist, opening a path toward truly soft manipulators approaching the operational capabilities of rigid robots without sacrificing morphological richness.

  </details>



- **Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty**  
  Mangyu Kong, Jaewon Lee, Seongwon Lee, Euntai Kim  
  _2026-03-17_ · https://arxiv.org/abs/2603.16538v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.

  </details>



- **Industrial cuVSLAM Benchmark & Integration**  
  Charbel Abi Hana, Kameel Amareen, Mohamad Mostafa, Dmitry Slepichev, Hesam Rabeti, Zheng Wang, Mihir Acharya, Anthony Rizk  
  _2026-03-17_ · https://arxiv.org/abs/2603.16240v1  
  <details><summary>Abstract</summary>

  This work presents a comprehensive benchmark evaluation of visual odometry (VO) and visual SLAM (VSLAM) systems for mobile robot navigation in real-world logistical environments. We compare multiple visual odometry approaches across controlled trajectories covering translational, rotational, and mixed motion patterns, as well as a large-scale production facility dataset spanning approximately 1.7 km. Performance is evaluated using Absolute Pose Error (APE) against ground truth from a Vicon motion capture system and a LiDAR-based SLAM reference. Our results show that a hybrid stack combining the cuVSLAM front-end with a custom SLAM back-end achieves the strongest mapping accuracy, motivating a deeper integration of cuVSLAM as the core VO component in our robotics stack. We further validate this integration by deploying and testing the cuVSLAM-based VO stack on an NVIDIA Jetson platform.

  </details>



- **PA-LVIO: Real-Time LiDAR-Visual-Inertial Odometry and Mapping with Pose-Only Bundle Adjustment**  
  Hailiang Tang, Tisheng Zhang, Liqiang Wang, Xin Ding, Man Yuan, Xiaoji Niu  
  _2026-03-17_ · https://arxiv.org/abs/2603.16228v1  
  <details><summary>Abstract</summary>

  Real-time LiDAR-visual-inertial odometry and mapping is crucial for navigation and planning tasks in intelligent transportation systems. This study presents a pose-only bundle adjustment (PA) LiDAR-visual-inertial odometry (LVIO), named PA-LVIO, to meet the urgent need for real-time navigation and mapping. The proposed PA framework for LiDAR and visual measurements is highly accurate and efficient, and it can derive reliable frame-to-frame constraints within multiple frames. A marginalization-free and frame-to-map (F2M) LiDAR measurement model is integrated into the state estimator to eliminate odometry drifts. Meanwhile, an IMU-centric online spatial-temporal calibration is employed to obtain a pixel-wise LiDAR-camera alignment. With accurate estimated odometry and extrinsics, a high-quality and RGB-rendered point-cloud map can be built. Comprehensive experiments are conducted on both public and private datasets collected by wheeled robot, unmanned aerial vehicle (UAV), and handheld devices with 28 sequences and more than 50 km trajectories. Sufficient results demonstrate that the proposed PA-LVIO yields superior or comparable performance to state-of-the-art LVIO methods, in terms of the odometry accuracy and mapping quality. Besides, PA-LVIO can run in real-time on both the desktop PC and the onboard ARM computer.

  </details>



- **CLRNet: Targetless Extrinsic Calibration for Camera, Lidar and 4D Radar Using Deep Learning**  
  Marcell Kegl, Andras Palffy, Csaba Benedek, Dariu M. Gavrila  
  _2026-03-16_ · https://arxiv.org/abs/2603.15767v1  
  <details><summary>Abstract</summary>

  In this paper, we address extrinsic calibration for camera, lidar, and 4D radar sensors. Accurate extrinsic calibration of radar remains a challenge due to the sparsity of its data. We propose CLRNet, a novel, multi-modal end-to-end deep learning (DL) calibration network capable of addressing joint camera-lidar-radar calibration, or pairwise calibration between any two of these sensors. We incorporate equirectangular projection, camera-based depth image prediction, additional radar channels, and leverage lidar with a shared feature space and loop closure loss. In extensive experiments using the View-of-Delft and Dual-Radar datasets, we demonstrate superior calibration accuracy compared to existing state-of-the-art methods, reducing both median translational and rotational calibration errors by at least 50%. Finally, we examine the domain transfer capabilities of the proposed network and baselines, when evaluating across datasets. The code will be made publicly available upon acceptance at: https://github.com/tudelft-iv.

  </details>



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


