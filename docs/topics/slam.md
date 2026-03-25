# SLAM & Localization

_Updated: 2026-03-25 07:16 UTC_

Total papers shown: **12**


---

- **WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG**  
  Zhen Li, Zian Meng, Shuwei Shi, Wenshuo Peng, Yuwei Wu, Bo Zheng, Chuanhao Li, Kaipeng Zhang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23497v1  
  <details><summary>Abstract</summary>

  Dynamical systems theory and reinforcement learning view world evolution as latent-state dynamics driven by actions, with visual observations providing partial information about the state. Recent video world models attempt to learn this action-conditioned dynamics from data. However, existing datasets rarely match the requirement: they typically lack diverse and semantically meaningful action spaces, and actions are directly tied to visual observations rather than mediated by underlying states. As a result, actions are often entangled with pixel-level changes, making it difficult for models to learn structured world dynamics and maintain consistent evolution over long horizons. In this paper, we propose WildWorld, a large-scale action-conditioned world modeling dataset with explicit state annotations, automatically collected from a photorealistic AAA action role-playing game (Monster Hunter: Wilds). WildWorld contains over 108 million frames and features more than 450 actions, including movement, attacks, and skill casting, together with synchronized per-frame annotations of character skeletons, world states, camera poses, and depth maps. We further derive WildBench to evaluate models through Action Following and State Alignment. Extensive experiments reveal persistent challenges in modeling semantically rich actions and maintaining long-horizon state consistency, highlighting the need for state-aware video generation. The project page is https://shandaai.github.io/wildworld-project/.

  </details>



- **Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors**  
  Chuanqing Zhuang, Xin Lu, Zehui Deng, Zhengda Lu, Yiqun Wang, Junqi Diao, Jun Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23324v1  
  <details><summary>Abstract</summary>

  Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

  </details>



- **Tightly-Coupled Radar-Visual-Inertial Odometry**  
  Morten Nissov, Mohit Singh, Kostas Alexis  
  _2026-03-24_ · https://arxiv.org/abs/2603.23052v1  
  <details><summary>Abstract</summary>

  Visual-Inertial Odometry (VIO) is a staple for reliable state estimation on constrained and lightweight platforms due to its versatility and demonstrated performance. However, pertinent challenges regarding robust operation in dark, low-texture, obscured environments complicate the use of such methods. Alternatively, Frequency Modulated Continuous Wave (FMCW) radars, and by extension Radar-Inertial Odometry (RIO), offer robustness to these visual challenges, albeit at the cost of reduced information density and worse long-term accuracy. To address these limitations, this work combines the two in a tightly coupled manner, enabling the resulting method to operate robustly regardless of environmental conditions or trajectory dynamics. The proposed method fuses image features, radar Doppler measurements, and Inertial Measurement Unit (IMU) measurements within an Iterated Extended Kalman Filter (IEKF) in real-time, with radar range data augmenting the visual feature depth initialization. The method is evaluated through flight experiments conducted in both indoor and outdoor environments, as well as through challenges to both exteroceptive modalities (such as darkness, fog, or fast flight), thoroughly demonstrating its robustness. The implementation of the proposed method is available at: https://github.com/ntnu-arl/radvio .

  </details>



- **Design Guidelines for Nonlinear Kalman Filters via Covariance Compensation**  
  Shida Jiang, Jaewoong Lee, Shengyu Tao, Scott Moura  
  _2026-03-24_ · https://arxiv.org/abs/2603.22992v1  
  <details><summary>Abstract</summary>

  Nonlinear extensions of the Kalman filter (KF), such as the extended Kalman filter (EKF) and the unscented Kalman filter (UKF), are indispensable for state estimation in complex dynamical systems, yet the conditions for a nonlinear KF to provide robust and accurate estimations remain poorly understood. This work proposes a theoretical framework that identifies the causes of failure and success in certain nonlinear KFs and establishes guidelines for their improvement. Central to our framework is the concept of covariance compensation: the deviation between the covariance predicted by a nonlinear KF and that of the EKF. With this definition and detailed theoretical analysis, we derive three design guidelines for nonlinear KFs: (i) invariance under orthogonal transformations, (ii) sufficient covariance compensation beyond the EKF baseline, and (iii) selection of compensation magnitude that favors underconfidence. Both theoretical analysis and empirical validation confirm that adherence to these principles significantly improves estimation accuracy, whereas fixed parameter choices commonly adopted in the literature are often suboptimal. The codes and the proofs for all the theorems in this paper are available at https://github.com/Shida-Jiang/Guidelines-for-Nonlinear-Kalman-Filters.

  </details>



- **MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects**  
  Shiyu Li, Hannah Schieber, Kristoffer Waldow, Benjamin Busam, Julian Kreimeier, Daniel Roth  
  _2026-03-24_ · https://arxiv.org/abs/2603.22839v1  
  <details><summary>Abstract</summary>

  Multi-camera dynamic Augmented Reality (AR) applications require a camera pose estimation to leverage individual information from each camera in one common system. This can be achieved by combining contextual information, such as markers or objects, across multiple views. While commonly cameras are calibrated in an initial step or updated through the constant use of markers, another option is to leverage information already present in the scene, like known objects. Another downside of marker-based tracking is that markers have to be tracked inside the field-of-view (FoV) of the cameras. To overcome these limitations, we propose a constant dynamic camera pose estimation leveraging spatiotemporal FoV overlaps of known objects on the fly. To achieve that, we enhance the state-of-the-art object pose estimator to update our spatiotemporal scene graph, enabling a relation even among non-overlapping FoV cameras. To evaluate our approach, we introduce a multi-camera, multi-object pose estimation dataset with temporal FoV overlap, including static and dynamic cameras. Furthermore, in FoV overlapping scenarios, we outperform the state-of-the-art on the widely used YCB-V and T-LESS dataset in camera pose accuracy. Our performance on both previous and our proposed datasets validates the effectiveness of our marker-less approach for AR applications. The code and dataset are available on https://github.com/roth-hex-lab/IEEE-VR-2026-MultiCam.

  </details>



- **Typography-Based Monocular Distance Estimation Framework for Vehicle Safety Systems**  
  Manognya Lokesh Reddy, Zheng Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22781v1  
  <details><summary>Abstract</summary>

  Accurate inter-vehicle distance estimation is a cornerstone of advanced driver assistance systems and autonomous driving. While LiDAR and radar provide high precision, their cost prohibits widespread adoption in mass-market vehicles. Monocular vision offers a low-cost alternative but suffers from scale ambiguity and sensitivity to environmental disturbances. This paper introduces a typography-based monocular distance estimation framework, which exploits the standardized typography of license plates as passive fiducial markers for metric distance estimation. The core geometric module uses robust plate detection and character segmentation to measure character height and computes distance via the pinhole camera model. The system incorporates interactive calibration, adaptive detection with strict and permissive modes, and multi-method character segmentation leveraging both adaptive and global thresholding. To enhance robustness, the framework further includes camera pose compensation using lane-based horizon estimation, hybrid deep-learning fusion, temporal Kalman filtering for velocity estimation, and multi-feature fusion that exploits additional typographic cues such as stroke width, character spacing, and plate border thickness. Experimental validation with a calibrated monocular camera in a controlled indoor setup achieved a coefficient of variation of 2.3% in character height across consecutive frames and a mean absolute error of 7.7%. The framework operates without GPU acceleration, demonstrating real-time feasibility. A comprehensive comparison with a plate-width based method shows that character-based ranging reduces the standard deviation of estimates by 35%, translating to smoother, more consistent distance readings in practice, where erratic estimates could trigger unnecessary braking or acceleration.

  </details>



- **SOUPLE: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts**  
  Khanh Binh Nguyen, Chae Jung Park  
  _2026-03-24_ · https://arxiv.org/abs/2603.22732v1  
  <details><summary>Abstract</summary>

  Large-scale pre-trained image-text models exhibit robust multimodal representations, yet applying the Contrastive Language-Image Pre-training (CLIP) model to audio-visual localization remains challenging. Replacing the classification token ([CLS]) with an audio-embedded token ([V_A]) struggles to capture semantic cues, and the prompt "a photo of a [V_A]" fails to establish meaningful connections between audio embeddings and context tokens. To address these issues, we propose Sound-aware Prompt Learning (SOUPLE), which replaces fixed prompts with learnable context tokens. These tokens incorporate visual features to generate conditional context for a mask decoder, effectively bridging semantic correspondence between audio and visual inputs. Experiments on VGGSound, SoundNet, and AVSBench demonstrate that SOUPLE improves localization and segmentation performance.

  </details>



- **Variable-Resolution Virtual Maps for Autonomous Exploration with Unmanned Surface Vehicles (USVs)**  
  Ye Li, Yewei Huang, Wenlong GaoZhang, Alberto Quattrini Li, Brendan Englot, Yuanchang Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22667v1  
  <details><summary>Abstract</summary>

  Autonomous exploration by unmanned surface vehicles (USVs) in near-shore waters requires reliable localisation and consistent mapping over extended areas, but this is challenged by GNSS degradation, environment-induced localisation uncertainty, and limited on-board computation. Virtual map-based methods explicitly model localisation and mapping uncertainty by tightly coupling factor-graph SLAM with a map uncertainty criterion. However, their storage and computational costs scale poorly with fixed-resolution workspace discretisations, leading to inefficiency in large near-shore environments. Moreover, overvaluing feature-sparse open-water regions can increase the risk of SLAM failure as a result of imbalance between exploration and exploitation. To address these limitations, we propose a Variable-Resolution Virtual Map (VRVM), a computationally efficient method for representing map uncertainty using bivariate Gaussian virtual landmarks placed in the cells of an adaptive quadtree. The adaptive quadtree enables an area-weighted uncertainty representation that keeps coarse, far-field virtual landmarks deliberately uncertain while allocating higher resolution to information-dense regions, and reduces the sensitivity of the map valuation to local refinements of the tree. An expectation-maximisation (EM) planner is adopted to evaluate pose and map uncertainty along frontiers using the VRVM, balancing exploration and exploitation. We evaluate VRVM against several state-of-the-art exploration algorithms in the VRX Gazebo simulator, using a realistic marina environment across different testing scenarios with an increasing level of exploration difficulty. The results indicate that our method offers safer behaviour and better utilisation of on-board computation in GNSS-degraded near-shore environments.

  </details>



- **FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario**  
  Hang Dai, Hongwei Fan, Han Zhang, Duojin Wu, Jiyao Zhang, Hao Dong  
  _2026-03-23_ · https://arxiv.org/abs/2603.22102v1  
  <details><summary>Abstract</summary>

  The increasing demand for augmented reality and robotics is driving the need for articulated object reconstruction with high scalability. However, existing settings for reconstructing from discrete articulation states or casual monocular videos require non-trivial axis alignment or suffer from insufficient coverage, limiting their applicability. In this paper, we introduce FreeArtGS, a novel method for reconstructing articulated objects under free-moving scenario, a new setting with a simple setup and high scalability. FreeArtGS combines free-moving part segmentation with joint estimation and end-to-end optimization, taking only a monocular RGB-D video as input. By optimizing with the priors from off-the-shelf point-tracking and feature models, the free-moving part segmentation module identifies rigid parts from relative motion under unconstrained capture. The joint estimation module calibrates the unified object-to-camera poses and recovers joint type and axis robustly from part segmentation. Finally, 3DGS-based end-to-end optimization is implemented to jointly reconstruct visual textures, geometry, and joint angles of the articulated object. We conduct experiments on two benchmarks and real-world free-moving articulated objects. Experimental results demonstrate that FreeArtGS consistently excels in reconstructing free-moving articulated objects and remains highly competitive in previous reconstruction settings, proving itself a practical and effective solution for realistic asset generation. The project page is available at: https://freeartgs.github.io/

  </details>



- **MineRobot: A Unified Framework for Kinematics Modeling and Solving of Underground Mining Robots in Virtual Environments**  
  Shengzhe Hou, Xinming Lu, Tianyu Zhang, Changqing Yan, Xingli Zhang  
  _2026-03-23_ · https://arxiv.org/abs/2603.22055v1  
  <details><summary>Abstract</summary>

  Underground mining robots are increasingly operated in virtual environments (VEs) for training, planning, and digital-twin applications, where reliable kinematics is essential for avoiding hazardous in-situ trials. Unlike typical open-chain industrial manipulators, mining robots are often closed-chain mechanisms driven by linear actuators and involving planar four-bar linkages, which makes both kinematics modeling and real-time solving challenging. We present \emph{MineRobot}, a unified framework for modeling and solving the kinematics of underground mining robots in VEs. First, we introduce the Mining Robot Description Format (MRDF), a domain-specific representation that parameterizes kinematics for mining robots with native semantics for actuators and loop closures. Second, we develop a topology-processing pipeline that contracts four-bar substructures into generalized joints and, for each actuator, extracts an Independent Topologically Equivalent Path (ITEP), which is classified into one of four canonical types. Third, leveraging ITEP independence, we compose per-type solvers into an actuator-centered sequential forward-kinematics (FK) pipeline. Building on the same decomposition, we formulate inverse kinematics (IK) as a bound-constrained optimization problem and solve it with a Gauss--Seidel-style procedure that alternates actuator-length updates. By converting coupled closed-loop kinematics into a sequence of small topology-aware solves, the framework avoids robot-specific hand derivations and supports efficient computation. Experiments demonstrate that MineRobot provides the real-time performance and robustness required by VE applications.

  </details>



- **LRC-WeatherNet: LiDAR, RADAR, and Camera Fusion Network for Real-time Weather-type Classification in Autonomous Driving**  
  Nour Alhuda Albashir, Lars Pernickel, Danial Hamoud, Idriss Gouigah, Eren Erdal Aksoy  
  _2026-03-23_ · https://arxiv.org/abs/2603.21987v1  
  <details><summary>Abstract</summary>

  Autonomous vehicles face major perception and navigation challenges in adverse weather such as rain, fog, and snow, which degrade the performance of LiDAR, RADAR, and RGB camera sensors. While each sensor type offers unique strengths, such as RADAR robustness in poor visibility and LiDAR precision in clear conditions, they also suffer distinct limitations when exposed to environmental obstructions. This study proposes LRC-WeatherNet, a novel multi-sensor fusion framework that integrates LiDAR, RADAR, and camera data for real-time classification of weather conditions. By employing both early fusion using a unified Bird's Eye View representation and mid-level gated fusion of modality-specific feature maps, our approach adapts to the varying reliability of each sensor under changing weather. Evaluated on the extensive MSU-4S dataset covering nine weather types, LRC-WeatherNet achieves superior classification performance and computational efficiency, significantly outperforming unimodal baselines in adverse conditions. This work is the first to combine all three modalities for robust, real-time weather classification in autonomous driving. We release our trained models and source code in https://github.com/nouralhudaalbashir/LRC-WeatherNet.

  </details>



- **Image-Conditioned Adaptive Parameter Tuning for Visual Odometry Frontends**  
  Simone Nascivera, Leonard Bauersfeld, Jeff Delaune, Davide Scaramuzza  
  _2026-03-23_ · https://arxiv.org/abs/2603.21785v1  
  <details><summary>Abstract</summary>

  Resource-constrained autonomous robots rely on sparse direct and semi-direct visual-(inertial)-odometry (VO) pipelines, as they provide a favorable tradeoff between accuracy, robustness, and computational cost. However, the performance of most systems depends critically on hand-tuned hyperparameters governing feature detection, tracking, and outlier rejection. These parameters are typically fixed during deployment, even though their optimal values vary with scene characteristics such as texture density, illumination, motion blur, and sensor noise, leading to brittle performance in real-world environments. We propose the first image-conditioned reinforcement learning framework for online tuning of VO frontend parameters, effectively embedding the expert into the system. Our key idea is to formulate the frontend configuration as a sequential decision-making problem and learn a policy that directly maps visual input to feature detection and tracking parameters. The policy uses a lightweight texture-aware CNN encoder and a privileged critic during training. Unlike prior RL-based approaches that rely solely on internal VO statistics, our method observes the image content and proactively adapts parameters before tracking degrades. Experiments on TartanAirV2 and TUM RGB-D show 3x longer feature tracks and 3x lower computational cost, despite training entirely in simulation.

  </details>


