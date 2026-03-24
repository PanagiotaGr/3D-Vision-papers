# SLAM & Localization

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **9**


---

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



- **Two Experts Are Better Than One Generalist: Decoupling Geometry and Appearance for Feed-Forward 3D Gaussian Splatting**  
  Hwasik Jeong, Seungryong Lee, Gyeongjin Kang, Seungkwon Yang, Xiangyu Sun, Seungtae Nam, Eunbyung Park  
  _2026-03-22_ · https://arxiv.org/abs/2603.21064v1  
  <details><summary>Abstract</summary>

  Pose-free feed-forward 3D Gaussian Splatting (3DGS) has opened a new frontier for rapid 3D modeling, enabling high-quality Gaussian representations to be generated from uncalibrated multi-view images in a single forward pass. The dominant approach in this space adopts unified monolithic architectures, often built on geometry-centric 3D foundation models, to jointly estimate camera poses and synthesize 3DGS representations within a single network. While architecturally streamlined, such "all-in-one" designs may be suboptimal for high-fidelity 3DGS generation, as they entangle geometric reasoning and appearance modeling within a shared representation. In this work, we introduce 2Xplat, a pose-free feed-forward 3DGS framework based on a two-expert design that explicitly separates geometry estimation from Gaussian generation. A dedicated geometry expert first predicts camera poses, which are then explicitly passed to a powerful appearance expert that synthesizes 3D Gaussians. Despite its conceptual simplicity, being largely underexplored in prior works, the proposed approach proves highly effective. In fewer than 5K training iterations, the proposed two-experts pipeline substantially outperforms prior pose-free feed-forward 3DGS approaches and achieves performance on par with state-of-the-art posed methods. These results challenge the prevailing unified paradigm and suggest the potential advantages of modular design principles for complex 3D geometric estimation and appearance synthesis tasks.

  </details>



- **SGAD-SLAM: Splatting Gaussians at Adjusted Depth for Better Radiance Fields in RGBD SLAM**  
  Pengchong Hu, Zhizhong Han  
  _2026-03-22_ · https://arxiv.org/abs/2603.21055v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has made remarkable progress in RGBD SLAM. Current methods usually use 3D Gaussians or view-tied 3D Gaussians to represent radiance fields in tracking and mapping. However, these Gaussians are either too flexible or too limited in movements, resulting in slow convergence or limited rendering quality. To resolve this issue, we adopt pixel-aligned Gaussians but allow each Gaussian to adjust its position along its ray to maximize the rendering quality, even if Gaussians are simplified to improve system scalability. To speed up the tracking, we model the depth distribution around each pixel as a Gaussian distribution, and then use these distributions to align each frame to the 3D scene quickly. We report our evaluations on widely used benchmarks, justify our designs, and show advantages over the latest methods in view rendering, camera tracking, runtime, and storage complexity. Please see our project page for code and videos at https://machineperceptionlab.github.io/SGAD-SLAM-Project .

  </details>



- **Implementing Robust M-Estimators with Certifiable Factor Graph Optimization**  
  Zhexin Xu, Hanna Jiamei Zhang, Helena Calatrava, Pau Closas, David M. Rosen  
  _2026-03-21_ · https://arxiv.org/abs/2603.20932v1  
  <details><summary>Abstract</summary>

  Parameter estimation in robotics and computer vision faces formidable challenges from both outlier contamination and nonconvex optimization landscapes. While M-estimation addresses the problem of outliers through robust loss functions, it creates severely nonconvex problems that are difficult to solve globally. Adaptive reweighting schemes provide one particularly appealing strategy for implementing M-estimation in practice: these methods solve a sequence of simpler weighted least squares (WLS) subproblems, enabling both the use of standard least squares solvers and the recovery of higher-quality estimates than simple local search. However, adaptive reweighting still crucially relies upon solving the inner WLS problems effectively, a task that remains challenging in many robotics applications due to the intrinsic nonconvexity of many common parameter spaces (e.g. rotations and poses). In this paper, we show how one can easily implement adaptively reweighted M-estimators with certifiably correct solvers for the inner WLS subproblems using only fast local optimization over smooth manifolds. Our approach exploits recent work on certifiable factor graph optimization to provide global optimality certificates for the inner WLS subproblems while seamlessly integrating into existing factor graph-based software libraries and workflows. Experimental evaluation on pose-graph optimization and landmark SLAM tasks demonstrates that our adaptively reweighted certifiable estimation approach provides higher-quality estimates than alternative local search-based methods, while scaling tractably to realistic problem sizes.

  </details>



- **PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching**  
  Hanqiao Ye, Yuzhou Liu, Yangdong Liu, Shuhan Shen  
  _2026-03-21_ · https://arxiv.org/abs/2603.20818v1  
  <details><summary>Abstract</summary>

  While structure-based relocalizers have long strived for point correspondences when establishing or regressing query-map associations, in this paper, we pioneer the use of planar primitives and 3D planar maps for lightweight 6-DoF camera relocalization in structured environments. Planar primitives, beyond being fundamental entities in projective geometry, also serve as region-based representations that encapsulate both structural and semantic richness. This motivates us to introduce PlanaReLoc, a streamlined plane-centric paradigm where a deep matcher associates planar primitives across the query image and the map within a learned unified embedding space, after which the 6-DoF pose is solved and refined under a robust framework. Through comprehensive experiments on the ScanNet and 12Scenes datasets across hundreds of scenes, our method demonstrates the superiority of planar primitives in facilitating reliable cross-modal structural correspondences and achieving effective camera relocalization without requiring realistically textured/colored maps, pose priors, or per-scene training. The code and data are available at https://github.com/3dv-casia/PlanaReLoc .

  </details>



- **PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization**  
  Xiaoya Cheng, Long Wang, Yan Liu, Xinyi Liu, Hanlin Tan, Yu Liu, Maojun Zhang, Shen Yan  
  _2026-03-21_ · https://arxiv.org/abs/2603.20778v1  
  <details><summary>Abstract</summary>

  We present PiLoT, a unified framework that tackles UAV-based ego and target geo-localization. Conventional approaches rely on decoupled pipelines that fuse GNSS and Visual-Inertial Odometry (VIO) for ego-pose estimation, and active sensors like laser rangefinders for target localization. However, these methods are susceptible to failure in GNSS-denied environments and incur substantial hardware costs and complexity. PiLoT breaks this paradigm by directly registering live video stream against a geo-referenced 3D map. To achieve robust, accurate, and real-time performance, we introduce three key contributions: 1) a Dual-Thread Engine that decouples map rendering from core localization thread, ensuring both low latency while maintaining drift-free accuracy; 2) a large-scale synthetic dataset with precise geometric annotations (camera pose, depth maps). This dataset enables the training of a lightweight network that generalizes in a zero-shot manner from simulation to real data; and 3) a Joint Neural-Guided Stochastic-Gradient Optimizer (JNGO) that achieves robust convergence even under aggressive motion. Evaluations on a comprehensive set of public and newly collected benchmarks show that PiLoT outperforms state-of-the-art methods while running over 25 FPS on NVIDIA Jetson Orin platform. Our code and dataset is available at: https://github.com/Choyaa/PiLoT.

  </details>


