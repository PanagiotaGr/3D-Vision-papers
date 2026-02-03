# SLAM & Localization

_Updated: 2026-02-03 07:08 UTC_

Total papers shown: **11**


---

- **Relationship-Aware Hierarchical 3D Scene Graph for Task Reasoning**  
  Albert Gassol Puigjaner, Angelos Zacharia, Kostas Alexis  
  _2026-02-02_ · https://arxiv.org/abs/2602.02456v1  
  <details><summary>Abstract</summary>

  Representing and understanding 3D environments in a structured manner is crucial for autonomous agents to navigate and reason about their surroundings. While traditional Simultaneous Localization and Mapping (SLAM) methods generate metric reconstructions and can be extended to metric-semantic mapping, they lack a higher level of abstraction and relational reasoning. To address this gap, 3D scene graphs have emerged as a powerful representation for capturing hierarchical structures and object relationships. In this work, we propose an enhanced hierarchical 3D scene graph that integrates open-vocabulary features across multiple abstraction levels and supports object-relational reasoning. Our approach leverages a Vision Language Model (VLM) to infer semantic relationships. Notably, we introduce a task reasoning module that combines Large Language Models (LLM) and a VLM to interpret the scene graph's semantic and relational information, enabling agents to reason about tasks and interact with their environment more intelligently. We validate our method by deploying it on a quadruped robot in multiple environments and tasks, highlighting its ability to reason about them.

  </details>



- **3D Foundation Model-Based Loop Closing for Decentralized Collaborative SLAM**  
  Pierre-Yves Lajoie, Benjamin Ramtoula, Daniele De Martini, Giovanni Beltrame  
  _2026-02-02_ · https://arxiv.org/abs/2602.02430v1  
  <details><summary>Abstract</summary>

  Decentralized Collaborative Simultaneous Localization And Mapping (C-SLAM) techniques often struggle to identify map overlaps due to significant viewpoint variations among robots. Motivated by recent advancements in 3D foundation models, which can register images despite large viewpoint differences, we propose a robust loop closing approach that leverages these models to establish inter-robot measurements. In contrast to resource-intensive methods requiring full 3D reconstruction within a centralized map, our approach integrates foundation models into existing SLAM pipelines, yielding scalable and robust multi-robot mapping. Our contributions include: (1) integrating 3D foundation models to reliably estimate relative poses from monocular image pairs within decentralized C-SLAM; (2) introducing robust outlier mitigation techniques critical to the use of these relative poses; and (3) developing specialized pose graph optimization formulations that efficiently resolve scale ambiguities. We evaluate our method against state-of-the-art approaches, demonstrating improvements in localization and mapping accuracy, alongside significant gains in computational and memory efficiency. These results highlight the potential of our approach for deployment in large-scale multi-robot scenarios.

  </details>



- **Mapping-Guided Task Discovery and Allocation for Robotic Inspection of Underwater Structures**  
  Marina Ruediger, Ashis G. Banerjee  
  _2026-02-02_ · https://arxiv.org/abs/2602.02389v1  
  <details><summary>Abstract</summary>

  Task generation for underwater multi-robot inspections without prior knowledge of existing geometry can be achieved and optimized through examination of simultaneous localization and mapping (SLAM) data. By considering hardware parameters and environmental conditions, a set of tasks is generated from SLAM meshes and optimized through expected keypoint scores and distance-based pruning. In-water tests are used to demonstrate the effectiveness of the algorithm and determine the appropriate parameters. These results are compared to simulated Voronoi partitions and boustrophedon patterns for inspection coverage on a model of the test environment. The key benefits of the presented task discovery method include adaptability to unexpected geometry and distributions that maintain coverage while focusing on areas more likely to present defects or damage.

  </details>



- **Reformulating AI-based Multi-Object Relative State Estimation for Aleatoric Uncertainty-based Outlier Rejection of Partial Measurements**  
  Thomas Jantos, Giulio Delama, Stephan Weiss, Jan Steinbrener  
  _2026-02-02_ · https://arxiv.org/abs/2602.02006v1  
  <details><summary>Abstract</summary>

  Precise localization with respect to a set of objects of interest enables mobile robots to perform various tasks. With the rise of edge devices capable of deploying deep neural networks (DNNs) for real-time inference, it stands to reason to use artificial intelligence (AI) for the extraction of object-specific, semantic information from raw image data, such as the object class and the relative six degrees of freedom (6-DoF) pose. However, fusing such AI-based measurements in an Extended Kalman Filter (EKF) requires quantifying the DNNs' uncertainty and outlier rejection capabilities. This paper presents the benefits of reformulating the measurement equation in AI-based, object-relative state estimation. By deriving an EKF using the direct object-relative pose measurement, we can decouple the position and rotation measurements, thus limiting the influence of erroneous rotation measurements and allowing partial measurement rejection. Furthermore, we investigate the performance and consistency improvements for state estimators provided by replacing the fixed measurement covariance matrix of the 6-DoF object-relative pose measurements with the predicted aleatoric uncertainty of the DNN.

  </details>



- **Vision-only UAV State Estimation for Fast Flights Without External Localization Systems: A2RL Drone Racing Finalist Approach**  
  Filip Novák, Matěj Petrlík, Matej Novosad, Parakh M. Gupta, Robert Pěnička, Martin Saska  
  _2026-02-02_ · https://arxiv.org/abs/2602.01860v1  
  <details><summary>Abstract</summary>

  Fast flights with aggressive maneuvers in cluttered GNSS-denied environments require fast, reliable, and accurate UAV state estimation. In this paper, we present an approach for onboard state estimation of a high-speed UAV using a monocular RGB camera and an IMU. Our approach fuses data from Visual-Inertial Odometry (VIO), an onboard landmark-based camera measurement system, and an IMU to produce an accurate state estimate. Using onboard measurement data, we estimate and compensate for VIO drift through a novel mathematical drift model. State-of-the-art approaches often rely on more complex hardware (e.g., stereo cameras or rangefinders) and use uncorrected drifting VIO velocities, orientation, and angular rates, leading to errors during fast maneuvers. In contrast, our method corrects all VIO states (position, orientation, linear and angular velocity), resulting in accurate state estimation even during rapid and dynamic motion. Our approach was thoroughly validated through 1600 simulations and numerous real-world experiments. Furthermore, we applied the proposed method in the A2RL Drone Racing Challenge 2025, where our team advanced to the final four out of 210 teams and earned a medal.

  </details>



- **Real-Time Loop Closure Detection in Visual SLAM via NetVLAD and Faiss**  
  Enguang Fan  
  _2026-02-02_ · https://arxiv.org/abs/2602.01673v1  
  <details><summary>Abstract</summary>

  Loop closure detection (LCD) is a core component of simultaneous localization and mapping (SLAM): it identifies revisited places and enables pose-graph constraints that correct accumulated drift. Classic bag-of-words approaches such as DBoW are efficient but often degrade under appearance change and perceptual aliasing. In parallel, deep learning-based visual place recognition (VPR) descriptors (e.g., NetVLAD and Transformer-based models) offer stronger robustness, but their computational cost is often viewed as a barrier to real-time SLAM. In this paper, we empirically evaluate NetVLAD as an LCD module and compare it against DBoW on the KITTI dataset. We introduce a Fine-Grained Top-K precision-recall curve that better reflects LCD settings where a query may have zero or multiple valid matches. With Faiss-accelerated nearestneighbor search, NetVLAD achieves real-time query speed while improving accuracy and robustness over DBoW, making it a practical drop-in alternative for LCD in SLAM.

  </details>



- **Visible Light Positioning With Lamé Curve LEDs: A Generic Approach for Camera Pose Estimation**  
  Wenxuan Pan, Yang Yang, Dong Wei, Zhiyu Zhu, Jintao Wang, Huan Wu, Yao Nie  
  _2026-02-02_ · https://arxiv.org/abs/2602.01577v1  
  <details><summary>Abstract</summary>

  Camera-based visible light positioning (VLP) is a promising technique for accurate and low-cost indoor camera pose estimation (CPE). To reduce the number of required light-emitting diodes (LEDs), advanced methods commonly exploit LED shape features for positioning. Although interesting, they are typically restricted to a single LED geometry, leading to failure in heterogeneous LED-shape scenarios. To address this challenge, this paper investigates Lamé curves as a unified representation of common LED shapes and proposes a generic VLP algorithm using Lamé curve-shaped LEDs, termed LC-VLP. In the considered system, multiple ceiling-mounted Lamé curve-shaped LEDs periodically broadcast their curve parameters via visible light communication, which are captured by a camera-equipped receiver. Based on the received LED images and curve parameters, the receiver can estimate the camera pose using LC-VLP. Specifically, an LED database is constructed offline to store the curve parameters, while online positioning is formulated as a nonlinear least-squares problem and solved iteratively. To provide a reliable initialization, a correspondence-free perspective-\textit{n}-points (FreeP\textit{n}P) algorithm is further developed, enabling approximate CPE without any pre-calibrated reference points. The performance of LC-VLP is verified by both simulations and experiments. Simulations show that LC-VLP outperforms state-of-the-art methods in both circular- and rectangular-LED scenarios, achieving reductions of over 40% in position error and 25% in rotation error. Experiments further show that LC-VLP can achieve an average position accuracy of less than 4 cm.

  </details>



- **TreeLoc: 6-DoF LiDAR Global Localization in Forests via Inter-Tree Geometric Matching**  
  Minwoo Jung, Nived Chebrolu, Lucas Carvalho de Lima, Haedam Oh, Maurice Fallon, Ayoung Kim  
  _2026-02-02_ · https://arxiv.org/abs/2602.01501v1  
  <details><summary>Abstract</summary>

  Reliable localization is crucial for navigation in forests, where GPS is often degraded and LiDAR measurements are repetitive, occluded, and structurally complex. These conditions weaken the assumptions of traditional urban-centric localization methods, which assume that consistent features arise from unique structural patterns, necessitating forest-centric solutions to achieve robustness in these environments. To address these challenges, we propose TreeLoc, a LiDAR-based global localization framework for forests that handles place recognition and 6-DoF pose estimation. We represent scenes using tree stems and their Diameter at Breast Height (DBH), which are aligned to a common reference frame via their axes and summarized using the tree distribution histogram (TDH) for coarse matching, followed by fine matching with a 2D triangle descriptor. Finally, pose estimation is achieved through a two-step geometric verification. On diverse forest benchmarks, TreeLoc outperforms baselines, achieving precise localization. Ablation studies validate the contribution of each component. We also propose applications for long-term forest management using descriptors from a compact global tree database. TreeLoc is open-sourced for the robotics community at https://github.com/minwoo0611/TreeLoc.

  </details>



- **Interacted Planes Reveal 3D Line Mapping**  
  Zeran Ke, Bin Tan, Gui-Song Xia, Yujun Shen, Nan Xue  
  _2026-02-01_ · https://arxiv.org/abs/2602.01296v1  
  <details><summary>Abstract</summary>

  3D line mapping from multi-view RGB images provides a compact and structured visual representation of scenes. We study the problem from a physical and topological perspective: a 3D line most naturally emerges as the edge of a finite 3D planar patch. We present LiP-Map, a line-plane joint optimization framework that explicitly models learnable line and planar primitives. This coupling enables accurate and detailed 3D line mapping while maintaining strong efficiency (typically completing a reconstruction in 3 to 5 minutes per scene). LiP-Map pioneers the integration of planar topology into 3D line mapping, not by imposing pairwise coplanarity constraints but by explicitly constructing interactions between plane and line primitives, thus offering a principled route toward structured reconstruction in man-made environments. On more than 100 scenes from ScanNetV2, ScanNet++, Hypersim, 7Scenes, and Tanks\&Temple, LiP-Map improves both accuracy and completeness over state-of-the-art methods. Beyond line mapping quality, LiP-Map significantly advances line-assisted visual localization, establishing strong performance on 7Scenes. Our code is released at https://github.com/calmke/LiPMAP for reproducible research.

  </details>



- **Invariance on Manifolds: Understanding Robust Visual Representations for Place Recognition**  
  Jintao Cheng, Weibin Li, Zhijian He, Jin Wu, Chi Man Vong, Wei Zhang  
  _2026-01-31_ · https://arxiv.org/abs/2602.00841v1  
  <details><summary>Abstract</summary>

  Visual Place Recognition (VPR) demands representations robust to drastic environmental and viewpoint shifts. Current aggregation paradigms, however, either rely on data-hungry supervision or simplistic first-order statistics, often neglecting intrinsic structural correlations. In this work, we propose a Second-Order Geometric Statistics framework that inherently captures geometric stability without training. We conceptualize scenes as covariance descriptors on the Symmetric Positive Definite (SPD) manifold, where perturbations manifest as tractable congruence transformations. By leveraging geometry-aware Riemannian mappings, we project these descriptors into a linearized Euclidean embedding, effectively decoupling signal structure from noise. Our approach introduces a training-free framework built upon fixed, pre-trained backbones, achieving strong zero-shot generalization without parameter updates. Extensive experiments confirm that our method achieves highly competitive performance against state-of-the-art baselines, particularly excelling in challenging zero-shot scenarios.

  </details>



- **VVLoc: Prior-free 3-DoF Vehicle Visual Localization**  
  Ze Huang, Zhongyang Xiao, Mingliang Song, Longan Yang, Hongyuan Yuan, Li Sun  
  _2026-01-31_ · https://arxiv.org/abs/2602.00810v1  
  <details><summary>Abstract</summary>

  Localization is a critical technology in autonomous driving, encompassing both topological localization, which identifies the most similar map keyframe to the current observation, and metric localization, which provides precise spatial coordinates. Conventional methods typically address these tasks independently, rely on single-camera setups, and often require additional 3D semantic or pose priors, while lacking mechanisms to quantify the confidence of localization results, making them less feasible for real industrial applications. In this paper, we propose VVLoc, a unified pipeline that employs a single neural network to concurrently achieve topological and metric vehicle localization using multi-camera system. VVLoc first evaluates the geo-proximity between visual observations, then estimates their relative metric poses using a matching strategy, while also providing a confidence measure. Additionally, the training process for VVLoc is highly efficient, requiring only pairs of visual data and corresponding ground-truth poses, eliminating the need for complex supplementary data. We evaluate VVLoc not only on the publicly available datasets, but also on a more challenging self-collected dataset, demonstrating its ability to deliver state-of-the-art localization accuracy across a wide range of localization tasks.

  </details>


