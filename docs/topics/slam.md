# SLAM & Localization

_Updated: 2026-03-03 07:08 UTC_

Total papers shown: **7**


---

- **WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments**  
  Joshua Knights, Joseph Reid, Kaushik Roy, David Hall, Mark Cox, Peyman Moghadam  
  _2026-03-02_ · https://arxiv.org/abs/2603.01475v1  
  <details><summary>Abstract</summary>

  Recent years have seen a significant increase in demand for robotic solutions in unstructured natural environments, alongside growing interest in bridging 2D and 3D scene understanding. However, existing robotics datasets are predominantly captured in structured urban environments, making them inadequate for addressing the challenges posed by complex, unstructured natural settings. To address this gap, we propose WildCross, a cross-modal benchmark for place recognition and metric depth estimation in large-scale natural environments. WildCross comprises over 476K sequential RGB frames with semi-dense depth and surface normal annotations, each aligned with accurate 6DoF poses and synchronized dense lidar submaps. We conduct comprehensive experiments on visual, lidar, and cross-modal place recognition, as well as metric depth estimation, demonstrating the value of WildCross as a challenging benchmark for multi-modal robotic perception tasks. We provide access to the code repository and dataset at https://csiro-robotics.github.io/WildCross.

  </details>



- **D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems**  
  Yarong Luo, Wentao Lu, Chi Guo  
  _2026-03-02_ · https://arxiv.org/abs/2603.01404v1  
  <details><summary>Abstract</summary>

  Cooperative localization is essential for swarm applications like collaborative exploration and search-and-rescue missions. However, maintaining real-time capability, robustness, and computational efficiency on resource-constrained platforms presents significant challenges. To address these challenges, we propose D-GVIO, a buffer-driven and fully decentralized GNSS-Visual-Inertial Odometry (GVIO) framework that leverages a novel buffering strategy to support efficient and robust distributed state estimation. The proposed framework is characterized by four core mechanisms. Firstly, through covariance segmentation, covariance intersection and buffering strategy, we modularize propagation and update steps in distributed state estimation, significantly reducing computational and communication burdens. Secondly, the left-invariant extended Kalman filter (L-IEKF) is adopted for information fusion, which exhibits superior state estimation performance over the traditional extended Kalman filter (EKF) since its state transition matrix is independent of the system state. Thirdly, a buffer-based re-propagation strategy is employed to handle delayed measurements efficiently and accurately by leveraging the L-IEKF, eliminating the need for costly re-computation. Finally, an adaptive buffer-driven outlier detection method is proposed to dynamically cull GNSS outliers, enhancing robustness in GNSS-challenged environments.

  </details>



- **Certifiable Estimation with Factor Graphs**  
  Zhexin Xu, Nikolas R. Sanderson, Hanna Jiamei Zhang, David M. Rosen  
  _2026-03-01_ · https://arxiv.org/abs/2603.01267v1  
  <details><summary>Abstract</summary>

  Factor graphs provide a convenient modular modeling language that enables practitioners to design and deploy high-performance robotic state estimation systems by composing simple, reusable building blocks. However, inference in these models is typically performed using local optimization methods that can converge to suboptimal solutions, a serious reliability concern in safety-critical applications. Conversely, certifiable estimators based on convex relaxation can recover verifiably globally optimal solutions in many practical settings, but the computational cost of solving their large-scale relaxations necessitates specialized, structure-exploiting solvers that require substantial expertise to implement, significantly hampering practical deployment. In this paper, we show that these two paradigms, which have thus far been treated as independent in the literature, can be naturally synthesized into a unified framework for certifiable factor graph optimization. The key insight is that factor graph structure is preserved under Shor's relaxation and Burer-Monteiro factorization: applying these transformations to a QCQP with an associated factor graph representation yields a lifted problem admitting a factor graph model with identical connectivity, in which variables and factors are simple one-to-one algebraic transformations of those in the original QCQP. This structural preservation enables the Riemannian Staircase methodology for certifiable estimation to be implemented using the same mature, highly-performant factor graph libraries and workflows already ubiquitously employed throughout robotics and computer vision, making certifiable estimation as straightforward to design and deploy as conventional factor graph inference.

  </details>



- **riMESA: Consensus ADMM for Real-World Collaborative SLAM**  
  Daniel McGann, Michael Kaess  
  _2026-03-01_ · https://arxiv.org/abs/2603.01178v1  
  <details><summary>Abstract</summary>

  Collaborative Simultaneous Localization and Mapping (C-SLAM) is a fundamental capability for multi-robot teams as it enables downstream tasks like planning and navigation. However, existing C-SLAM back-end algorithms that are required to solve this problem struggle to address the practical realities of real-world deployments (i.e. communication limitations, outlier measurements, and online operation). In this paper we propose Robust Incremental Manifold Edge-based Separable ADMM (riMESA) -- a robust, incremental, and distributed C-SLAM back-end that is resilient to outliers, reliable in the face of limited communication, and can compute accurate state estimates for a multi-robot team in real-time. Through the development of riMESA, we, more broadly, make an argument for the use of Consensus Alternating Direction Method of Multipliers as a theoretical foundation for distributed optimization tasks in robotics like C-SLAM due to its flexibility, accuracy, and fast convergence. We conclude this work with an in-depth evaluation of riMESA on a variety of C-SLAM problem scenarios and communication network conditions using both synthetic and real-world C-SLAM data. These experiments demonstrate that riMESA is able to generalize across conditions, produce accurate state estimates, operate in real-time, and outperform the accuracy of prior works by a factor >7x on real-world datasets.

  </details>



- **Path Integral Particle Filtering for Hybrid Systems via Saltation Matrices**  
  Karthik Shaji, Sreeranj Jayadevan, Bo Yuan, Hongzhe Yu, Yongxin Chen  
  _2026-03-01_ · https://arxiv.org/abs/2603.01176v1  
  <details><summary>Abstract</summary>

  We present an optimal-control-based particle filtering method for state estimation in hybrid systems that undergo intermittent contact with their environments. We follow the path integral filtering framework that exploits the duality between the smoothing problem and optimal control. We leverage saltation matrices to map out the uncertainty propagation during contact events for hybrid systems. The resulting path integral optimal control problem allows for a state estimation algorithm robust to outlier effects, flexible to non-Gaussian noise distributions, that also handles the challenging contact dynamics in hybrid systems. This work offers a computationally efficient and reliable estimation algorithm for hybrid systems with stochastic dynamics. We also present extensive experimental results demonstrating that our approach consistently outperforms strong baselines across multiple settings.

  </details>



- **VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection**  
  Yang Cao, Feize Wu, Dave Zhenyu Chen, Yingji Zhong, Lanqing Hong, Dan Xu  
  _2026-03-01_ · https://arxiv.org/abs/2603.00912v1  
  <details><summary>Abstract</summary>

  Current multi-view indoor 3D object detectors rely on sensor geometry that is costly to obtain (i.e., precisely calibrated multi-view camera poses) to fuse multi-view information into a global scene representation, limiting deployment in real-world scenes. We target a more practical setting: Sensor-Geometry-Free (SG-Free) multi-view indoor 3D object detection, where there are no sensor-provided geometric inputs (multi-view poses or depth). Recent Visual Geometry Grounded Transformer (VGGT) shows that strong 3D cues can be inferred directly from images. Building on this insight, we present VGGT-Det, the first framework tailored for SG-Free multi-view indoor 3D object detection. Rather than merely consuming VGGT predictions, our method integrates VGGT encoder into a transformer-based pipeline. To effectively leverage both the semantic and geometric priors from inside VGGT, we introduce two novel key components: (i) Attention-Guided Query Generation (AG): exploits VGGT attention maps as semantic priors to initialize object queries, improving localization by focusing on object regions while preserving global spatial structure; (ii) Query-Driven Feature Aggregation (QD): a learnable See-Query interacts with object queries to 'see' what they need, and then dynamically aggregates multi-level geometric features across VGGT layers that progressively lift 2D features into 3D. Experiments show that VGGT-Det significantly surpasses the best-performing method in the SG-Free setting by 4.4 and 8.6 mAP@0.25 on ScanNet and ARKitScenes, respectively. Ablation study shows that VGGT's internally learned semantic and geometric priors can be effectively leveraged by our AG and QD.

  </details>



- **TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward Pose-free Reconstruction**  
  Yihui Li, Chengxin Lv, Zichen Tang, Hongyu Yang, Di Huang  
  _2026-02-28_ · https://arxiv.org/abs/2603.00697v1  
  <details><summary>Abstract</summary>

  We present TokenSplat, a feed-forward framework for joint 3D Gaussian reconstruction and camera pose estimation from unposed multi-view images. At its core, TokenSplat introduces a Token-aligned Gaussian Prediction module that aligns semantically corresponding information across views directly in the feature space. Guided by coarse token positions and fusion confidence, it aggregates multi-scale contextual features to enable long-range cross-view reasoning and reduce redundancy from overlapping Gaussians. To further enhance pose robustness and disentangle viewpoint cues from scene semantics, TokenSplat employs learnable camera tokens and an Asymmetric Dual-Flow Decoder (ADF-Decoder) that enforces directionally constrained communication between camera and image tokens. This maintains clean factorization within a feed-forward architecture, enabling coherent reconstruction and stable pose estimation without iterative refinement. Extensive experiments demonstrate that TokenSplat achieves higher reconstruction fidelity and novel-view synthesis quality in pose-free settings, and significantly improves pose estimation accuracy compared to prior pose-free methods. Project page: https://kidleyh.github.io/tokensplat/.

  </details>


