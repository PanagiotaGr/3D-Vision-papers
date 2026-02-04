# SLAM & Localization

_Updated: 2026-02-04 07:08 UTC_

Total papers shown: **11**


---

- **Z3D: Zero-Shot 3D Visual Grounding from Images**  
  Nikita Drozdov, Andrey Lemeshko, Nikita Gavrilov, Anton Konushin, Danila Rukhovich, Maksim Kolodiazhnyi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03361v1  
  <details><summary>Abstract</summary>

  3D visual grounding (3DVG) aims to localize objects in a 3D scene based on natural language queries. In this work, we explore zero-shot 3DVG from multi-view images alone, without requiring any geometric supervision or object priors. We introduce Z3D, a universal grounding pipeline that flexibly operates on multi-view images while optionally incorporating camera poses and depth maps. We identify key bottlenecks in prior zero-shot methods causing significant performance degradation and address them with (i) a state-of-the-art zero-shot 3D instance segmentation method to generate high-quality 3D bounding box proposals and (ii) advanced reasoning via prompt-based segmentation, which utilizes full capabilities of modern VLMs. Extensive experiments on the ScanRefer and Nr3D benchmarks demonstrate that our approach achieves state-of-the-art performance among zero-shot methods. Code is available at https://github.com/col14m/z3d .

  </details>



- **Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization**  
  Manuel Hofer, Markus Steinberger, Thomas Köhler  
  _2026-02-03_ · https://arxiv.org/abs/2602.03327v1  
  <details><summary>Abstract</summary>

  Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.

  </details>



- **LEVIO: Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices**  
  Jonas Kühne, Christian Vogt, Michele Magno, Luca Benini  
  _2026-02-03_ · https://arxiv.org/abs/2602.03294v1  
  <details><summary>Abstract</summary>

  Accurate, infrastructure-less sensor systems for motion tracking are essential for mobile robotics and augmented reality (AR) applications. The most popular state-of-the-art visual-inertial odometry (VIO) systems, however, are too computationally demanding for resource-constrained hardware, such as micro-drones and smart glasses. This work presents LEVIO, a fully featured VIO pipeline optimized for ultra-low-power compute platforms, allowing six-degrees-of-freedom (DoF) real-time sensing. LEVIO incorporates established VIO components such as Oriented FAST and Rotated BRIEF (ORB) feature tracking and bundle adjustment, while emphasizing a computationally efficient architecture with parallelization and low memory usage to suit embedded microcontrollers and low-power systems-on-chip (SoCs). The paper proposes and details the algorithmic design choices and the hardware-software co-optimization approach, and presents real-time performance on resource-constrained hardware. LEVIO is validated on a parallel-processing ultra-low-power RISC-V SoC, achieving 20 FPS while consuming less than 100 mW, and benchmarked against public VIO datasets, offering a compelling balance between efficiency and accuracy. To facilitate reproducibility and adoption, the complete implementation is released as open-source.

  </details>



- **LaVPR: Benchmarking Language and Vision for Place Recognition**  
  Ofer Idan, Dan Badur, Yosi Keller, Yoli Shavit  
  _2026-02-03_ · https://arxiv.org/abs/2602.03253v1  
  <details><summary>Abstract</summary>

  Visual Place Recognition (VPR) often fails under extreme environmental changes and perceptual aliasing. Furthermore, standard systems cannot perform "blind" localization from verbal descriptions alone, a capability needed for applications such as emergency response. To address these challenges, we introduce LaVPR, a large-scale benchmark that extends existing VPR datasets with over 650,000 rich natural-language descriptions. Using LaVPR, we investigate two paradigms: Multi-Modal Fusion for enhanced robustness and Cross-Modal Retrieval for language-based localization. Our results show that language descriptions yield consistent gains in visually degraded conditions, with the most significant impact on smaller backbones. Notably, adding language allows compact models to rival the performance of much larger vision-only architectures. For cross-modal retrieval, we establish a baseline using Low-Rank Adaptation (LoRA) and Multi-Similarity loss, which substantially outperforms standard contrastive methods across vision-language models. Ultimately, LaVPR enables a new class of localization systems that are both resilient to real-world stochasticity and practical for resource-constrained deployment. Our dataset and code are available at https://github.com/oferidan1/LaVPR.

  </details>



- **From Single Scan to Sequential Consistency: A New Paradigm for LIDAR Relocalization**  
  Minghang Zhu, Zhijing Wang, Yuxin Guo, Wen Li, Sheng Ao, Cheng Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03198v1  
  <details><summary>Abstract</summary>

  LiDAR relocalization aims to estimate the global 6-DoF pose of a sensor in the environment. However, existing regression-based approaches are prone to dynamic or ambiguous scenarios, as they either solely rely on single-frame inference or neglect the spatio-temporal consistency across scans. In this paper, we propose TempLoc, a new LiDAR relocalization framework that enhances the robustness of localization by effectively modeling sequential consistency. Specifically, a Global Coordinate Estimation module is first introduced to predict point-wise global coordinates and associated uncertainties for each LiDAR scan. A Prior Coordinate Generation module is then presented to estimate inter-frame point correspondences by the attention mechanism. Lastly, an Uncertainty-Guided Coordinate Fusion module is deployed to integrate both predictions of point correspondence in an end-to-end fashion, yielding a more temporally consistent and accurate global 6-DoF pose. Experimental results on the NCLT and Oxford Robot-Car benchmarks show that our TempLoc outperforms stateof-the-art methods by a large margin, demonstrating the effectiveness of temporal-aware correspondence modeling in LiDAR relocalization. Our code will be released soon.

  </details>



- **PokeNet: Learning Kinematic Models of Articulated Objects from Human Observations**  
  Anmol Gupta, Weiwei Gu, Omkar Patil, Jun Ki Lee, Nakul Gopalan  
  _2026-02-02_ · https://arxiv.org/abs/2602.02741v1  
  <details><summary>Abstract</summary>

  Articulation modeling enables robots to learn joint parameters of articulated objects for effective manipulation which can then be used downstream for skill learning or planning. Existing approaches often rely on prior knowledge about the objects, such as the number or type of joints. Some of these approaches also fail to recover occluded joints that are only revealed during interaction. Others require large numbers of multi-view images for every object, which is impractical in real-world settings. Furthermore, prior works neglect the order of manipulations, which is essential for many multi-DoF objects where one joint must be operated before another, such as a dishwasher. We introduce PokeNet, an end-to-end framework that estimates articulation models from a single human demonstration without prior object knowledge. Given a sequence of point cloud observations of a human manipulating an unknown object, PokeNet predicts joint parameters, infers manipulation order, and tracks joint states over time. PokeNet outperforms existing state-of-the-art methods, improving joint axis and state estimation accuracy by an average of over 27% across diverse objects, including novel and unseen categories. We demonstrate these gains in both simulation and real-world environments.

  </details>



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


