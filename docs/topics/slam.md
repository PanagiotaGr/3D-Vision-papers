# SLAM & Localization

_Updated: 2026-03-04 07:04 UTC_

Total papers shown: **9**


---

- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>



- **Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting**  
  Kaiqiang Xiong, Rui Peng, Jiahao Wu, Zhanke Wang, Jie Liang, Xiaoyun Zheng, Feng Gao, Ronggang Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02893v1  
  <details><summary>Abstract</summary>

  3D human reconstruction from a single image is a challenging problem and has been exclusively studied in the literature. Recently, some methods have resorted to diffusion models for guidance, optimizing a 3D representation via Score Distillation Sampling(SDS) or generating a back-view image for facilitating reconstruction. However, these methods tend to produce unsatisfactory artifacts (\textit{e.g.} flattened human structure or over-smoothing results caused by inconsistent priors from multiple views) and struggle with real-world generalization in the wild. In this work, we present \emph{MVD-HuGaS}, enabling free-view 3D human rendering from a single image via a multi-view human diffusion model. We first generate multi-view images from the single reference image with an enhanced multi-view diffusion model, which is well fine-tuned on high-quality 3D human datasets to incorporate 3D geometry priors and human structure priors. To infer accurate camera poses from the sparse generated multi-view images for reconstruction, an alignment module is introduced to facilitate joint optimization of 3D Gaussians and camera poses. Furthermore, we propose a depth-based Facial Distortion Mitigation module to refine the generated facial regions, thereby improving the overall fidelity of the reconstruction. Finally, leveraging the refined multi-view images, along with their accurate camera poses, MVD-HuGaS optimizes the 3D Gaussians of the target human for high-fidelity free-view renderings. Extensive experiments on Thuman2.0 and 2K2K datasets show that the proposed MVD-HuGaS achieves state-of-the-art performance on single-view 3D human rendering.

  </details>



- **Emerging trends in Cislunar Space for Lunar Science Exploration and Space Robotics aiding Human Spaceflight Safety**  
  Arsalan Muhammad, Yue Wang, Hai Huang, Hao Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02878v1  
  <details><summary>Abstract</summary>

  In recent years, the Moon has emerged as an unparalleled extraterrestrial testbed for advancing cuttingedge technological and scientific research critical to enabling sustained human presence on its surface and supporting future interplanetary exploration. This study identifies and investigates two pivotal research domains with substantial transformative potential for accelerating humanity interplanetary aspirations. First is Lunar Science Exploration with Artificial Intelligence and Space Robotics which focusses on AI and Space Robotics redefining the frontiers of space exploration. Second being Space Robotics aid in manned spaceflight to the Moon serving as critical assets for pre-deployment infrastructure development, In-Situ Resource Utilization, surface operations support, and astronaut safety assurance. By integrating autonomy, machine learning, and realtime sensor fusion, space robotics not only augment human capabilities but also serve as force multipliers in achieving sustainable lunar exploration, paving the way for future crewed missions to Mars and beyond.

  </details>



- **Agentic Self-Evolutionary Replanning for Embodied Navigation**  
  Guoliang Li, Ruihua Han, Chengyang Li, He Li, Shuai Wang, Wenchao Ding, Hong Zhang, Chengzhong Xu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02772v1  
  <details><summary>Abstract</summary>

  Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.

  </details>



- **Robust Tightly-Coupled Filter-Based Monocular Visual-Inertial State Estimation and Graph-Based Evaluation for Autonomous Drone Racing**  
  Maulana Bisyir Azhari, Donghun Han, SungJun Park, David Hyunchul Shim  
  _2026-03-03_ · https://arxiv.org/abs/2603.02742v1  
  <details><summary>Abstract</summary>

  Autonomous drone racing (ADR) demands state estimation that is simultaneously computationally efficient and resilient to the perceptual degradation experienced during extreme velocity and maneuvers. Traditional frameworks typically rely on conventional visual-inertial pipelines with loosely-coupled gate-based Perspective-n-Points (PnP) corrections that suffer from a rigid requirement for four visible features and information loss in intermediate steps. Furthermore, the absence of GNSS and Motion Capture systems in uninstrumented, competitive racing environments makes the objective evaluation of such systems remarkably difficult. To address these limitations, we propose ADR-VINS, a robust, monocular visual-inertial state estimation framework based on an Error-State Kalman Filter (ESKF) tailored for autonomous drone racing. Our approach integrates direct pixel reprojection errors from gate corners features as innovation terms within the filter. By bypassing intermediate PnP solvers, ADR-VINS maintains valid state updates with as few as two visible corners and utilizes robust reweighting instead of RANSAC-based schemes to handle outliers, enhancing computational efficiency. Furthermore, we introduce ADR-FGO, an offline Factor-Graph Optimization framework to generate high-fidelity reference trajectories that facilitate post-flight performance evaluation and analysis on uninstrumented, GNSS-denied environments. The proposed system is validated using TII-RATM dataset, where ADR-VINS achieves an average RMS translation error of 0.134 m, while ADR-FGO yields 0.060 m as a smoothing-based reference. Finally, ADR-VINS was successfully deployed in the A2RL Drone Championship Season 2, maintaining stable and robust estimation despite noisy detections during high-agility flight at top speeds of 20.9 m/s. We further utilize ADR-FGO for post-flight evaluation in uninstrumented racing environments.

  </details>



- **Cross-view geo-localization, Image retrieval, Multiscale geometric modeling, Frequency domain enhancement**  
  Hongying Zhang, ShuaiShuai Ma  
  _2026-03-03_ · https://arxiv.org/abs/2603.02726v1  
  <details><summary>Abstract</summary>

  Cross-view geo-localization (CVGL) aims to establish spatial correspondences between images captured from significantly different viewpoints and constitutes a fundamental technique for visual localization in GNSS-denied environments. Nevertheless, CVGL remains challenging due to severe geometric asymmetry, texture inconsistency across imaging domains, and the progressive degradation of discriminative local information. Existing methods predominantly rely on spatial domain feature alignment, which is inherently sensitive to large scale viewpoint variations and local disturbances. To alleviate these limitations, this paper proposes the Spatial and Frequency Domain Enhancement Network (SFDE), which leverages complementary representations from spatial and frequency domains. SFDE adopts a three branch parallel architecture to model global semantic context, local geometric structure, and statistical stability in the frequency domain, respectively, thereby characterizing consistency across domains from the perspectives of scene topology, multiscale structural patterns, and frequency invariance. The resulting complementary features are jointly optimized in a unified embedding space via progressive enhancement and coupled constraints, enabling the learning of cross-view representations with consistency across multiple granularities. Comprehensive experiments show that SFDE achieves competitive performance and in many cases even surpasses state-of-the-art methods, while maintaining a lightweight and computationally efficient design. {Our code is available at https://github.com/Mashuaishuai669/SFDE

  </details>



- **Tensegrity Robot Endcap-Ground Contact Estimation with Symmetry-aware Heterogeneous Graph Neural Network**  
  Wenzhe Tong, Yicheng Jiang, Chi Zhang, Maani Ghaffari, Xiaonan Huang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02596v1  
  <details><summary>Abstract</summary>

  Tensegrity robots possess lightweight and resilient structures but present significant challenges for state estimation due to compliant and distributed ground contacts. This paper introduces a symmetry-aware heterogeneous graph neural network (Sym-HGNN) that infers contact states directly from proprioceptive measurements, including IMU and cable-length histories, without dedicated contact sensors. The network incorporates the robot's dihedral symmetry $D_3$ into the message-passing process to enhance sample efficiency and generalization. The predicted contacts are integrated into a state-of-the-art contact-aided invariant extended Kalman filter (InEKF) for improved pose estimation. Simulation results demonstrate that the proposed method achieves up to 15% higher accuracy and 5% higher F1-score using only 20% of the training data compared to the CNN and MI-HGNN baselines, while maintaining low-drift and physically consistent state estimation results comparable to ground truth contacts. This work highlights the potential of fully proprioceptive sensing for accurate and robust state estimation in tensegrity robots. Code available at: https://github.com/Jonathan-Twz/Tensegrity-Sym-HGNN

  </details>



- **PathSpace: Rapid continuous map approximation for efficient SLAM using B-Splines in constrained environments**  
  Aduen Benjumea, Andrew Bradley, Alexander Rast, Matthias Rolf  
  _2026-03-03_ · https://arxiv.org/abs/2603.02538v1  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) plays a crucial role in enabling autonomous vehicles to navigate previously unknown environments. Semantic SLAM mostly extends visual SLAM, leveraging the higher density information available to reason about the environment in a more human-like manner. This allows for better decision making by exploiting prior structural knowledge of the environment, usually in the form of labels. Current semantic SLAM techniques still mostly rely on a dense geometric representation of the environment, limiting their ability to apply constraints based on context. We propose PathSpace, a novel semantic SLAM framework that uses continuous B-splines to represent the environment in a compact manner, while also maintaining and reasoning through the continuous probability density functions required for probabilistic reasoning. This system applies the multiple strengths of B-splines in the context of SLAM to interpolate and fit otherwise discrete sparse environments. We test this framework in the context of autonomous racing, where we exploit pre-specified track characteristics to produce significantly reduced representations at comparable levels of accuracy to traditional landmark based methods and demonstrate its potential in limiting the resources used by a system with minimal accuracy loss.

  </details>



- **MERG3R: A Divide-and-Conquer Approach to Large-Scale Neural Visual Geometry**  
  Leo Kaixuan Cheng, Abdus Shaikh, Ruofan Liang, Zhijie Wu, Yushi Guan, Nandita Vijaykumar  
  _2026-03-02_ · https://arxiv.org/abs/2603.02351v1  
  <details><summary>Abstract</summary>

  Recent advancements in neural visual geometry, including transformer-based models such as VGGT and Pi3, have achieved impressive accuracy on 3D reconstruction tasks. However, their reliance on full attention makes them fundamentally limited by GPU memory capacity, preventing them from scaling to large, unordered image collections. We introduce MERG3R, a training-free divide-and-conquer framework that enables geometric foundation models to operate far beyond their native memory limits. MERG3R first reorders and partitions unordered images into overlapping, geometrically diverse subsets that can be reconstructed independently. It then merges the resulting local reconstructions through an efficient global alignment and confidence-weighted bundle adjustment procedure, producing a globally consistent 3D model. Our framework is model-agnostic and can be paired with existing neural geometry models. Across large-scale datasets, including 7-Scenes, NRGBD, Tanks & Temples, and Cambridge Landmarks, MERG3R consistently improves reconstruction accuracy, memory efficiency, and scalability, enabling high-quality reconstruction when the dataset exceeds memory capacity limits.

  </details>


