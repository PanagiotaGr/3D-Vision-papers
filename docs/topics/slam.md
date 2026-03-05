# SLAM & Localization

_Updated: 2026-03-05 07:08 UTC_

Total papers shown: **16**


---

- **Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight**  
  Chao Qin, Jiaxu Xing, Rudolf Reiter, Angel Romero, Yifan Lin, Hugh H. -T. Liu, Davide Scaramuzza  
  _2026-03-04_ · https://arxiv.org/abs/2603.04305v1  
  <details><summary>Abstract</summary>

  Agile quadrotor flight pushes the limits of control, actuation, and onboard perception. While time-optimal trajectory planning has been extensively studied, existing approaches typically neglect the tight coupling between vehicle dynamics, environmental geometry, and the visual requirements of onboard state estimation. As a result, trajectories that are dynamically feasible may fail in closed-loop execution due to degraded visual quality. This paper introduces a unified time-optimal trajectory optimization framework for vision-based quadrotors that explicitly incorporates perception constraints alongside full nonlinear dynamics, rotor actuation limits, aerodynamic effects, camera field-of-view constraints, and convex geometric gate representations. The proposed formulation solves minimum-time lap trajectories for arbitrary racetracks with diverse gate shapes and orientations, while remaining numerically robust and computationally efficient. We derive an information-theoretic position uncertainty metric to quantify visual state-estimation quality and integrate it into the planner through three perception objectives: position uncertainty minimization, sequential field-of-view constraints, and look-ahead alignment. This enables systematic exploration of the trade-offs between speed and perceptual reliability. To accurately track the resulting perception-aware trajectories, we develop a model predictive contouring tracking controller that separates lateral and progress errors. Experiments demonstrate real-world flight speeds up to 9.8 m/s with 0.07 m average tracking error, and closed-loop success rates improved from 55% to 100% on a challenging Split-S course. The proposed system provides a scalable benchmark for studying the fundamental limits of perception-aware, time-optimal autonomous flight.

  </details>



- **SSR: A Generic Framework for Text-Aided Map Compression for Localization**  
  Mohammad Omama, Po-han Li, Harsh Goel, Minkyu Choi, Behdad Chalaki, Vaishnav Tadiparthi, Hossein Nourkhiz Mahjoub, Ehsan Moradi Pari, Sandeep P. Chinchali  
  _2026-03-04_ · https://arxiv.org/abs/2603.04272v1  
  <details><summary>Abstract</summary>

  Mapping is crucial in robotics for localization and downstream decision-making. As robots are deployed in ever-broader settings, the maps they rely on continue to increase in size. However, storing these maps indefinitely (cold storage), transferring them across networks, or sending localization queries to cloud-hosted maps imposes prohibitive memory and bandwidth costs. We propose a text-enhanced compression framework that reduces both memory and bandwidth footprints while retaining high-fidelity localization. The key idea is to treat text as an alternative modality: one that can be losslessly compressed with large language models. We propose leveraging lightweight text descriptions combined with very small image feature vectors, which capture "complementary information" as a compact representation for the mapping task. Building on this, our novel technique, Similarity Space Replication (SSR), learns an adaptive image embedding in one shot that captures only the information "complementary" to the text descriptions. We validate our compression framework on multiple downstream localization tasks, including Visual Place Recognition as well as object-centric Monte Carlo localization in both indoor and outdoor settings. SSR achieves 2 times better compression than competing baselines on state-of-the-art datasets, including TokyoVal, Pittsburgh30k, Replica, and KITTI.

  </details>



- **HBRB-BoW: A Retrained Bag-of-Words Vocabulary for ORB-SLAM via Hierarchical BRB-KMeans**  
  Minjae Lee, Sang-Min Choi, Gun-Woo Kim, Suwon Lee  
  _2026-03-04_ · https://arxiv.org/abs/2603.04144v1  
  <details><summary>Abstract</summary>

  In visual simultaneous localization and mapping (SLAM), the quality of the visual vocabulary is fundamental to the system's ability to represent environments and recognize locations. While ORB-SLAM is a widely used framework, its binary vocabulary, trained through the k-majority-based bag-of-words (BoW) approach, suffers from inherent precision loss. The inability of conventional binary clustering to represent subtle feature distributions leads to the degradation of visual words, a problem that is compounded as errors accumulate and propagate through the hierarchical tree structure. To address these structural deficiencies, this paper proposes hierarchical binary-to-real-and-back (HBRB)-BoW, a refined hierarchical binary vocabulary training algorithm. By integrating a global real-valued flow within the hierarchical clustering process, our method preserves high-fidelity descriptor information until the final binarization at the leaf nodes. Experimental results demonstrate that the proposed approach yields a more discriminative and well-structured vocabulary than traditional methods, significantly enhancing the representational integrity of the visual dictionary in complex environments. Furthermore, replacing the default ORB-SLAM vocabulary file with our HBRB-BoW file is expected to improve performance in loop closing and relocalization tasks.

  </details>



- **Long-Term Visual Localization in Dynamic Benthic Environments: A Dataset, Footprint-Based Ground Truth, and Visual Place Recognition Benchmark**  
  Martin Kvisvik Larsen, Oscar Pizarro  
  _2026-03-04_ · https://arxiv.org/abs/2603.04056v1  
  <details><summary>Abstract</summary>

  Long-term visual localization has the potential to reduce cost and improve mapping quality in optical benthic monitoring with autonomous underwater vehicles (AUVs). Despite this potential, long-term visual localization in benthic environments remains understudied, primarily due to the lack of curated datasets for benchmarking. Moreover, limited georeferencing accuracy and image footprints necessitate precise geometric information for accurate ground-truthing. In this work, we address these gaps by presenting a curated dataset for long-term visual localization in benthic environments and a novel method to ground-truth visual localization results for near-nadir underwater imagery. Our dataset comprises georeferenced AUV imagery from five benthic reference sites, revisited over periods up to six years, and includes raw and color-corrected stereo imagery, camera calibrations, and sub-decimeter registered camera poses. To our knowledge, this is the first curated underwater dataset for long-term visual localization spanning multiple sites and photic-zone habitats. Our ground-truthing method estimates 3D seafloor image footprints and links camera views with overlapping footprints, ensuring that ground-truth links reflect shared visual content. Building on this dataset and ground truth, we benchmark eight state-of-the-art visual place recognition (VPR) methods and find that Recall@K is significantly lower on our dataset than on established terrestrial and underwater benchmarks. Finally, we compare our footprint-based ground truth to a traditional location-based ground truth and show that distance-threshold ground-truthing can overestimate VPR Recall@K at sites with rugged terrain and altitude variations. Together, the curated dataset, ground-truthing method, and VPR benchmark provide a stepping stone for advancing long-term visual localization in dynamic benthic environments.

  </details>



- **HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance**  
  Mengfan He, Xingyu Shao, Chunyu Li, Chao Chen, Liangzheng Sun, Ziyang Meng, Yuanqing Wu  
  _2026-03-04_ · https://arxiv.org/abs/2603.04050v1  
  <details><summary>Abstract</summary>

  In this work, we propose HE-VPR, a visual place recognition (VPR) framework that incorporates height estimation. Our system decouples height inference from place recognition, allowing both modules to share a frozen DINOv2 backbone. Two lightweight bypass adapter branches are integrated into our system. The first estimates the height partition of the query image via retrieval from a compact height database, and the second performs VPR within the corresponding height-specific sub-database. The adaptation design reduces training cost and significantly decreases the search space of the database. We also adopt a center-weighted masking strategy to further enhance the robustness against scale differences. Experiments on two self-collected challenging multi-altitude datasets demonstrate that HE-VPR achieves up to 6.1\% Recall@1 improvement over state-of-the-art ViT-based baselines and reduces memory usage by up to 90\%. These results indicate that HE-VPR offers a scalable and efficient solution for height-aware aerial VPR, enabling practical deployment in GNSS-denied environments. All the code and datasets for this work have been released on https://github.com/hmf21/HE-VPR.

  </details>



- **DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation**  
  Tuan Duc Ngo, Jiahui Huang, Seoung Wug Oh, Kevin Blackburn-Matzen, Evangelos Kalogerakis, Chuang Gan, Joon-Young Lee  
  _2026-03-04_ · https://arxiv.org/abs/2603.03744v1  
  <details><summary>Abstract</summary>

  Estimating accurate, view-consistent geometry and camera poses from uncalibrated multi-view/video inputs remains challenging - especially at high spatial resolutions and over long sequences. We present DAGE, a dual-stream transformer whose main novelty is to disentangle global coherence from fine detail. A low-resolution stream operates on aggressively downsampled frames with alternating frame/global attention to build a view-consistent representation and estimate cameras efficiently, while a high-resolution stream processes the original images per-frame to preserve sharp boundaries and small structures. A lightweight adapter fuses these streams via cross-attention, injecting global context without disturbing the pretrained single-frame pathway. This design scales resolution and clip length independently, supports inputs up to 2K, and maintains practical inference cost. DAGE delivers sharp depth/pointmaps, strong cross-view consistency, and accurate poses, establishing new state-of-the-art results for video geometry estimation and multi-view reconstruction.

  </details>



- **Large-Language-Model-Guided State Estimation for Partially Observable Task and Motion Planning**  
  Yoonwoo Kim, Raghav Arora, Roberto Martín-Martín, Peter Stone, Ben Abbatematteo, Yoonchang Sung  
  _2026-03-04_ · https://arxiv.org/abs/2603.03704v1  
  <details><summary>Abstract</summary>

  Robot planning in partially observable environments, where not all objects are known or visible, is a challenging problem, as it requires reasoning under uncertainty through partially observable Markov decision processes. During the execution of a computed plan, a robot may unexpectedly observe task-irrelevant objects, which are typically ignored by naive planners. In this work, we propose incorporating two types of common-sense knowledge: (1) certain objects are more likely to be found in specific locations; and (2) similar objects are likely to be co-located, while dissimilar objects are less likely to be found together. Manually engineering such knowledge is complex, so we explore leveraging the powerful common-sense reasoning capabilities of large language models (LLMs). Our planning and execution framework, CoCo-TAMP, introduces a hierarchical state estimation that uses LLM-guided information to shape the belief over task-relevant objects, enabling efficient solutions to long-horizon task and motion planning problems. In experiments, CoCo-TAMP achieves an average reduction of 62.7 in planning and execution time in simulation, and 72.6 in real-world demonstrations, compared to a baseline that does not incorporate either type of common-sense knowledge.

  </details>



- **Real-time tightly coupled GNSS and IMU integration via Factor Graph Optimization**  
  Radu-Andrei Cioaca, Paul Irofti, Cristian Rusu, Gianluca Caparra, Andrei-Alexandru Marinache, Florin Stoican  
  _2026-03-03_ · https://arxiv.org/abs/2603.03556v1  
  <details><summary>Abstract</summary>

  Reliable positioning in dense urban environments remains challenging due to frequent GNSS signal blockage, multipath, and rapidly varying satellite geometry. While factor graph optimization (FGO)-based GNSS-IMU fusion has demonstrated strong robustness and accuracy, most formulations remain offline. In this work, we present a real-time tightly coupled GNSS-IMU FGO method that enables causal state estimation via incremental optimization with fixed-lag marginalization, and we evaluate its performance in a highly urbanized GNSS-degraded environment using the UrbanNav dataset.

  </details>



- **Overlapping Domain Decomposition for Distributed Pose Graph Optimization**  
  Aneesa Sonawalla, Yulun Tian, Jonathan P. How  
  _2026-03-03_ · https://arxiv.org/abs/2603.03499v1  
  <details><summary>Abstract</summary>

  We present ROBO (Riemannian Overlapping Block Optimization), a distributed and parallel approach to multi-robot pose graph optimization (PGO) based on the idea of overlapping domain decomposition. ROBO offers a middle ground between centralized and fully distributed solvers, where the amount of pose information shared between robots at each optimization iteration can be set according to the available communication resources. Sharing additional pose information between neighboring robots effectively creates overlapping optimization blocks in the underlying pose graph, which substantially reduces the number of iterations required to converge. Through extensive experiments on benchmark PGO datasets, we demonstrate the applicability and feasibility of ROBO in different initialization scenarios, using various cost functions, and under different communication regimes. We also analyze the tradeoff between the increased communication and local computation required by ROBO's overlapping blocks and the resulting faster convergence. We show that overlaps with an average inter-robot data cost of only 36 Kb per iteration can converge 3.1$\times$ faster in terms of iterations than state-of-the-art distributed PGO approaches. Furthermore, we develop an asynchronous variant of ROBO that is robust to network delays and suitable for real-world robotic applications.

  </details>



- **Radar-based Pose Optimization for HD Map Generation from Noisy Multi-Drive Vehicle Fleet Data**  
  Alexander Blumberg, Jonas Merkert, Christoph Stiller  
  _2026-03-03_ · https://arxiv.org/abs/2603.03453v1  
  <details><summary>Abstract</summary>

  High-definition (HD) maps are important for autonomous driving, but their manual generation and maintenance is very expensive. This motivates the usage of an automated map generation pipeline. Fleet vehicles provide sufficient sensors for map generation, but their measurements are less precise, introducing noise into the mapping pipeline. This work focuses on mitigating the localization noise component through aligning radar measurements in terms of raw radar point clouds of vehicle poses of different drives and performing pose graph optimization to produce a globally optimized solution between all drives present in the dataset. Improved poses are first used to generate a global radar occupancy map, aimed to facilitate precise on-vehicle localization. Through qualitative analysis we show contrast-rich feature clarity, focusing on omnipresent guardrail posts as the main feature type observable in the map. Second, the improved poses can be used as a basis for an existing lane boundary map generation pipeline, majorly improving map output compared to its original pure line detection based optimization approach.

  </details>



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


