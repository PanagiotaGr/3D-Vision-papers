# SLAM & Localization

_Updated: 2026-03-07 06:57 UTC_

Total papers shown: **11**


---

- **Loop Closure via Maximal Cliques in 3D LiDAR-Based SLAM**  
  Javier Laserna, Saurabh Gupta, Oscar Martinez Mozos, Cyrill Stachniss, Pablo San Segundo  
  _2026-03-05_ · https://arxiv.org/abs/2603.05397v1  
  <details><summary>Abstract</summary>

  Reliable loop closure detection remains a critical challenge in 3D LiDAR-based SLAM, especially under sensor noise, environmental ambiguity, and viewpoint variation conditions. RANSAC is often used in the context of loop closures for geometric model fitting in the presence of outliers. However, this approach may fail, leading to map inconsistency. We introduce a novel deterministic algorithm, CliReg, for loop closure validation that replaces RANSAC verification with a maximal clique search over a compatibility graph of feature correspondences. This formulation avoids random sampling and increases robustness in the presence of noise and outliers. We integrated our approach into a real- time pipeline employing binary 3D descriptors and a Hamming distance embedding binary search tree-based matching. We evaluated it on multiple real-world datasets featuring diverse LiDAR sensors. The results demonstrate that our proposed technique consistently achieves a lower pose error and more reliable loop closures than RANSAC, especially in sparse or ambiguous conditions. Additional experiments on 2D projection-based maps confirm its generality across spatial domains, making our approach a robust and efficient alternative for loop closure detection.

  </details>



- **Dark3R: Learning Structure from Motion in the Dark**  
  Andrew Y Guo, Anagh Malik, SaiKiran Tedla, Yutong Dai, Yiqian Qin, Zach Salehe, Benjamin Attal, Sotiris Nousias, Kyros Kutulakos, David B. Lindell  
  _2026-03-05_ · https://arxiv.org/abs/2603.05330v1  
  <details><summary>Abstract</summary>

  We introduce Dark3R, a framework for structure from motion in the dark that operates directly on raw images with signal-to-noise ratios (SNRs) below $-4$ dB -- a regime where conventional feature- and learning-based methods break down. Our key insight is to adapt large-scale 3D foundation models to extreme low-light conditions through a teacher--student distillation process, enabling robust feature matching and camera pose estimation in low light. Dark3R requires no 3D supervision; it is trained solely on noisy--clean raw image pairs, which can be either captured directly or synthesized using a simple Poisson--Gaussian noise model applied to well-exposed raw images. To train and evaluate our approach, we introduce a new, exposure-bracketed dataset that includes $\sim$42,000 multi-view raw images with ground-truth 3D annotations, and we demonstrate that Dark3R achieves state-of-the-art structure from motion in the low-SNR regime. Further, we demonstrate state-of-the-art novel view synthesis in the dark using Dark3R's predicted poses and a coarse-to-fine radiance field optimization procedure.

  </details>



- **AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model**  
  Jinwoo Jeon, Dong-Uk Seo, Eungchang Mason Lee, Hyun Myung  
  _2026-03-05_ · https://arxiv.org/abs/2603.05097v1  
  <details><summary>Abstract</summary>

  Recent advances in geometric foundation models have emerged as a promising alternative for addressing the challenge of dense reconstruction in monocular visual simultaneous localization and mapping (SLAM). Although geometric foundation models enable SLAM to leverage variable input views, the previous methods remain confined to two-view pairs or fixed-length inputs without sufficient deliberation of geometric context for view selection. To tackle this problem, we propose AIM-SLAM, a dense monocular SLAM framework that exploits an adaptive and informative multi-view keyframe prioritization with dense pointmap predictions from visual geometry grounded transformer (VGGT). Specifically, we introduce the selective information- and geometric-aware multi-view adaptation (SIGMA) module, which employs voxel overlap and information gain to retrieve a candidate set of keyframes and adaptively determine its size. Furthermore, we formulate a joint multi-view Sim(3) optimization that enforces consistent alignment across selected views, substantially improving pose estimation accuracy. The effectiveness of AIM-SLAM is demonstrated on real-world datasets, where it achieves state-of-the-art performance in both pose estimation and dense reconstruction. Our system supports ROS integration, with code is available at https://aimslam.github.io/.

  </details>



- **MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer**  
  Juntong Fang, Zequn Chen, Weiqi Zhang, Donglin Di, Xuancheng Zhang, Chengmin Yang, Yu-Shen Liu  
  _2026-03-05_ · https://arxiv.org/abs/2603.05078v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic 4D scenes remains challenging due to the presence of moving objects that corrupt camera pose estimation. Existing optimization methods alleviate this issue with additional supervision, but they are mostly computationally expensive and impractical in real-time applications. To address these limitations, we propose MoRe, a feedforward 4D reconstruction network that efficiently recovers dynamic 3D scenes from monocular videos. Built upon a strong static reconstruction backbone, MoRe employs an attention-forcing strategy to disentangle dynamic motion from static structure. To further enhance robustness, we fine-tune the model on large-scale, diverse datasets encompassing both dynamic and static scenes. Moreover, our grouped causal attention captures temporal dependencies and adapts to varying token lengths across frames, ensuring temporally coherent geometry reconstruction. Extensive experiments on multiple benchmarks demonstrate that MoRe achieves high-quality dynamic reconstructions with exceptional efficiency.

  </details>



- **VinePT-Map: Pole-Trunk Semantic Mapping for Resilient Autonomous Robotics in Vineyards**  
  Giorgio Audrito, Mauro Martini, Alessandro Navone, Giorgia Galluzzo, Marcello Chiaberge  
  _2026-03-05_ · https://arxiv.org/abs/2603.05070v1  
  <details><summary>Abstract</summary>

  Reliable long-term deployment of autonomous robots in agricultural environments remains challenging due to perceptual aliasing, seasonal variability, and the dynamic nature of crop canopies. Vineyards, characterized by repetitive row structures and significant visual changes across phenological stages, represent a pivotal field challenge, limiting the robustness of conventional feature-based localization and mapping approaches. This paper introduces VinePT-Map, a semantic mapping framework that leverages vine trunks and support poles as persistent structural landmarks to enable season-agnostic and resilient robot localization. The proposed method formulates the mapping problem as a factor graph, integrating GPS, IMU, and RGB-D observations through robust geometrical constraints that exploit vineyard structure. An efficient perception pipeline based on instance segmentation and tracking, combined with a clustering filter for outlier rejection and pose refinement, enables accurate landmark detection using low-cost sensors and onboard computation. To validate the pipeline, we present a multi-season dataset for trunk and pole segmentation and tracking. Extensive field experiments conducted across diverse seasons demonstrate the robustness and accuracy of the proposed approach, highlighting its suitability for long-term autonomous operation in agricultural environments.

  </details>



- **Distributed State Estimation for Vision-Based Cooperative Slung Load Transportation in GPS-Denied Environments**  
  Jack R. Pence, Jackson Fezell, Jack W. Langelaan, Junyi Geng  
  _2026-03-04_ · https://arxiv.org/abs/2603.04571v1  
  <details><summary>Abstract</summary>

  Transporting heavy or oversized slung loads using rotorcraft has traditionally relied on single-aircraft systems, which limits both payload capacity and control authority. Cooperative multilift using teams of rotorcraft offers a scalable and efficient alternative, especially for infrequent but challenging "long-tail" payloads without the need of building larger and larger rotorcraft. Most prior multilift research assumes GPS availability, uses centralized estimation architectures, or relies on controlled laboratory motion-capture setups. As a result, these methods lack robustness to sensor loss and are not viable in GPS-denied or operationally constrained environments. This paper addresses this limitation by presenting a distributed and decentralized payload state estimation framework for vision-based multilift operations. Using onboard monocular cameras, each UAV detects a fiducial marker on the payload and estimates its relative pose. These measurements are fused via a Distributed and Decentralized Extended Information Filter (DDEIF), enabling robust and scalable estimation that is resilient to individual sensor dropouts. This payload state estimate is then used for closed-loop trajectory tracking control. Monte Carlo simulation results in Gazebo show the effectiveness of the proposed approach, including the effect of communication loss during flight.

  </details>



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


