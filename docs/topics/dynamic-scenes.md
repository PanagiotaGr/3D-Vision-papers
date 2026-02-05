# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-05 07:14 UTC_

Total papers shown: **8**


---

- **Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis**  
  Seong-Eun Hong, JaeYoung Seon, JuYeong Hwang, JongHwan Shin, HyeongYeop Kang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04292v1  
  <details><summary>Abstract</summary>

  Text-to-motion generation has advanced with diffusion models, yet existing systems often collapse complex multi-action prompts into a single embedding, leading to omissions, reordering, or unnatural transitions. In this work, we shift perspective by introducing a principled definition of an event as the smallest semantically self-contained action or state change in a text prompt that can be temporally aligned with a motion segment. Building on this definition, we propose Event-T2M, a diffusion-based framework that decomposes prompts into events, encodes each with a motion-aware retrieval model, and integrates them through event-based cross-attention in Conformer blocks. Existing benchmarks mix simple and multi-event prompts, making it unclear whether models that succeed on single actions generalize to multi-action cases. To address this, we construct HumanML3D-E, the first benchmark stratified by event count. Experiments on HumanML3D, KIT-ML, and HumanML3D-E show that Event-T2M matches state-of-the-art baselines on standard tests while outperforming them as event complexity increases. Human studies validate the plausibility of our event definition, the reliability of HumanML3D-E, and the superiority of Event-T2M in generating multi-event motions that preserve order and naturalness close to ground-truth. These results establish event-level conditioning as a generalizable principle for advancing text-to-motion generation beyond single-action prompts.

  </details>



- **SkeletonGaussian: Editable 4D Generation through Gaussian Skeletonization**  
  Lifan Wu, Ruijie Zhu, Yubo Ai, Tianzhu Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04271v1  
  <details><summary>Abstract</summary>

  4D generation has made remarkable progress in synthesizing dynamic 3D objects from input text, images, or videos. However, existing methods often represent motion as an implicit deformation field, which limits direct control and editability. To address this issue, we propose SkeletonGaussian, a novel framework for generating editable dynamic 3D Gaussians from monocular video input. Our approach introduces a hierarchical articulated representation that decomposes motion into sparse rigid motion explicitly driven by a skeleton and fine-grained non-rigid motion. Concretely, we extract a robust skeleton and drive rigid motion via linear blend skinning, followed by a hexplane-based refinement for non-rigid deformations, enhancing interpretability and editability. Experimental results demonstrate that SkeletonGaussian surpasses existing methods in generation quality while enabling intuitive motion editing, establishing a new paradigm for editable 4D generation. Project page: https://wusar.github.io/projects/skeletongaussian/

  </details>



- **Occlusion-Free Conformal Lensing for Spatiotemporal Visualization in 3D Urban Analytics**  
  Roberta Mota, Julio D. Silva, Fabio Miranda, Usman Alim, Ehud Sharlin, Nivan Ferreira  
  _2026-02-03_ · https://arxiv.org/abs/2602.03743v1  
  <details><summary>Abstract</summary>

  The visualization of temporal data on urban buildings, such as shadows, noise, and solar potential, plays a critical role in the analysis of dynamic urban phenomena. However, in dense and geographically constrained 3D urban environments, visual representations of time-varying building data often suffer from occlusion and visual clutter. To address these two challenges, we introduce an immersive lens visualization that integrates a view-dependent cutaway de-occlusion technique and a temporal display derived from a conformal mapping algorithm. The mapping process first partitions irregular building footprints into smaller, sufficiently regular subregions that serve as structural primitives. These subregions are then seamlessly recombined to form a conformal, layered layout for our temporal lens visualization. The view-responsive cutaway is inspired by traditional architectural illustrations, preserving the overall layout of the building and its surroundings to maintain users' sense of spatial orientation. This lens design enables the occlusion-free embedding of shape-adaptive temporal displays across building facades on demand, supporting rapid time-space association for the discovery, access and interpretation of spatiotemporal urban patterns. Guided by domain and design goals, we outline the rationale behind the lens visual and interaction design choices, such as the encoding of time progression and temporal values in the conforming lens image. A user study compares our approach against conventional juxtaposition and x-ray spatiotemporal designs. Results validate the usage and utility of our lens, showing that it improves task accuracy and completion time, reduces navigation effort, and increases user confidence. From these findings, we distill design recommendations and promising directions for future research on spatially-embedded lenses in 3D visualization and urban analytics.

  </details>



- **Constrained Dynamic Gaussian Splatting**  
  Zihan Zheng, Zhenglong Wu, Xuanxuan Wang, Houqiang Zhong, Xiaoyun Zhang, Qiang Hu, Guangtao Zhai, Wenjun Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03538v1  
  <details><summary>Abstract</summary>

  While Dynamic Gaussian Splatting enables high-fidelity 4D reconstruction, its deployment is severely hindered by a fundamental dilemma: unconstrained densification leads to excessive memory consumption incompatible with edge devices, whereas heuristic pruning fails to achieve optimal rendering quality under preset Gaussian budgets. In this work, we propose Constrained Dynamic Gaussian Splatting (CDGS), a novel framework that formulates dynamic scene reconstruction as a budget-constrained optimization problem to enforce a strict, user-defined Gaussian budget during training. Our key insight is to introduce a differentiable budget controller as the core optimization driver. Guided by a multi-modal unified importance score, this controller fuses geometric, motion, and perceptual cues for precise capacity regulation. To maximize the utility of this fixed budget, we further decouple the optimization of static and dynamic elements, employing an adaptive allocation mechanism that dynamically distributes capacity based on motion complexity. Furthermore, we implement a three-phase training strategy to seamlessly integrate these constraints, ensuring precise adherence to the target count. Coupled with a dual-mode hybrid compression scheme, CDGS not only strictly adheres to hardware constraints (error < 2%}) but also pushes the Pareto frontier of rate-distortion performance. Extensive experiments demonstrate that CDGS delivers optimal rendering quality under varying capacity limits, achieving over 3x compression compared to state-of-the-art methods.

  </details>



- **From Single Scan to Sequential Consistency: A New Paradigm for LIDAR Relocalization**  
  Minghang Zhu, Zhijing Wang, Yuxin Guo, Wen Li, Sheng Ao, Cheng Wang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03198v1  
  <details><summary>Abstract</summary>

  LiDAR relocalization aims to estimate the global 6-DoF pose of a sensor in the environment. However, existing regression-based approaches are prone to dynamic or ambiguous scenarios, as they either solely rely on single-frame inference or neglect the spatio-temporal consistency across scans. In this paper, we propose TempLoc, a new LiDAR relocalization framework that enhances the robustness of localization by effectively modeling sequential consistency. Specifically, a Global Coordinate Estimation module is first introduced to predict point-wise global coordinates and associated uncertainties for each LiDAR scan. A Prior Coordinate Generation module is then presented to estimate inter-frame point correspondences by the attention mechanism. Lastly, an Uncertainty-Guided Coordinate Fusion module is deployed to integrate both predictions of point correspondence in an end-to-end fashion, yielding a more temporally consistent and accurate global 6-DoF pose. Experimental results on the NCLT and Oxford Robot-Car benchmarks show that our TempLoc outperforms stateof-the-art methods by a large margin, demonstrating the effectiveness of temporal-aware correspondence modeling in LiDAR relocalization. Our code will be released soon.

  </details>



- **4DPC$^2$hat: Towards Dynamic Point Cloud Understanding with Failure-Aware Bootstrapping**  
  Xindan Zhang, Weilong Yan, Yufei Shi, Xuerui Qiu, Tao He, Ying Li, Ming Li, Hehe Fan  
  _2026-02-03_ · https://arxiv.org/abs/2602.03890v1  
  <details><summary>Abstract</summary>

  Point clouds provide a compact and expressive representation of 3D objects, and have recently been integrated into multimodal large language models (MLLMs). However, existing methods primarily focus on static objects, while understanding dynamic point cloud sequences remains largely unexplored. This limitation is mainly caused by the lack of large-scale cross-modal datasets and the difficulty of modeling motions in spatio-temporal contexts. To bridge this gap, we present 4DPC$^2$hat, the first MLLM tailored for dynamic point cloud understanding. To this end, we construct a large-scale cross-modal dataset 4DPC$^2$hat-200K via a meticulous two-stage pipeline consisting of topology-consistent 4D point construction and two-level captioning. The dataset contains over 44K dynamic object sequences, 700K point cloud frames, and 200K curated question-answer (QA) pairs, supporting inquiries about counting, temporal relationship, action, spatial relationship, and appearance. At the core of the framework, we introduce a Mamba-enhanced temporal reasoning MLLM to capture long-range dependencies and dynamic patterns among a point cloud sequence. Furthermore, we propose a failure-aware bootstrapping learning strategy that iteratively identifies model deficiencies and generates targeted QA supervision to continuously strengthen corresponding reasoning capabilities. Extensive experiments demonstrate that our 4DPC$^2$hat significantly improves action understanding and temporal reasoning compared with existing models, establishing a strong foundation for 4D dynamic point cloud understanding.

  </details>



- **JRDB-Pose3D: A Multi-person 3D Human Pose and Shape Estimation Dataset for Robotics**  
  Sandika Biswas, Kian Izadpanah, Hamid Rezatofighi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03064v1  
  <details><summary>Abstract</summary>

  Real-world scenes are inherently crowded. Hence, estimating 3D poses of all nearby humans, tracking their movements over time, and understanding their activities within social and environmental contexts are essential for many applications, such as autonomous driving, robot perception, robot navigation, and human-robot interaction. However, most existing 3D human pose estimation datasets primarily focus on single-person scenes or are collected in controlled laboratory environments, which restricts their relevance to real-world applications. To bridge this gap, we introduce JRDB-Pose3D, which captures multi-human indoor and outdoor environments from a mobile robotic platform. JRDB-Pose3D provides rich 3D human pose annotations for such complex and dynamic scenes, including SMPL-based pose annotations with consistent body-shape parameters and track IDs for each individual over time. JRDB-Pose3D contains, on average, 5-10 human poses per frame, with some scenes featuring up to 35 individuals simultaneously. The proposed dataset presents unique challenges, including frequent occlusions, truncated bodies, and out-of-frame body parts, which closely reflect real-world environments. Moreover, JRDB-Pose3D inherits all available annotations from the JRDB dataset, such as 2D pose, information about social grouping, activities, and interactions, full-scene semantic masks with consistent human- and object-level tracking, and detailed annotations for each individual, such as age, gender, and race, making it a holistic dataset for a wide range of downstream perception and human-centric understanding tasks.

  </details>



- **SharpTimeGS: Sharp and Stable Dynamic Gaussian Splatting via Lifespan Modulation**  
  Zhanfeng Liao, Jiajun Zhang, Hanzhang Tu, Zhixi Wang, Yunqi Gao, Hongwen Zhang, Yebin Liu  
  _2026-02-03_ · https://arxiv.org/abs/2602.02989v1  
  <details><summary>Abstract</summary>

  Novel view synthesis of dynamic scenes is fundamental to achieving photorealistic 4D reconstruction and immersive visual experiences. Recent progress in Gaussian-based representations has significantly improved real-time rendering quality, yet existing methods still struggle to maintain a balance between long-term static and short-term dynamic regions in both representation and optimization. To address this, we present SharpTimeGS, a lifespan-aware 4D Gaussian framework that achieves temporally adaptive modeling of both static and dynamic regions under a unified representation. Specifically, we introduce a learnable lifespan parameter that reformulates temporal visibility from a Gaussian-shaped decay into a flat-top profile, allowing primitives to remain consistently active over their intended duration and avoiding redundant densification. In addition, the learned lifespan modulates each primitives' motion, reducing drift in long-lived static points while retaining unrestricted motion for short-lived dynamic ones. This effectively decouples motion magnitude from temporal duration, improving long-term stability without compromising dynamic fidelity. Moreover, we design a lifespan-velocity-aware densification strategy that mitigates optimization imbalance between static and dynamic regions by allocating more capacity to regions with pronounced motion while keeping static areas compact and stable. Extensive experiments on multiple benchmarks demonstrate that our method achieves state-of-the-art performance while supporting real-time rendering up to 4K resolution at 100 FPS on one RTX 4090.

  </details>


