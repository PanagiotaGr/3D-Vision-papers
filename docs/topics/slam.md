# SLAM & Localization

_Updated: 2026-02-11 07:17 UTC_

Total papers shown: **4**


---

- **Robo3R: Enhancing Robotic Manipulation with Accurate Feed-Forward 3D Reconstruction**  
  Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, Jiangmiao Pang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10101v1  
  <details><summary>Abstract</summary>

  3D spatial perception is fundamental to generalizable robotic manipulation, yet obtaining reliable, high-quality 3D geometry remains challenging. Depth sensors suffer from noise and material sensitivity, while existing reconstruction models lack the precision and metric consistency required for physical interaction. We introduce Robo3R, a feed-forward, manipulation-ready 3D reconstruction model that predicts accurate, metric-scale scene geometry directly from RGB images and robot states in real time. Robo3R jointly infers scale-invariant local geometry and relative camera poses, which are unified into the scene representation in the canonical robot frame via a learned global similarity transformation. To meet the precision demands of manipulation, Robo3R employs a masked point head for sharp, fine-grained point clouds, and a keypoint-based Perspective-n-Point (PnP) formulation to refine camera extrinsics and global alignment. Trained on Robo3R-4M, a curated large-scale synthetic dataset with four million high-fidelity annotated frames, Robo3R consistently outperforms state-of-the-art reconstruction methods and depth sensors. Across downstream tasks including imitation learning, sim-to-real transfer, grasp synthesis, and collision-free motion planning, we observe consistent gains in performance, suggesting the promise of this alternative 3D sensing module for robotic manipulation.

  </details>



- **Thegra: Graph-based SLAM for Thermal Imagery**  
  Anastasiia Kornilova, Ivan Moskalenko, Arabella Gromova, Gonzalo Ferrer, Alexander Menshchikov  
  _2026-02-09_ · https://arxiv.org/abs/2602.08531v1  
  <details><summary>Abstract</summary>

  Thermal imaging provides a practical sensing modality for visual SLAM in visually degraded environments such as low illumination, smoke, or adverse weather. However, thermal imagery often exhibits low texture, low contrast, and high noise, complicating feature-based SLAM. In this work, we propose a sparse monocular graph-based SLAM system for thermal imagery that leverages general-purpose learned features -- the SuperPoint detector and LightGlue matcher, trained on large-scale visible-spectrum data to improve cross-domain generalization. To adapt these components to thermal data, we introduce a preprocessing pipeline to enhance input suitability and modify core SLAM modules to handle sparse and outlier-prone feature matches. We further incorporate keypoint confidence scores from SuperPoint into a confidence-weighted factor graph to improve estimation robustness. Evaluations on public thermal datasets demonstrate that the proposed system achieves reliable performance without requiring dataset-specific training or fine-tuning a desired feature detector, given the scarcity of quality thermal data. Code will be made available upon publication.

  </details>



- **Graph-Loc: Robust Graph-Based LiDAR Pose Tracking with Compact Structural Map Priors under Low Observability and Occlusion**  
  Wentao Zhao, Yihe Niu, Zikun Chen, Rui Li, Yanbo Wang, Tianchen Deng, Jingchuan Wang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08417v1  
  <details><summary>Abstract</summary>

  Map-based LiDAR pose tracking is essential for long-term autonomous operation, where onboard map priors need be compact for scalable storage and fast retrieval, while online observations are often partial, repetitive, and heavily occluded. We propose Graph-Loc, a graph-based localization framework that tracks the platform pose against compact structural map priors represented as a lightweight point-line graph. Such priors can be constructed from heterogeneous sources commonly available in practice, including polygon outlines vectorized from occupancy/grid maps and CAD/model/floor-plan layouts. For each incoming LiDAR scan, Graph-Loc extracts sparse point and line primitives to form an observation graph, retrieves a pose-conditioned visible subgraph via LiDAR ray simulation, and performs scan-to-map association through unbalanced optimal transport with a local graph-context regularizer. The unbalanced formulation relaxes mass conservation, improving robustness to missing, spurious, and fragmented structures under occlusion. To enhance stability in low-observability segments, we estimate information anisotropy from the refinement normal matrix and defer updates along weakly constrained directions until sufficient constraints reappear. Experiments on public benchmarks, controlled stress tests, and real-world deployments demonstrate accurate and stable tracking with KB-level priors from heterogeneous map sources, including under geometrically degenerate and sustained occlusion and in the presence of gradual scene changes.

  </details>



- **ReRoPE: Repurposing RoPE for Relative Camera Control**  
  Chunyang Li, Yuanbo Yang, Jiahao Shao, Hongyu Zhou, Katja Schwarz, Yiyi Liao  
  _2026-02-08_ · https://arxiv.org/abs/2602.08068v1  
  <details><summary>Abstract</summary>

  Video generation with controllable camera viewpoints is essential for applications such as interactive content creation, gaming, and simulation. Existing methods typically adapt pre-trained video models using camera poses relative to a fixed reference, e.g., the first frame. However, these encodings lack shift-invariance, often leading to poor generalization and accumulated drift. While relative camera pose embeddings defined between arbitrary view pairs offer a more robust alternative, integrating them into pre-trained video diffusion models without prohibitive training costs or architectural changes remains challenging. We introduce ReRoPE, a plug-and-play framework that incorporates relative camera information into pre-trained video diffusion models without compromising their generation capability. Our approach is based on the insight that Rotary Positional Embeddings (RoPE) in existing models underutilize their full spectral bandwidth, particularly in the low-frequency components. By seamlessly injecting relative camera pose information into these underutilized bands, ReRoPE achieves precise control while preserving strong pre-trained generative priors. We evaluate our method on both image-to-video (I2V) and video-to-video (V2V) tasks in terms of camera control accuracy and visual fidelity. Our results demonstrate that ReRoPE offers a training-efficient path toward controllable, high-fidelity video generation. See project page for more results: https://sisyphe-lee.github.io/ReRoPE/

  </details>


