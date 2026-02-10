# SLAM & Localization

_Updated: 2026-02-10 07:20 UTC_

Total papers shown: **7**


---

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



- **Incremental Mapping with Measurement Synchronization & Compression**  
  Mark Griguletskii, Danil Belov, Pavel Osinenko  
  _2026-02-08_ · https://arxiv.org/abs/2602.07901v1  
  <details><summary>Abstract</summary>

  Modern autonomous vehicles and robots utilize versatile sensors for localization and mapping. The fidelity of these maps is paramount, as an accurate environmental representation is a prerequisite for stable and precise localization. Factor graphs provide a powerful approach for sensor fusion, enabling the estimation of the maximum a posteriori solution. However, the discrete nature of graph-based representations, combined with asynchronous sensor measurements, complicates consistent state estimation. The design of an optimal factor graph topology remains an open challenge, especially in multi-sensor systems with asynchronous data. Conventional approaches rely on a rigid graph structure, which becomes inefficient with sensors of disparate rates. Although preintegration techniques can mitigate this for high-rate sensors, their applicability is limited. To address this problem, this work introduces a novel approach that incrementally constructs connected factor graphs, ensuring the incorporation of all available sensor data by choosing the optimal graph topology based on the external evaluation criteria. The proposed methodology facilitates graph compression, reducing the number of nodes (optimized variables) by ~30% on average while maintaining map quality at a level comparable to conventional approaches.

  </details>



- **Research on a Camera Position Measurement Method based on a Parallel Perspective Error Transfer Model**  
  Ning Hu, Shuai Li, Jindong Tan  
  _2026-02-08_ · https://arxiv.org/abs/2602.07888v1  
  <details><summary>Abstract</summary>

  Camera pose estimation from sparse correspondences is a fundamental problem in geometric computer vision and remains particularly challenging in near-field scenarios, where strong perspective effects and heterogeneous measurement noise can significantly degrade the stability of analytic PnP solutions. In this paper, we present a geometric error propagation framework for camera pose estimation based on a parallel perspective approximation. By explicitly modeling how image measurement errors propagate through perspective geometry, we derive an error transfer model that characterizes the relationship between feature point distribution, camera depth, and pose estimation uncertainty. Building on this analysis, we develop a pose estimation method that leverages parallel perspective initialization and error-aware weighting within a Gauss-Newton optimization scheme, leading to improved robustness in proximity operations. Extensive experiments on both synthetic data and real-world images, covering diverse conditions such as strong illumination, surgical lighting, and underwater low-light environments, demonstrate that the proposed approach achieves accuracy and robustness comparable to state-of-the-art analytic and iterative PnP methods, while maintaining high computational efficiency. These results highlight the importance of explicit geometric error modeling for reliable camera pose estimation in challenging near-field settings.

  </details>



- **Looking and Listening Inside and Outside: Multimodal Artificial Intelligence Systems for Driver Safety Assessment and Intelligent Vehicle Decision-Making**  
  Ross Greer, Laura Fleig, Maitrayee Keskar, Erika Maquiling, Giovanni Tapia Lopez, Angel Martinez-Sanchez, Parthib Roy, Jake Rattigan, Mira Sur, Alejandra Vidrio, et al.  
  _2026-02-07_ · https://arxiv.org/abs/2602.07668v1  
  <details><summary>Abstract</summary>

  The looking-in-looking-out (LILO) framework has enabled intelligent vehicle applications that understand both the outside scene and the driver state to improve safety outcomes, with examples in smart airbag deployment, takeover time prediction in autonomous control transitions, and driver attention monitoring. In this research, we propose an augmentation to this framework, making a case for the audio modality as an additional source of information to understand the driver, and in the evolving autonomy landscape, also the passengers and those outside the vehicle. We expand LILO by incorporating audio signals, forming the looking-and-listening inside-and-outside (L-LIO) framework to enhance driver state assessment and environment understanding through multimodal sensor fusion. We evaluate three example cases where audio enhances vehicle safety: supervised learning on driver speech audio to classify potential impairment states (e.g., intoxication), collection and analysis of passenger natural language instructions (e.g., "turn after that red building") to motivate how spoken language can interface with planning systems through audio-aligned instruction data, and limitations of vision-only systems where audio may disambiguate the guidance and gestures of external agents. Datasets include custom-collected in-vehicle and external audio samples in real-world environments. Pilot findings show that audio yields safety-relevant insights, particularly in nuanced or context-rich scenarios where sound is critical to safe decision-making or visual signals alone are insufficient. Challenges include ambient noise interference, privacy considerations, and robustness across human subjects, motivating further work on reliability in dynamic real-world contexts. L-LIO augments driver and scene understanding through multimodal fusion of audio and visual sensing, offering new paths for safety intervention.

  </details>



- **Thermal odometry and dense mapping using learned ddometry and Gaussian splatting**  
  Tianhao Zhou, Yujia Chen, Zhihao Zhan, Yuhang Ming, Jianzhu Huai  
  _2026-02-07_ · https://arxiv.org/abs/2602.07493v1  
  <details><summary>Abstract</summary>

  Thermal infrared sensors, with wavelengths longer than smoke particles, can capture imagery independent of darkness, dust, and smoke. This robustness has made them increasingly valuable for motion estimation and environmental perception in robotics, particularly in adverse conditions. Existing thermal odometry and mapping approaches, however, are predominantly geometric and often fail across diverse datasets while lacking the ability to produce dense maps. Motivated by the efficiency and high-quality reconstruction ability of recent Gaussian Splatting (GS) techniques, we propose TOM-GS, a thermal odometry and mapping method that integrates learning-based odometry with GS-based dense mapping. TOM-GS is among the first GS-based SLAM systems tailored for thermal cameras, featuring dedicated thermal image enhancement and monocular depth integration. Extensive experiments on motion estimation and novel-view rendering demonstrate that TOM-GS outperforms existing learning-based methods, confirming the benefits of learning-based pipelines for robust thermal odometry and dense reconstruction.

  </details>


