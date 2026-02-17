# 3D Reconstruction

_Updated: 2026-02-17 07:13 UTC_

Total papers shown: **12**


---

- **AnchorWeave: World-Consistent Video Generation with Retrieved Local Spatial Memories**  
  Zun Wang, Han Lin, Jaehong Yoon, Jaemin Cho, Yue Zhang, Mohit Bansal  
  _2026-02-16_ · https://arxiv.org/abs/2602.14941v1  
  <details><summary>Abstract</summary>

  Maintaining spatial world consistency over long horizons remains a central challenge for camera-controllable video generation. Existing memory-based approaches often condition generation on globally reconstructed 3D scenes by rendering anchor videos from the reconstructed geometry in the history. However, reconstructing a global 3D scene from multiple views inevitably introduces cross-view misalignment, as pose and depth estimation errors cause the same surfaces to be reconstructed at slightly different 3D locations across views. When fused, these inconsistencies accumulate into noisy geometry that contaminates the conditioning signals and degrades generation quality. We introduce AnchorWeave, a memory-augmented video generation framework that replaces a single misaligned global memory with multiple clean local geometric memories and learns to reconcile their cross-view inconsistencies. To this end, AnchorWeave performs coverage-driven local memory retrieval aligned with the target trajectory and integrates the selected local memories through a multi-anchor weaving controller during generation. Extensive experiments demonstrate that AnchorWeave significantly improves long-term scene consistency while maintaining strong visual quality, with ablation and analysis studies further validating the effectiveness of local geometric conditioning, multi-anchor control, and coverage-driven retrieval.

  </details>



- **Wrivinder: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery**  
  Chandrakanth Gudavalli, Tajuddin Manhar Mohammed, Abhay Yadav, Ananth Vishnu Bhaskar, Hardik Prajapati, Cheng Peng, Rama Chellappa, Shivkumar Chandrasekaran, B. S. Manjunath  
  _2026-02-16_ · https://arxiv.org/abs/2602.14929v1  
  <details><summary>Abstract</summary>

  Aligning ground-level imagery with geo-registered satellite maps is crucial for mapping, navigation, and situational awareness, yet remains challenging under large viewpoint gaps or when GPS is unreliable. We introduce Wrivinder, a zero-shot, geometry-driven framework that aggregates multiple ground photographs to reconstruct a consistent 3D scene and align it with overhead satellite imagery. Wrivinder combines SfM reconstruction, 3D Gaussian Splatting, semantic grounding, and monocular depth--based metric cues to produce a stable zenith-view rendering that can be directly matched to satellite context for metrically accurate camera geo-localization. To support systematic evaluation of this task, which lacks suitable benchmarks, we also release MC-Sat, a curated dataset linking multi-view ground imagery with geo-registered satellite tiles across diverse outdoor environments. Together, Wrivinder and MC-Sat provide a first comprehensive baseline and testbed for studying geometry-centered cross-view alignment without paired supervision. In zero-shot experiments, Wrivinder achieves sub-30\,m geolocation accuracy across both dense and large-area scenes, highlighting the promise of geometry-based aggregation for robust ground-to-satellite localization.

  </details>



- **Cross-view Domain Generalization via Geometric Consistency for LiDAR Semantic Segmentation**  
  Jindong Zhao, Yuan Gao, Yang Xia, Sheng Nie, Jun Yue, Weiwei Sun, Shaobo Xia  
  _2026-02-16_ · https://arxiv.org/abs/2602.14525v1  
  <details><summary>Abstract</summary>

  Domain-generalized LiDAR semantic segmentation (LSS) seeks to train models on source-domain point clouds that generalize reliably to multiple unseen target domains, which is essential for real-world LiDAR applications. However, existing approaches assume similar acquisition views (e.g., vehicle-mounted) and struggle in cross-view scenarios, where observations differ substantially due to viewpoint-dependent structural incompleteness and non-uniform point density. Accordingly, we formulate cross-view domain generalization for LiDAR semantic segmentation and propose a novel framework, termed CVGC (Cross-View Geometric Consistency). Specifically, we introduce a cross-view geometric augmentation module that models viewpoint-induced variations in visibility and sampling density, generating multiple cross-view observations of the same scene. Subsequently, a geometric consistency module enforces consistent semantic and occupancy predictions across geometrically augmented point clouds of the same scene. Extensive experiments on six public LiDAR datasets establish the first systematic evaluation of cross-view domain generalization for LiDAR semantic segmentation, demonstrating that CVGC consistently outperforms state-of-the-art methods when generalizing from a single source domain to multiple target domains with heterogeneous acquisition viewpoints. The source code will be publicly available at https://github.com/KintomZi/CVGC-DG

  </details>



- **Gaussian Mesh Renderer for Lightweight Differentiable Rendering**  
  Xinpeng Liu, Fumio Okura  
  _2026-02-16_ · https://arxiv.org/abs/2602.14493v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has enabled high-fidelity virtualization with fast rendering and optimization for novel view synthesis. On the other hand, triangle mesh models still remain a popular choice for surface reconstruction but suffer from slow or heavy optimization in traditional mesh-based differentiable renderers. To address this problem, we propose a new lightweight differentiable mesh renderer leveraging the efficient rasterization process of 3DGS, named Gaussian Mesh Renderer (GMR), which tightly integrates the Gaussian and mesh representations. Each Gaussian primitive is analytically derived from the corresponding mesh triangle, preserving structural fidelity and enabling the gradient flow. Compared to the traditional mesh renderers, our method achieves smoother gradients, which especially contributes to better optimization using smaller batch sizes with limited memory. Our implementation is available in the public GitHub repository at https://github.com/huntorochi/Gaussian-Mesh-Renderer.

  </details>



- **Learning Significant Persistent Homology Features for 3D Shape Understanding**  
  Prachi Kudeshia, Jiju Poovvancheri  
  _2026-02-15_ · https://arxiv.org/abs/2602.14228v1  
  <details><summary>Abstract</summary>

  Geometry and topology constitute complementary descriptors of three-dimensional shape, yet existing benchmark datasets primarily capture geometric information while neglecting topological structure. This work addresses this limitation by introducing topologically-enriched versions of ModelNet40 and ShapeNet, where each point cloud is augmented with its corresponding persistent homology features. These benchmarks with the topological signatures establish a foundation for unified geometry-topology learning and enable systematic evaluation of topology-aware deep learning architectures for 3D shape analysis. Building on this foundation, we propose a deep learning-based significant persistent point selection method, \textit{TopoGAT}, that learns to identify the most informative topological features directly from input data and the corresponding topological signatures, circumventing the limitations of hand-crafted statistical selection criteria. A comparative study verifies the superiority of the proposed method over traditional statistical approaches in terms of stability and discriminative power. Integrating the selected significant persistent points into standard point cloud classification and part-segmentation pipelines yields improvements in both classification accuracy and segmentation metrics. The presented topologically-enriched datasets, coupled with our learnable significant feature selection approach, enable the broader integration of persistent homology into the practical deep learning workflows for 3D point cloud analysis.

  </details>



- **Learnable Multi-level Discrete Wavelet Transforms for 3D Gaussian Splatting Frequency Modulation**  
  Hung Nguyen, An Le, Truong Nguyen  
  _2026-02-15_ · https://arxiv.org/abs/2602.14199v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful approach for novel view synthesis. However, the number of Gaussian primitives often grows substantially during training as finer scene details are reconstructed, leading to increased memory and storage costs. Recent coarse-to-fine strategies regulate Gaussian growth by modulating the frequency content of the ground-truth images. In particular, AutoOpti3DGS employs the learnable Discrete Wavelet Transform (DWT) to enable data-adaptive frequency modulation. Nevertheless, its modulation depth is limited by the 1-level DWT, and jointly optimizing wavelet regularization with 3D reconstruction introduces gradient competition that promotes excessive Gaussian densification. In this paper, we propose a multi-level DWT-based frequency modulation framework for 3DGS. By recursively decomposing the low-frequency subband, we construct a deeper curriculum that provides progressively coarser supervision during early training, consistently reducing Gaussian counts. Furthermore, we show that the modulation can be performed using only a single scaling parameter, rather than learning the full 2-tap high-pass filter. Experimental results on standard benchmarks demonstrate that our method further reduces Gaussian counts while maintaining competitive rendering quality.

  </details>



- **Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation**  
  Yue Chen, Muqing Jiang, Kaifeng Zheng, Jiaqi Liang, Chenrui Tie, Haoran Lu, Ruihai Wu, Hao Dong  
  _2026-02-15_ · https://arxiv.org/abs/2602.14193v1  
  <details><summary>Abstract</summary>

  Articulated object manipulation is essential for various real-world robotic tasks, yet generalizing across diverse objects remains a major challenge. A key to generalization lies in understanding functional parts (e.g., door handles and knobs), which indicate where and how to manipulate across diverse object categories and shapes. Previous works attempted to achieve generalization by introducing foundation features, while these features are mostly 2D-based and do not specifically consider functional parts. When lifting these 2D features to geometry-profound 3D space, challenges arise, such as long runtimes, multi-view inconsistencies, and low spatial resolution with insufficient geometric information. To address these issues, we propose Part-Aware 3D Feature Field (PA3FF), a novel dense 3D feature with part awareness for generalizable articulated object manipulation. PA3FF is trained by 3D part proposals from a large-scale labeled dataset, via a contrastive learning formulation. Given point clouds as input, PA3FF predicts a continuous 3D feature field in a feedforward manner, where the distance between point features reflects the proximity of functional parts: points with similar features are more likely to belong to the same part. Building on this feature, we introduce the Part-Aware Diffusion Policy (PADP), an imitation learning framework aimed at enhancing sample efficiency and generalization for robotic manipulation. We evaluate PADP on several simulated and real-world tasks, demonstrating that PA3FF consistently outperforms a range of 2D and 3D representations in manipulation scenarios, including CLIP, DINOv2, and Grounded-SAM. Beyond imitation learning, PA3FF enables diverse downstream methods, including correspondence learning and segmentation tasks, making it a versatile foundation for robotic manipulation. Project page: https://pa3ff.github.io

  </details>



- **DenseMLLM: Standard Multimodal LLMs are Intrinsic Dense Predictors**  
  Yi Li, Hongze Shen, Lexiang Tang, Xin Li, Xinpeng Ding, Yinsong Liu, Deqiang Jiang, Xing Sun, Xiaomeng Li  
  _2026-02-15_ · https://arxiv.org/abs/2602.14134v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in high-level visual understanding. However, extending these models to fine-grained dense prediction tasks, such as semantic segmentation and depth estimation, typically necessitates the incorporation of complex, task-specific decoders and other customizations. This architectural fragmentation increases model complexity and deviates from the generalist design of MLLMs, ultimately limiting their practicality. In this work, we challenge this paradigm by accommodating standard MLLMs to perform dense predictions without requiring additional task-specific decoders. The proposed model is called DenseMLLM, grounded in the standard architecture with a novel vision token supervision strategy for multiple labels and tasks. Despite its minimalist design, our model achieves highly competitive performance across a wide range of dense prediction and vision-language benchmarks, demonstrating that a standard, general-purpose MLLM can effectively support dense perception without architectural specialization.

  </details>



- **GeoFusionLRM: Geometry-Aware Self-Correction for Consistent 3D Reconstruction**  
  Ahmet Burak Yildirim, Tuna Saygin, Duygu Ceylan, Aysegul Dundar  
  _2026-02-15_ · https://arxiv.org/abs/2602.14119v1  
  <details><summary>Abstract</summary>

  Single-image 3D reconstruction with large reconstruction models (LRMs) has advanced rapidly, yet reconstructions often exhibit geometric inconsistencies and misaligned details that limit fidelity. We introduce GeoFusionLRM, a geometry-aware self-correction framework that leverages the model's own normal and depth predictions to refine structural accuracy. Unlike prior approaches that rely solely on features extracted from the input image, GeoFusionLRM feeds back geometric cues through a dedicated transformer and fusion module, enabling the model to correct errors and enforce consistency with the conditioning image. This design improves the alignment between the reconstructed mesh and the input views without additional supervision or external signals. Extensive experiments demonstrate that GeoFusionLRM achieves sharper geometry, more consistent normals, and higher fidelity than state-of-the-art LRM baselines.

  </details>



- **High-fidelity 3D reconstruction for planetary exploration**  
  Alfonso Martínez-Petersen, Levin Gerdes, David Rodríguez-Martínez, C. J. Pérez-del-Pulgar  
  _2026-02-14_ · https://arxiv.org/abs/2602.13909v1  
  <details><summary>Abstract</summary>

  Planetary exploration increasingly relies on autonomous robotic systems capable of perceiving, interpreting, and reconstructing their surroundings in the absence of global positioning or real-time communication with Earth. Rovers operating on planetary surfaces must navigate under sever environmental constraints, limited visual redundancy, and communication delays, making onboard spatial awareness and visual localization key components for mission success. Traditional techniques based on Structure-from-Motion (SfM) and Simultaneous Localization and Mapping (SLAM) provide geometric consistency but struggle to capture radiometric detail or to scale efficiently in unstructured, low-texture terrains typical of extraterrestrial environments. This work explores the integration of radiance field-based methods - specifically Neural Radiance Fields (NeRF) and Gaussian Splatting - into a unified, automated environment reconstruction pipeline for planetary robotics. Our system combines the Nerfstudio and COLMAP frameworks with a ROS2-compatible workflow capable of processing raw rover data directly from rosbag recordings. This approach enables the generation of dense, photorealistic, and metrically consistent 3D representations from minimal visual input, supporting improved perception and planning for autonomous systems operating in planetary-like conditions. The resulting pipeline established a foundation for future research in radiance field-based mapping, bridging the gap between geometric and neural representations in planetary exploration.

  </details>



- **Synthetic Dataset Generation and Validation for Robotic Surgery Instrument Segmentation**  
  Giorgio Chiesa, Rossella Borra, Vittorio Lauro, Sabrina De Cillis, Daniele Amparore, Cristian Fiori, Riccardo Renzulli, Marco Grangetto  
  _2026-02-14_ · https://arxiv.org/abs/2602.13844v1  
  <details><summary>Abstract</summary>

  This paper presents a comprehensive workflow for generating and validating a synthetic dataset designed for robotic surgery instrument segmentation. A 3D reconstruction of the Da Vinci robotic arms was refined and animated in Autodesk Maya through a fully automated Python-based pipeline capable of producing photorealistic, labeled video sequences. Each scene integrates randomized motion patterns, lighting variations, and synthetic blood textures to mimic intraoperative variability while preserving pixel-accurate ground truth masks. To validate the realism and effectiveness of the generated data, several segmentation models were trained under controlled ratios of real and synthetic data. Results demonstrate that a balanced composition of real and synthetic samples significantly improves model generalization compared to training on real data only, while excessive reliance on synthetic data introduces a measurable domain shift. The proposed framework provides a reproducible and scalable tool for surgical computer vision, supporting future research in data augmentation, domain adaptation, and simulation-based pretraining for robotic-assisted surgery. Data and code are available at https://github.com/EIDOSLAB/Sintetic-dataset-DaVinci.

  </details>



- **Joint Orientation and Weight Optimization for Robust Watertight Surface Reconstruction via Dirichlet-Regularized Winding Fields**  
  Jiaze Li, Daisheng Jin, Fei Hou, Junhui Hou, Zheng Liu, Shiqing Xin, Wenping Wang, Ying He  
  _2026-02-14_ · https://arxiv.org/abs/2602.13801v1  
  <details><summary>Abstract</summary>

  We propose Dirichlet Winding Reconstruction (DiWR), a robust method for reconstructing watertight surfaces from unoriented point clouds with non-uniform sampling, noise, and outliers. Our method uses the generalized winding number (GWN) field as the target implicit representation and jointly optimizes point orientations, per-point area weights, and confidence coefficients in a single pipeline. The optimization minimizes the Dirichlet energy of the induced winding field together with additional GWN-based constraints, allowing DiWR to compensate for non-uniform sampling, reduce the impact of noise, and downweight outliers during reconstruction, with no reliance on separate preprocessing. We evaluate DiWR on point clouds from 3D Gaussian Splatting, a computer-vision pipeline, and corrupted graphics benchmarks. Experiments show that DiWR produces plausible watertight surfaces on these challenging inputs and outperforms both traditional multi-stage pipelines and recent joint orientation-reconstruction methods.

  </details>


