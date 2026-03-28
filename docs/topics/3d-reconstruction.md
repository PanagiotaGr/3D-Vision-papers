# 3D Reconstruction

_Updated: 2026-03-28 07:11 UTC_

Total papers shown: **11**


---

- **DC-Reg: Globally Optimal Point Cloud Registration via Tight Bounding with Difference of Convex Programming**  
  Wei Lian, Fei Ma, Hang Pan, Zhesen Cui, Wangmeng Zuo  
  _2026-03-26_ · https://arxiv.org/abs/2603.25442v1  
  <details><summary>Abstract</summary>

  Achieving globally optimal point cloud registration under partial overlaps and large misalignments remains a fundamental challenge. While simultaneous transformation ($\boldsymbolθ$) and correspondence ($\mathbf{P}$) estimation has the advantage of being robust to nonrigid deformation, its non-convex coupled objective often leads to local minima for heuristic methods and prohibitive convergence times for existing global solvers due to loose lower bounds. To address this, we propose DC-Reg, a robust globally optimal framework that significantly tightens the Branch-and-Bound (BnB) search. Our core innovation is the derivation of a holistic concave underestimator for the coupled transformation-assignment objective, grounded in the Difference of Convex (DC) programming paradigm. Unlike prior works that rely on term-wise relaxations (e.g., McCormick envelopes) which neglect variable interplay, our holistic DC decomposition captures the joint structural interaction between $\boldsymbolθ$ and $\mathbf{P}$. This formulation enables the computation of remarkably tight lower bounds via efficient Linear Assignment Problems (LAP) evaluated at the vertices of the search boxes. We validate our framework on 2D similarity and 3D rigid registration, utilizing rotation-invariant features for the latter to achieve high efficiency without sacrificing optimality. Experimental results on synthetic data and the 3DMatch benchmark demonstrate that DC-Reg achieves significantly faster convergence and superior robustness to extreme noise and outliers compared to state-of-the-art global techniques.

  </details>



- **Towards Practical Lossless Neural Compression for LiDAR Point Clouds**  
  Pengpeng Yu, Haoran Li, Runqing Jiang, Dingquan Li, Jing Wang, Liang Lin, Yulan Guo  
  _2026-03-26_ · https://arxiv.org/abs/2603.25260v1  
  <details><summary>Abstract</summary>

  LiDAR point clouds are fundamental to various applications, yet the extreme sparsity of high-precision geometric details hinders efficient context modeling, thereby limiting the compression speed and performance of existing methods. To address this challenge, we propose a compact representation for efficient predictive lossless coding. Our framework comprises two lightweight modules. First, the Geometry Re-Densification Module iteratively densifies encoded sparse geometry, extracts features at a dense scale, and then sparsifies the features for predictive coding. This module avoids costly computation on highly sparse details while maintaining a lightweight prediction head. Second, the Cross-scale Feature Propagation Module leverages occupancy cues from multiple resolution levels to guide hierarchical feature propagation, enabling information sharing across scales and reducing redundant feature extraction. Additionally, we introduce an integer-only inference pipeline to enable bit-exact cross-platform consistency, which avoids the entropy-coding collapse observed in existing neural compression methods and further accelerates coding. Experiments demonstrate competitive compression performance at real-time speed. Code will be released upon acceptance. Code is available at https://github.com/pengpeng-yu/FastPCC.

  </details>



- **Towards Foundation Models for 3D Scene Understanding: Instance-Aware Self-Supervised Learning for Point Clouds**  
  Bin Yang, Mohamed Abdelsamad, Miao Zhang, Alexandru Paul Condurache  
  _2026-03-26_ · https://arxiv.org/abs/2603.25165v1  
  <details><summary>Abstract</summary>

  Recent advances in self-supervised learning (SSL) for point clouds have substantially improved 3D scene understanding without human annotations. Existing approaches emphasize semantic awareness by enforcing feature consistency across augmented views or by masked scene modeling. However, the resulting representations transfer poorly to instance localization, and often require full finetuning for strong performance. Instance awareness is a fundamental component of 3D perception, thus bridging this gap is crucial for progressing toward true 3D foundation models that support all downstream tasks on 3D data. In this work, we introduce PointINS, an instance-oriented self-supervised framework that enriches point cloud representations through geometry-aware learning. PointINS employs an orthogonal offset branch to jointly learn high-level semantic understanding and geometric reasoning, yielding instance awareness. We identify two consistent properties essential for robust instance localization and formulate them as complementary regularization strategies, Offset Distribution Regularization (ODR), which aligns predicted offsets with empirically observed geometric priors, and Spatial Clustering Regularization (SCR), which enforces local coherence by regularizing offsets with pseudo-instance masks. Through extensive experiments across five datasets, PointINS achieves on average +3.5% mAP improvement for indoor instance segmentation and +4.1% PQ gain for outdoor panoptic segmentation, paving the way for scalable 3D foundation models.

  </details>



- **A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection**  
  SuYeon Kim, Wongyu Lee, MyeongAh Cho  
  _2026-03-26_ · https://arxiv.org/abs/2603.25159v1  
  <details><summary>Abstract</summary>

  3D anomaly detection targets the detection and localization of defects in 3D point clouds trained solely on normal data. While a unified model improves scalability by learning across multiple categories, it often suffers from Inter-Category Entanglement (ICE)-where latent features from different categories overlap, causing the model to adopt incorrect semantic priors during reconstruction and ultimately yielding unreliable anomaly scores. To address this issue, we propose the Semantically Disentangled Unified Model for 3D Anomaly Detection, which reconstructs features conditioned on disentangled semantic representations. Our framework consists of three key components: (i) Coarse-to-Fine Global Tokenization for forming instance-level semantic identity, (ii) Category-Conditioned Contrastive Learning for disentangling category semantics, and (iii) a Geometry-Guided Decoder for semantically consistent reconstruction. Extensive experiments on Real3D-AD and Anomaly-ShapeNet demonstrate that our method achieves state-of-the-art for both unified and category-specific models, improving object-level AUROC by 2.8% and 9.1%, respectively, while enhancing the reliability of unified 3D anomaly detection.

  </details>



- **GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator**  
  Liyuan Zhu, Manjunath Narayana, Michal Stary, Will Hutchcroft, Gordon Wetzstein, Iro Armeni  
  _2026-03-26_ · https://arxiv.org/abs/2603.25053v1  
  <details><summary>Abstract</summary>

  We present GaussFusion, a novel approach for improving 3D Gaussian splatting (3DGS) reconstructions in the wild through geometry-informed video generation. GaussFusion mitigates common 3DGS artifacts, including floaters, flickering, and blur caused by camera pose errors, incomplete coverage, and noisy geometry initialization. Unlike prior RGB-based approaches limited to a single reconstruction pipeline, our method introduces a geometry-informed video-to-video generator that refines 3DGS renderings across both optimization-based and feed-forward methods. Given an existing reconstruction, we render a Gaussian primitive video buffer encoding depth, normals, opacity, and covariance, which the generator refines to produce temporally coherent, artifact-free frames. We further introduce an artifact synthesis pipeline that simulates diverse degradation patterns, ensuring robustness and generalization. GaussFusion achieves state-of-the-art performance on novel-view synthesis benchmarks, and an efficient variant runs in real time at 21 FPS while maintaining similar performance, enabling interactive 3D applications.

  </details>



- **Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields**  
  Thanh-Hai Le, Hoang-Hau Tran, Trong-Nghia Vu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25008v1  
  <details><summary>Abstract</summary>

  This paper presents Few TensoRF, a 3D reconstruction framework that combines TensorRF's efficient tensor based representation with FreeNeRF's frequency driven few shot regularization. Using TensorRF to significantly accelerate rendering speed and introducing frequency and occlusion masks, the method improves stability and reconstruction quality under sparse input views. Experiments on the Synthesis NeRF benchmark show that Few TensoRF method improves the average PSNR from 21.45 dB (TensorRF) to 23.70 dB, with the fine tuned version reaching 24.52 dB, while maintaining TensorRF's fast \(\approx10-15\) minute training time. Experiments on the THuman 2.0 dataset further demonstrate competitive performance in human body reconstruction, achieving 27.37 - 34.00 dB with only eight input images. These results highlight Few TensoRF as an efficient and data effective solution for real-time 3D reconstruction across diverse scenes.

  </details>



- **ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects**  
  Jing Yang, Krithika Dharanikota, Emily Jia, Haiwei Chen, Yajie Zhao  
  _2026-03-26_ · https://arxiv.org/abs/2603.24912v1  
  <details><summary>Abstract</summary>

  Accurately modeling how real-world materials reflect light remains a core challenge in inverse rendering, largely due to the scarcity of real measured reflectance data. Existing approaches rely heavily on synthetic datasets with simplified illumination and limited material realism, preventing models from generalizing to real-world images. We introduce a large-scale polarized reflection and material dataset of real-world objects, captured with an 8-camera, 346-light Light Stage equipped with cross/parallel polarization. Our dataset spans 218 everyday objects across five acquisition dimensions-multiview, multi-illumination, polarization, reflectance separation, and material attributes-yielding over 1.2M high-resolution images with diffuse-specular separation and analytically derived diffuse albedo, specular albedo, and surface normals. Using this dataset, we train and evaluate state-of-the-art inverse and forward rendering models on intrinsic decomposition, relighting, and sparse-view 3D reconstruction, demonstrating significant improvements in material separation, illumination fidelity, and geometric consistency. We hope that our work can establish a new foundation for physically grounded material understanding and enable real-world generalization beyond synthetic training regimes. Project page: https://jingyangcarl.github.io/ICTPolarReal/

  </details>



- **DRoPS: Dynamic 3D Reconstruction of Pre-Scanned Objects**  
  Narek Tumanyan, Samuel Rota Bulò, Denis Rozumny, Lorenzo Porzi, Adam Harley, Tali Dekel, Peter Kontschieder, Jonathon Luiten  
  _2026-03-25_ · https://arxiv.org/abs/2603.24770v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction from casual videos has seen recent remarkable progress. Numerous approaches have attempted to overcome the ill-posedness of the task by distilling priors from 2D foundational models and by imposing hand-crafted regularization on the optimized motion. However, these methods struggle to reconstruct scenes from extreme novel viewpoints, especially when highly articulated motions are present. In this paper, we present DRoPS, a novel approach that leverages a static pre-scan of the dynamic object as an explicit geometric and appearance prior. While existing state-of-the-art methods fail to fully exploit the pre-scan, DRoPS leverages our novel setup to effectively constrain the solution space and ensure geometrical consistency throughout the sequence. The core of our novelty is twofold: first, we establish a grid-structured and surface-aligned model by organizing Gaussian primitives into pixel grids anchored to the object surface. Second, by leveraging the grid structure of our primitives, we parameterize motion using a CNN conditioned on those grids, injecting strong implicit regularization and correlating the motion of nearby points. Extensive experiments demonstrate that our method significantly outperforms the current state of the art in rendering quality and 3D tracking accuracy.

  </details>



- **Accurate Point Measurement in 3DGS -- A New Alternative to Traditional Stereoscopic-View Based Measurements**  
  Deyan Deng, Rongjun Qin  
  _2026-03-25_ · https://arxiv.org/abs/2603.24716v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized real-time rendering with its state-of-the-art novel view synthesis, but its utility for accurate geometric measurement remains underutilized. Compared to multi-view stereo (MVS) point clouds or meshes, 3DGS rendered views present superior visual quality and completeness. However, current point measurement methods still rely on demanding stereoscopic workstations or direct picking on often-incomplete and inaccurate 3D meshes. As a novel view synthesizer, 3DGS renders exact source views and smoothly interpolates in-between views. This allows users to intuitively pick congruent points across different views while operating 3DGS models. By triangulating these congruent points, one can precisely generate 3D point measurements. This approach mimics traditional stereoscopic measurement but is significantly less demanding: it requires neither a stereo workstation nor specialized operator stereoscopic capability. Furthermore, it enables multi-view intersection (more than two views) for higher measurement accuracy. We implemented a web-based application to demonstrate this proof-of-concept (PoC). Using several UAV aerial datasets, we show this PoC allows users to successfully perform highly accurate point measurements, achieving accuracy matching or exceeding traditional stereoscopic methods on standard hardware. Specifically, our approach significantly outperforms direct mesh-based measurements. Quantitatively, our method achieves RMSEs in the 1-2 cm range on well-defined points. More critically, on challenging thin structures where mesh-based RMSE was 0.062 m, our method achieved 0.037 m. On sharp corners poorly reconstructed in the mesh, our method successfully measured all points with a 0.013 m RMSE, whereas the mesh method failed entirely. Code is available at: https://github.com/GDAOSU/3dgs_measurement_tool.

  </details>



- **KitchenTwin: Semantically and Geometrically Grounded 3D Kitchen Digital Twins**  
  Quanyun Wu, Kyle Gao, Daniel Long, David A. Clausi, Jonathan Li, Yuhao Chen  
  _2026-03-25_ · https://arxiv.org/abs/2603.24684v1  
  <details><summary>Abstract</summary>

  Embodied AI training and evaluation require object-centric digital twin environments with accurate metric geometry and semantic grounding. Recent transformer-based feedforward reconstruction methods can efficiently predict global point clouds from sparse monocular videos, yet these geometries suffer from inherent scale ambiguity and inconsistent coordinate conventions. This mismatch prevents the reliable fusion of these dimensionless point cloud predictions with locally reconstructed object meshes. We propose a novel scale-aware 3D fusion framework that registers visually grounded object meshes with transformer-predicted global point clouds to construct metrically consistent digital twins. Our method introduces a Vision-Language Model (VLM)-guided geometric anchor mechanism that resolves this fundamental coordinate mismatch by recovering an accurate real-world metric scale. To fuse these networks, we propose a geometry-aware registration pipeline that explicitly enforces physical plausibility through gravity-aligned vertical estimation, Manhattan-world structural constraints, and collision-free local refinement. Experiments on real indoor kitchen environments demonstrate improved cross-network object alignment and geometric consistency for downstream tasks, including multi-primitive fitting and metric measurement. We additionally introduce an open-source indoor digital twin dataset with metrically scaled scenes and semantically grounded and registered object-centric mesh annotations.

  </details>



- **EndoVGGT: GNN-Enhanced Depth Estimation for Surgical 3D Reconstruction**  
  Falong Fan, Yi Xie, Arnis Lektauers, Bo Liu, Jerzy Rozenblit  
  _2026-03-25_ · https://arxiv.org/abs/2603.24577v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of deformable soft tissues is essential for surgical robotic perception. However, low-texture surfaces, specular highlights, and instrument occlusions often fragment geometric continuity, posing a challenge for existing fixed-topology approaches. To address this, we propose EndoVGGT, a geometry-centric framework equipped with a Deformation-aware Graph Attention (DeGAT) module. Rather than using static spatial neighborhoods, DeGAT dynamically constructs feature-space semantic graphs to capture long-range correlations among coherent tissue regions. This enables robust propagation of structural cues across occlusions, enforcing global consistency and improving non-rigid deformation recovery. Extensive experiments on SCARED show that our method significantly improves fidelity, increasing PSNR by 24.6% and SSIM by 9.1% over prior state-of-the-art. Crucially, EndoVGGT exhibits strong zero-shot cross-dataset generalization to the unseen SCARED and EndoNeRF domains, confirming that DeGAT learns domain-agnostic geometric priors. These results highlight the efficacy of dynamic feature-space modeling for consistent surgical 3D reconstruction.

  </details>


