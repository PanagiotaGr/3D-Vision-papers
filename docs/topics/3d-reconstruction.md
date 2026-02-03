# 3D Reconstruction

_Updated: 2026-02-03 07:08 UTC_

Total papers shown: **17**


---

- **LiFlow: Flow Matching for 3D LiDAR Scene Completion**  
  Andrea Matteazzi, Dietmar Tutsch  
  _2026-02-02_ · https://arxiv.org/abs/2602.02232v1  
  <details><summary>Abstract</summary>

  In autonomous driving scenarios, the collected LiDAR point clouds can be challenged by occlusion and long-range sparsity, limiting the perception of autonomous driving systems. Scene completion methods can infer the missing parts of incomplete 3D LiDAR scenes. Recent methods adopt local point-level denoising diffusion probabilistic models, which require predicting Gaussian noise, leading to a mismatch between training and inference initial distributions. This paper introduces the first flow matching framework for 3D LiDAR scene completion, improving upon diffusion-based methods by ensuring consistent initial distributions between training and inference. The model employs a nearest neighbor flow matching loss and a Chamfer distance loss to enhance both local structure and global coverage in the alignment of point clouds. LiFlow achieves state-of-the-art performance across multiple metrics. Code: https://github.com/matteandre/LiFlow.

  </details>



- **Learning Topology-Aware Implicit Field for Unified Pulmonary Tree Modeling with Incomplete Topological Supervision**  
  Ziqiao Weng, Jiancheng Yang, Kangxian Xie, Bo Zhou, Weidong Cai  
  _2026-02-02_ · https://arxiv.org/abs/2602.02186v1  
  <details><summary>Abstract</summary>

  Pulmonary trees extracted from CT images frequently exhibit topological incompleteness, such as missing or disconnected branches, which substantially degrades downstream anatomical analysis and limits the applicability of existing pulmonary tree modeling pipelines. Current approaches typically rely on dense volumetric processing or explicit graph reasoning, leading to limited efficiency and reduced robustness under realistic structural corruption. We propose TopoField, a topology-aware implicit modeling framework that treats topology repair as a first-class modeling problem and enables unified multi-task inference for pulmonary tree analysis. TopoField represents pulmonary anatomy using sparse surface and skeleton point clouds and learns a continuous implicit field that supports topology repair without relying on complete or explicit disconnection annotations, by training on synthetically introduced structural disruptions over \textit{already} incomplete trees. Building upon the repaired implicit representation, anatomical labeling and lung segment reconstruction are jointly inferred through task-specific implicit functions within a single forward pass.Extensive experiments on the Lung3D+ dataset demonstrate that TopoField consistently improves topological completeness and achieves accurate anatomical labeling and lung segment reconstruction under challenging incomplete scenarios. Owing to its implicit formulation, TopoField attains high computational efficiency, completing all tasks in just over one second per case, highlighting its practicality for large-scale and time-sensitive clinical applications. Code and data will be available at https://github.com/HINTLab/TopoField.

  </details>



- **SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors**  
  Bing He, Jingnan Gao, Yunuo Chen, Ning Cao, Gang Chen, Zhengxue Cheng, Li Song, Wenjun Zhang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02000v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D scenes from sparse images remains a challenging task due to the difficulty of recovering accurate geometry and texture without optimization. Recent approaches leverage generalizable models to generate 3D scenes using 3D Gaussian Splatting (3DGS) primitive. However, they often fail to produce continuous surfaces and instead yield discrete, color-biased point clouds that appear plausible at normal resolution but reveal severe artifacts under close-up views. To address this issue, we present SurfSplat, a feedforward framework based on 2D Gaussian Splatting (2DGS) primitive, which provides stronger anisotropy and higher geometric precision. By incorporating a surface continuity prior and a forced alpha blending strategy, SurfSplat reconstructs coherent geometry together with faithful textures. Furthermore, we introduce High-Resolution Rendering Consistency (HRRC), a new evaluation metric designed to evaluate high-resolution reconstruction quality. Extensive experiments on RealEstate10K, DL3DV, and ScanNet demonstrate that SurfSplat consistently outperforms prior methods on both standard metrics and HRRC, establishing a robust solution for high-fidelity 3D reconstruction from sparse inputs. Project page: https://hebing-sjtu.github.io/SurfSplat-website/

  </details>



- **Multi-Task Learning for Robot Perception with Imbalanced Data**  
  Ozgur Erkent  
  _2026-02-02_ · https://arxiv.org/abs/2602.01899v1  
  <details><summary>Abstract</summary>

  Multi-task problem solving has been shown to improve the accuracy of the individual tasks, which is an important feature for robots, as they have a limited resource. However, when the number of labels for each task is not equal, namely imbalanced data exist, a problem may arise due to insufficient number of samples, and labeling is not very easy for mobile robots in every environment. We propose a method that can learn tasks even in the absence of the ground truth labels for some of the tasks. We also provide a detailed analysis of the proposed method. An interesting finding is related to the interaction of the tasks. We show a methodology to find out which tasks can improve the performance of other tasks. We investigate this by training the teacher network with the task outputs such as depth as inputs. We further provide empirical evidence when trained with a small amount of data. We use semantic segmentation and depth estimation tasks on different datasets, NYUDv2 and Cityscapes.

  </details>



- **Automated Discontinuity Set Characterisation in Enclosed Rock Face Point Clouds Using Single-Shot Filtering and Cyclic Orientation Transformation**  
  Dibyayan Patra, Pasindu Ranasinghe, Bikram Banerjee, Simit Raval  
  _2026-02-02_ · https://arxiv.org/abs/2602.01783v1  
  <details><summary>Abstract</summary>

  Characterisation of structural discontinuity sets in exposed rock faces of underground mine cavities is essential for assessing rock-mass stability, excavation safety, and operational efficiency. UAV and other mobile laser-scanning techniques provide efficient means of collecting point clouds from rock faces. However, the development of a robust and efficient approach for automatic characterisation of discontinuity sets in real-world scenarios, like fully enclosed rock faces in cavities, remains an open research problem. In this study, a new approach is proposed for automatic discontinuity set characterisation that uses a single-shot filtering strategy, an innovative cyclic orientation transformation scheme and a hierarchical clustering technique. The single-shot filtering step isolates planar regions while robustly suppressing noise and high-curvature artefacts in one pass using a signal-processing technique. To address the limitations of Cartesian clustering on polar orientation data, a cyclic orientation transformation scheme is developed, enabling accurate representation of dip angle and dip direction in Cartesian space. The transformed orientations are then characterised into sets using a hierarchical clustering technique, which handles varying density distributions and identifies clusters without requiring user-defined set numbers. The accuracy of the method is validated on real-world mine stope and against ground truth obtained using manually handpicked discontinuity planes identified with the Virtual Compass tool, as well as widely used automated structure mapping techniques. The proposed approach outperforms the other techniques by exhibiting the lowest mean absolute error in estimating discontinuity set orientations in real-world stope data with errors of 1.95° and 2.20° in nominal dip angle and dip direction, respectively, and dispersion errors lying below 3°.

  </details>



- **GDPR-Compliant Person Recognition in Industrial Environments Using MEMS-LiDAR and Hybrid Data**  
  Dennis Basile, Dennis Sprute, Helene Dörksen, Holger Flatt  
  _2026-02-02_ · https://arxiv.org/abs/2602.01764v1  
  <details><summary>Abstract</summary>

  The reliable detection of unauthorized individuals in safety-critical industrial indoor spaces is crucial to avoid plant shutdowns, property damage, and personal hazards. Conventional vision-based methods that use deep-learning approaches for person recognition provide image information but are sensitive to lighting and visibility conditions and often violate privacy regulations, such as the General Data Protection Regulation (GDPR) in the European Union. Typically, detection systems based on deep learning require annotated data for training. Collecting and annotating such data, however, is highly time-consuming and due to manual treatments not necessarily error free. Therefore, this paper presents a privacy-compliant approach based on Micro-Electro-Mechanical Systems LiDAR (MEMS-LiDAR), which exclusively captures anonymized 3D point clouds and avoids personal identification features. To compensate for the large amount of time required to record real LiDAR data and for post-processing and annotation, real recordings are augmented with synthetically generated scenes from the CARLA simulation framework. The results demonstrate that the hybrid data improves the average precision by 44 percentage points compared to a model trained exclusively with real data while reducing the manual annotation effort by 50 %. Thus, the proposed approach provides a scalable, cost-efficient alternative to purely real-data-based methods and systematically shows how synthetic LiDAR data can combine high performance in person detection with GDPR compliance in an industrial environment.

  </details>



- **HandMCM: Multi-modal Point Cloud-based Correspondence State Space Model for 3D Hand Pose Estimation**  
  Wencan Cheng, Gim Hee Lee  
  _2026-02-02_ · https://arxiv.org/abs/2602.01586v1  
  <details><summary>Abstract</summary>

  3D hand pose estimation that involves accurate estimation of 3D human hand keypoint locations is crucial for many human-computer interaction applications such as augmented reality. However, this task poses significant challenges due to self-occlusion of the hands and occlusions caused by interactions with objects. In this paper, we propose HandMCM to address these challenges. Our HandMCM is a novel method based on the powerful state space model (Mamba). By incorporating modules for local information injection/filtering and correspondence modeling, the proposed correspondence Mamba effectively learns the highly dynamic kinematic topology of keypoints across various occlusion scenarios. Moreover, by integrating multi-modal image features, we enhance the robustness and representational capacity of the input, leading to more accurate hand pose estimation. Empirical evaluations on three benchmark datasets demonstrate that our model significantly outperforms current state-of-the-art methods, particularly in challenging scenarios involving severe occlusions. These results highlight the potential of our approach to advance the accuracy and reliability of 3D hand pose estimation in practical applications.

  </details>



- **Where to Attend: A Principled Vision-Centric Position Encoding with Parabolas**  
  Christoffer Koo Øhrstrøm, Rafael I. Cabral Muchacho, Yifei Dong, Filippos Moumtzidellis, Ronja Güldenring, Florian T. Pokorny, Lazaros Nalpantidis  
  _2026-02-01_ · https://arxiv.org/abs/2602.01418v1  
  <details><summary>Abstract</summary>

  We propose Parabolic Position Encoding (PaPE), a parabola-based position encoding for vision modalities in attention-based architectures. Given a set of vision tokens-such as images, point clouds, videos, or event camera streams-our objective is to encode their positions while accounting for the characteristics of vision modalities. Prior works have largely extended position encodings from 1D-sequences in language to nD-structures in vision, but only with partial account of vision characteristics. We address this gap by designing PaPE from principles distilled from prior work: translation invariance, rotation invariance (PaPE-RI), distance decay, directionality, and context awareness. We evaluate PaPE on 8 datasets that span 4 modalities. We find that either PaPE or PaPE-RI achieves the top performance on 7 out of 8 datasets. Extrapolation experiments on ImageNet-1K show that PaPE extrapolates remarkably well, improving in absolute terms by up to 10.5% over the next-best position encoding. Code is available at https://github.com/DTU-PAS/parabolic-position-encoding.

  </details>



- **OASIS-DC: Generalizable Depth Completion via Output-level Alignment of Sparse-Integrated Monocular Pseudo Depth**  
  Jaehyeon Cho, Jhonghyun An  
  _2026-02-01_ · https://arxiv.org/abs/2602.01268v1  
  <details><summary>Abstract</summary>

  Recent monocular foundation models excel at zero-shot depth estimation, yet their outputs are inherently relative rather than metric, limiting direct use in robotics and autonomous driving. We leverage the fact that relative depth preserves global layout and boundaries: by calibrating it with sparse range measurements, we transform it into a pseudo metric depth prior. Building on this prior, we design a refinement network that follows the prior where reliable and deviates where necessary, enabling accurate metric predictions from very few labeled samples. The resulting system is particularly effective when curated validation data are unavailable, sustaining stable scale and sharp edges across few-shot regimes. These findings suggest that coupling foundation priors with sparse anchors is a practical route to robust, deployment-ready depth completion under real-world label scarcity.

  </details>



- **LightCity: An Urban Dataset for Outdoor Inverse Rendering and Reconstruction under Multi-illumination Conditions**  
  Jingjing Wang, Qirui Hu, Chong Bao, Yuke Zhu, Hujun Bao, Zhaopeng Cui, Guofeng Zhang  
  _2026-02-01_ · https://arxiv.org/abs/2602.01118v1  
  <details><summary>Abstract</summary>

  Inverse rendering in urban scenes is pivotal for applications like autonomous driving and digital twins. Yet, it faces significant challenges due to complex illumination conditions, including multi-illumination and indirect light and shadow effects. However, the effects of these challenges on intrinsic decomposition and 3D reconstruction have not been explored due to the lack of appropriate datasets. In this paper, we present LightCity, a novel high-quality synthetic urban dataset featuring diverse illumination conditions with realistic indirect light and shadow effects. LightCity encompasses over 300 sky maps with highly controllable illumination, varying scales with street-level and aerial perspectives over 50K images, and rich properties such as depth, normal, material components, light and indirect light, etc. Besides, we leverage LightCity to benchmark three fundamental tasks in the urban environments and conduct a comprehensive analysis of these benchmarks, laying a robust foundation for advancing related research.

  </details>



- **FUSE-Flow: Scalable Real-Time Multi-View Point Cloud Reconstruction Using Confidence**  
  Chentian Sun  
  _2026-02-01_ · https://arxiv.org/abs/2602.01035v1  
  <details><summary>Abstract</summary>

  Real-time multi-view point cloud reconstruction is a core problem in 3D vision and immersive perception, with wide applications in VR, AR, robotic navigation, digital twins, and computer interaction. Despite advances in multi-camera systems and high-resolution depth sensors, fusing large-scale multi-view depth observations into high-quality point clouds under strict real-time constraints remains challenging. Existing methods relying on voxel-based fusion, temporal accumulation, or global optimization suffer from high computational complexity, excessive memory usage, and limited scalability, failing to simultaneously achieve real-time performance, reconstruction quality, and multi-camera extensibility. We propose FUSE-Flow, a frame-wise, stateless, and linearly scalable point cloud streaming reconstruction framework. Each frame independently generates point cloud fragments, fused via two weights, measurement confidence and 3D distance consistency to suppress noise while preserving geometric details. For large-scale multi-camera efficiency, we introduce an adaptive spatial hashing-based weighted aggregation method: 3D space is adaptively partitioned by local point cloud density, representative points are selected per cell, and weighted fusion is performed to handle both sparse and dense regions. With GPU parallelization, FUSE-Flow achieves high-throughput, low-latency point cloud generation and fusion with linear complexity. Experiments demonstrate that the framework improves reconstruction stability and geometric fidelity in overlapping, depth-discontinuous, and dynamic scenes, while maintaining real-time frame rates on modern GPUs, verifying its effectiveness, robustness, and scalability.

  </details>



- **GMAC: Global Multi-View Constraint for Automatic Multi-Camera Extrinsic Calibration**  
  Chentian Sun  
  _2026-02-01_ · https://arxiv.org/abs/2602.01033v1  
  <details><summary>Abstract</summary>

  Automatic calibration of multi-camera systems, namely the accurate estimation of spatial extrinsic parameters, is fundamental for 3D reconstruction, panoramic perception, and multi-view data fusion. Existing methods typically rely on calibration targets, explicit geometric modeling, or task-specific neural networks. Such approaches often exhibit limited robustness and applicability in complex dynamic environments or online scenarios, making them difficult to deploy in practical applications. To address this, this paper proposes GMAC, a multi-camera extrinsic estimation framework based on the implicit geometric representations learned by multi-view reconstruction networks. GMAC models extrinsics as global variables constrained by the latent multi-view geometric structure and prunes and structurally reconfigures existing networks so that their latent features can directly support extrinsic prediction through a lightweight regression head, without requiring a completely new network design. Furthermore, GMAC jointly optimizes cross-view reprojection consistency and multi-view cycle consistency, ensuring geometric coherence across cameras while improving prediction accuracy and optimization stability. Experiments on both synthetic and real-world multi-camera datasets demonstrate that GMAC achieves accurate and stable extrinsic estimation without explicit 3D reconstruction or manual calibration, providing a new solution for efficient deployment and online calibration of multi-camera systems.

  </details>



- **CLAMP: Contrastive Learning for 3D Multi-View Action-Conditioned Robotic Manipulation Pretraining**  
  I-Chun Arthur Liu, Krzysztof Choromanski, Sandy Huang, Connor Schenck  
  _2026-01-31_ · https://arxiv.org/abs/2602.00937v1  
  <details><summary>Abstract</summary>

  Leveraging pre-trained 2D image representations in behavior cloning policies has achieved great success and has become a standard approach for robotic manipulation. However, such representations fail to capture the 3D spatial information about objects and scenes that is essential for precise manipulation. In this work, we introduce Contrastive Learning for 3D Multi-View Action-Conditioned Robotic Manipulation Pretraining (CLAMP), a novel 3D pre-training framework that utilizes point clouds and robot actions. From the merged point cloud computed from RGB-D images and camera extrinsics, we re-render multi-view four-channel image observations with depth and 3D coordinates, including dynamic wrist views, to provide clearer views of target objects for high-precision manipulation tasks. The pre-trained encoders learn to associate the 3D geometric and positional information of objects with robot action patterns via contrastive learning on large-scale simulated robot trajectories. During encoder pre-training, we pre-train a Diffusion Policy to initialize the policy weights for fine-tuning, which is essential for improving fine-tuning sample efficiency and performance. After pre-training, we fine-tune the policy on a limited amount of task demonstrations using the learned image and action representations. We demonstrate that this pre-training and fine-tuning design substantially improves learning efficiency and policy performance on unseen tasks. Furthermore, we show that CLAMP outperforms state-of-the-art baselines across six simulated tasks and five real-world tasks.

  </details>



- **Distill3R: A Pipeline for Democratizing 3D Foundation Models on Commodity Hardware**  
  Brandon Leblanc, Charalambos Poullis  
  _2026-01-31_ · https://arxiv.org/abs/2602.00865v1  
  <details><summary>Abstract</summary>

  While multi-view 3D reconstruction has shifted toward large-scale foundation models capable of inferring globally consistent geometry, their reliance on massive computational clusters for training has created a significant barrier to entry for most academic laboratories. To bridge this compute divide, we introduce Distill3R, a framework designed to distill the geometric reasoning of 3D foundation models into compact students fully trainable on a single workstation. Our methodology centers on two primary innovations: (1) an offline caching pipeline that decouples heavy teacher inference from the training loop through compressed supervision signals, and (2) a confidence-aware distillation loss that leverages teacher uncertainty to enable training on commodity hardware. We propose a 72M-parameter student model which achieves a 9x reduction in parameters and a 5x inference speedup compared to its 650M-parameter teacher. The student is fully trainable in under 3 days on a single workstation, whereas its teacher requires massive GPU clusters for up to a week. We demonstrate that the student preserves the structural consistency and qualitative geometric understanding required for functional 3D awareness. By providing a reproducible, single-workstation training recipe, Distill3R serves as an exploratory entry point for democratized 3D vision research and efficient edge deployment. This work is not intended to compete with state-of-the-art foundation models, but to provide an accessible research baseline for laboratories without access to large-scale compute to train and specialize models on their own domain-specific data at minimal cost.

  </details>



- **Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds**  
  Xianzhe Fan, Shengliang Deng, Xiaoyang Wu, Yuxiang Lu, Zhuoling Li, Mi Yan, Yujia Zhang, Zhizheng Zhang, He Wang, Hengshuang Zhao  
  _2026-01-31_ · https://arxiv.org/abs/2602.00807v1  
  <details><summary>Abstract</summary>

  Existing Vision-Language-Action (VLA) models typically take 2D images as visual input, which limits their spatial understanding in complex scenes. How can we incorporate 3D information to enhance VLA capabilities? We conduct a pilot study across different observation spaces and visual representations. The results show that explicitly lifting visual input into point clouds yields representations that better complement their corresponding 2D representations. To address the challenges of (1) scarce 3D data and (2) the domain gap induced by cross-environment differences and depth-scale biases, we propose Any3D-VLA. It unifies the simulator, sensor, and model-estimated point clouds within a training pipeline, constructs diverse inputs, and learns domain-agnostic 3D representations that are fused with the corresponding 2D representations. Simulation and real-world experiments demonstrate Any3D-VLA's advantages in improving performance and mitigating the domain gap. Our project homepage is available at https://xianzhefan.github.io/Any3D-VLA.github.io.

  </details>



- **Diffusion-Driven Inter-Outer Surface Separation for Point Clouds with Open Boundaries**  
  Zhengyan Qin, Liyuan Qiu  
  _2026-01-31_ · https://arxiv.org/abs/2602.00739v1  
  <details><summary>Abstract</summary>

  We propose a diffusion-based algorithm for separating the inter and outer layer surfaces from double-layered point clouds, particularly those exhibiting the "double surface artifact" caused by truncation in Truncated Signed Distance Function (TSDF) fusion during indoor or medical 3D reconstruction. This artifact arises from asymmetric truncation thresholds, leading to erroneous inter and outer shells in the fused volume, which our method addresses by extracting the true inter layer to mitigate challenges like overlapping surfaces and disordered normals. We focus on point clouds with \emph{open boundaries} (i.e., sampled surfaces with topological openings/holes through which particles may escape), rather than point clouds with \emph{missing surface regions} where no samples exist. Our approach enables robust processing of both watertight and open-boundary models, achieving extraction of the inter layer from 20,000 inter and 20,000 outer points in approximately 10 seconds. This solution is particularly effective for applications requiring accurate surface representations, such as indoor scene modeling and medical imaging, where double-layered point clouds are prevalent, and it accommodates both closed (watertight) and open-boundary surface geometries. Our goal is \emph{post-hoc} inter/outer shell separation as a lightweight module after TSDF fusion; we do not aim to replace full variational or learning-based reconstruction pipelines.

  </details>



- **Improving Neuropathological Reconstruction Fidelity via AI Slice Imputation**  
  Marina Crespo Aguirre, Jonathan Williams-Ramirez, Dina Zemlyanker, Xiaoling Hu, Lucas J. Deden-Binder, Rogeny Herisse, Mark Montine, Theresa R. Connors, Christopher Mount, Christine L. MacDonald, et al.  
  _2026-01-31_ · https://arxiv.org/abs/2602.00669v1  
  <details><summary>Abstract</summary>

  Neuropathological analyses benefit from spatially precise volumetric reconstructions that enhance anatomical delineation and improve morphometric accuracy. Our prior work has shown the feasibility of reconstructing 3D brain volumes from 2D dissection photographs. However these outputs sometimes exhibit coarse, overly smooth reconstructions of structures, especially under high anisotropy (i.e., reconstructions from thick slabs). Here, we introduce a computationally efficient super-resolution step that imputes slices to generate anatomically consistent isotropic volumes from anisotropic 3D reconstructions of dissection photographs. By training on domain-randomized synthetic data, we ensure that our method generalizes across dissection protocols and remains robust to large slab thicknesses. The imputed volumes yield improved automated segmentations, achieving higher Dice scores, particularly in cortical and white matter regions. Validation on surface reconstruction and atlas registration tasks demonstrates more accurate cortical surfaces and MRI registration. By enhancing the resolution and anatomical fidelity of photograph-based reconstructions, our approach strengthens the bridge between neuropathology and neuroimaging. Our method is publicly available at https://surfer.nmr.mgh.harvard.edu/fswiki/mri_3d_photo_recon

  </details>


