# 3D Reconstruction

_Updated: 2026-02-04 07:08 UTC_

Total papers shown: **13**


---

- **EventNeuS: 3D Mesh Reconstruction from a Single Event Camera**  
  Shreyas Sachan, Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03847v1  
  <details><summary>Abstract</summary>

  Event cameras offer a considerable alternative to RGB cameras in many scenarios. While there are recent works on event-based novel-view synthesis, dense 3D mesh reconstruction remains scarcely explored and existing event-based techniques are severely limited in their 3D reconstruction accuracy. To address this limitation, we present EventNeuS, a self-supervised neural model for learning 3D representations from monocular colour event streams. Our approach, for the first time, combines 3D signed distance function and density field learning with event-based supervision. Furthermore, we introduce spherical harmonics encodings into our model for enhanced handling of view-dependent effects. EventNeuS outperforms existing approaches by a significant margin, achieving 34% lower Chamfer distance and 31% lower mean absolute error on average compared to the best previous method.

  </details>



- **AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping**  
  Dingyi Zhou, Mu He, Zhuowei Fang, Xiangtong Yao, Yinlong Liu, Alois Knoll, Hu Cao  
  _2026-02-03_ · https://arxiv.org/abs/2602.03547v1  
  <details><summary>Abstract</summary>

  We introduce AffordanceGrasp-R1, a reasoning-driven affordance segmentation framework for robotic grasping that combines a chain-of-thought (CoT) cold-start strategy with reinforcement learning to enhance deduction and spatial grounding. In addition, we redesign the grasping pipeline to be more context-aware by generating grasp candidates from the global scene point cloud and subsequently filtering them using instruction-conditioned affordance masks. Extensive experiments demonstrate that AffordanceGrasp-R1 consistently outperforms state-of-the-art (SOTA) methods on benchmark datasets, and real-world robotic grasping evaluations further validate its robustness and generalization under complex language-conditioned manipulation scenarios.

  </details>



- **PWAVEP: Purifying Imperceptible Adversarial Perturbations in 3D Point Clouds via Spectral Graph Wavelets**  
  Haoran Li, Renyang Liu, Hongjia Liu, Chen Wang, Long Yin, Jian Xu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03333v1  
  <details><summary>Abstract</summary>

  Recent progress in adversarial attacks on 3D point clouds, particularly in achieving spatial imperceptibility and high attack performance, presents significant challenges for defenders. Current defensive approaches remain cumbersome, often requiring invasive model modifications, expensive training procedures or auxiliary data access. To address these threats, in this paper, we propose a plug-and-play and non-invasive defense mechanism in the spectral domain, grounded in a theoretical and empirical analysis of the relationship between imperceptible perturbations and high-frequency spectral components. Building upon these insights, we introduce a novel purification framework, termed PWAVEP, which begins by computing a spectral graph wavelet domain saliency score and local sparsity score for each point. Guided by these values, PWAVEP adopts a hierarchical strategy, it eliminates the most salient points, which are identified as hardly recoverable adversarial outliers. Simultaneously, it applies a spectral filtering process to a broader set of moderately salient points. This process leverages a graph wavelet transform to attenuate high-frequency coefficients associated with the targeted points, thereby effectively suppressing adversarial noise. Extensive evaluations demonstrate that the proposed PWAVEP achieves superior accuracy and robustness compared to existing approaches, advancing the state-of-the-art in 3D point cloud purification. Code and datasets are available at https://github.com/a772316182/pwavep

  </details>



- **Pi-GS: Sparse-View Gaussian Splatting with Dense π^3 Initialization**  
  Manuel Hofer, Markus Steinberger, Thomas Köhler  
  _2026-02-03_ · https://arxiv.org/abs/2602.03327v1  
  <details><summary>Abstract</summary>

  Novel view synthesis has evolved rapidly, advancing from Neural Radiance Fields to 3D Gaussian Splatting (3DGS), which offers real-time rendering and rapid training without compromising visual fidelity. However, 3DGS relies heavily on accurate camera poses and high-quality point cloud initialization, which are difficult to obtain in sparse-view scenarios. While traditional Structure from Motion (SfM) pipelines often fail in these settings, existing learning-based point estimation alternatives typically require reliable reference views and remain sensitive to pose or depth errors. In this work, we propose a robust method utilizing π^3, a reference-free point cloud estimation network. We integrate dense initialization from π^3 with a regularization scheme designed to mitigate geometric inaccuracies. Specifically, we employ uncertainty-guided depth supervision, normal consistency loss, and depth warping. Experimental results demonstrate that our approach achieves state-of-the-art performance on the Tanks and Temples, LLFF, DTU, and MipNeRF360 datasets.

  </details>



- **LmPT: Conditional Point Transformer for Anatomical Landmark Detection on 3D Point Clouds**  
  Matteo Bastico, Pierre Onghena, David Ryckelynck, Beatriz Marcotegui, Santiago Velasco-Forero, Laurent Corté, Caroline Robine--Decourcelle, Etienne Decencière  
  _2026-02-02_ · https://arxiv.org/abs/2602.02808v1  
  <details><summary>Abstract</summary>

  Accurate identification of anatomical landmarks is crucial for various medical applications. Traditional manual landmarking is time-consuming and prone to inter-observer variability, while rule-based methods are often tailored to specific geometries or limited sets of landmarks. In recent years, anatomical surfaces have been effectively represented as point clouds, which are lightweight structures composed of spatial coordinates. Following this strategy and to overcome the limitations of existing landmarking techniques, we propose Landmark Point Transformer (LmPT), a method for automatic anatomical landmark detection on point clouds that can leverage homologous bones from different species for translational research. The LmPT model incorporates a conditioning mechanism that enables adaptability to different input types to conduct cross-species learning. We focus the evaluation of our approach on femoral landmarking using both human and newly annotated dog femurs, demonstrating its generalization and effectiveness across species. The code and dog femur dataset will be publicly available at: https://github.com/Pierreoo/LandmarkPointTransformer.

  </details>



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
  _2026-02-02_ · https://arxiv.org/abs/2602.02000v2  
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


