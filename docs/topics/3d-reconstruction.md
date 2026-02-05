# 3D Reconstruction

_Updated: 2026-02-05 07:14 UTC_

Total papers shown: **19**


---

- **LitS: A novel Neighborhood Descriptor for Point Clouds**  
  Jonatan B. Bastos, Francisco F. Rivera, Oscar G. Lorenzo, David L. Vilariño, José C. Cabaleiro, Alberto M. Esmorís, Tomás F. Pena  
  _2026-02-04_ · https://arxiv.org/abs/2602.04838v1  
  <details><summary>Abstract</summary>

  With the advancement of 3D scanning technologies, point clouds have become fundamental for representing 3D spatial data, with applications that span across various scientific and technological fields. Practical analysis of this data depends crucially on available neighborhood descriptors to accurately characterize the local geometries of the point cloud. This paper introduces LitS, a novel neighborhood descriptor for 2D and 3D point clouds. LitS are piecewise constant functions on the unit circle that allow points to keep track of their surroundings. Each element in LitS' domain represents a direction with respect to a local reference system. Once constructed, evaluating LitS at any given direction gives us information about the number of neighbors in a cone-like region centered around that same direction. Thus, LitS conveys a lot of information about the local neighborhood of a point, which can be leveraged to gain global structural understanding by analyzing how LitS changes between close points. In addition, LitS comes in two versions ('regular' and 'cumulative') and has two parameters, allowing them to adapt to various contexts and types of point clouds. Overall, they are a versatile neighborhood descriptor, capable of capturing the nuances of local point arrangements and resilient to common point cloud data issues such as variable density and noise.

  </details>



- **How to rewrite the stars: Mapping your orchard over time through constellations of fruits**  
  Gonçalo P. Matos, Carlos Santiago, João P. Costeira, Ricardo L. Saldanha, Ernesto M. Morgado  
  _2026-02-04_ · https://arxiv.org/abs/2602.04722v1  
  <details><summary>Abstract</summary>

  Following crop growth through the vegetative cycle allows farmers to predict fruit setting and yield in early stages, but it is a laborious and non-scalable task if performed by a human who has to manually measure fruit sizes with a caliper or dendrometers. In recent years, computer vision has been used to automate several tasks in precision agriculture, such as detecting and counting fruits, and estimating their size. However, the fundamental problem of matching the exact same fruits from one video, collected on a given date, to the fruits visible in another video, collected on a later date, which is needed to track fruits' growth through time, remains to be solved. Few attempts were made, but they either assume that the camera always starts from the same known position and that there are sufficiently distinct features to match, or they used other sources of data like GPS. Here we propose a new paradigm to tackle this problem, based on constellations of 3D centroids, and introduce a descriptor for very sparse 3D point clouds that can be used to match fruits across videos. Matching constellations instead of individual fruits is key to deal with non-rigidity, occlusions and challenging imagery with few distinct visual features to track. The results show that the proposed method can be successfully used to match fruits across videos and through time, and also to build an orchard map and later use it to locate the camera pose in 6DoF, thus providing a method for autonomous navigation of robots in the orchard and for selective fruit picking, for example.

  </details>



- **AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation**  
  Jin-Chuan Shi, Binhong Ye, Tao Liu, Junzhe He, Yangjinhui Xu, Xiaoyang Liu, Zeju Li, Hao Chen, Chunhua Shen  
  _2026-02-04_ · https://arxiv.org/abs/2602.04672v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

  </details>



- **S-MUSt3R: Sliding Multi-view 3D Reconstruction**  
  Leonid Antsfeld, Boris Chidlovskii, Yohann Cabon, Vincent Leroy, Jerome Revaud  
  _2026-02-04_ · https://arxiv.org/abs/2602.04517v1  
  <details><summary>Abstract</summary>

  The recent paradigm shift in 3D vision led to the rise of foundation models with remarkable capabilities in 3D perception from uncalibrated images. However, extending these models to large-scale RGB stream 3D reconstruction remains challenging due to memory limitations. This work proposes S-MUSt3R, a simple and efficient pipeline that extends the limits of foundation models for monocular 3D reconstruction. Our approach addresses the scalability bottleneck of foundation models through a simple strategy of sequence segmentation followed by segment alignment and lightweight loop closure optimization. Without model retraining, we benefit from remarkable 3D reconstruction capacities of MUSt3R model and achieve trajectory and reconstruction performance comparable to traditional methods with more complex architecture. We evaluate S-MUSt3R on TUM, 7-Scenes and proprietary robot navigation datasets and show that S-MUSt3R runs successfully on long RGB sequences and produces accurate and consistent 3D reconstruction. Our results highlight the potential of leveraging the MUSt3R model for scalable monocular 3D scene in real-world settings, with an important advantage of making predictions directly in the metric space.

  </details>



- **TrajVG: 3D Trajectory-Coupled Visual Geometry Learning**  
  Xingyu Miao, Weiguang Zhao, Tao Lu, Linning Yu, Mulin Yu, Yang Long, Jiangmiao Pang, Junting Dong  
  _2026-02-04_ · https://arxiv.org/abs/2602.04439v1  
  <details><summary>Abstract</summary>

  Feed-forward multi-frame 3D reconstruction models often degrade on videos with object motion. Global-reference becomes ambiguous under multiple motions, while the local pointmap relies heavily on estimated relative poses and can drift, causing cross-frame misalignment and duplicated structures. We propose TrajVG, a reconstruction framework that makes cross-frame 3D correspondence an explicit prediction by estimating camera-coordinate 3D trajectories. We couple sparse trajectories, per-frame local point maps, and relative camera poses with geometric consistency objectives: (i) bidirectional trajectory-pointmap consistency with controlled gradient flow, and (ii) a pose consistency objective driven by static track anchors that suppresses gradients from dynamic regions. To scale training to in-the-wild videos where 3D trajectory labels are scarce, we reformulate the same coupling constraints into self-supervised objectives using only pseudo 2D tracks, enabling unified training with mixed supervision. Extensive experiments across 3D tracking, pose estimation, pointmap reconstruction, and video depth show that TrajVG surpasses the current feedforward performance baseline.

  </details>



- **Finding NeMO: A Geometry-Aware Representation of Template Views for Few-Shot Perception**  
  Sebastian Jung, Leonard Klüpfel, Rudolph Triebel, Maximilian Durner  
  _2026-02-04_ · https://arxiv.org/abs/2602.04343v1  
  <details><summary>Abstract</summary>

  We present Neural Memory Object (NeMO), a novel object-centric representation that can be used to detect, segment and estimate the 6DoF pose of objects unseen during training using RGB images. Our method consists of an encoder that requires only a few RGB template views depicting an object to generate a sparse object-like point cloud using a learned UDF containing semantic and geometric information. Next, a decoder takes the object encoding together with a query image to generate a variety of dense predictions. Through extensive experiments, we show that our method can be used for few-shot object perception without requiring any camera-specific parameters or retraining on target data. Our proposed concept of outsourcing object information in a NeMO and using a single network for multiple perception tasks enhances interaction with novel objects, improving scalability and efficiency by enabling quick object onboarding without retraining or extensive pre-processing. We report competitive and state-of-the-art results on various datasets and perception tasks of the BOP benchmark, demonstrating the versatility of our approach. https://github.com/DLR-RM/nemo

  </details>



- **Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity**  
  Chenhe Du, Qing Wu, Xuanyu Tian, Jingyi Yu, Hongjiang Wei, Yuyao Zhang  
  _2026-02-04_ · https://arxiv.org/abs/2602.04162v1  
  <details><summary>Abstract</summary>

  3D medical imaging is in high demand and essential for clinical diagnosis and scientific research. Currently, diffusion models (DMs) have become an effective tool for medical imaging reconstruction thanks to their ability to learn rich, high-quality data priors. However, learning the 3D data distribution with DMs in medical imaging is challenging, not only due to the difficulties in data collection but also because of the significant computational burden during model training. A common compromise is to train the DMs on 2D data priors and reconstruct stacked 2D slices to address 3D medical inverse problems. However, the intrinsic randomness of diffusion sampling causes severe inter-slice discontinuities of reconstructed 3D volumes. Existing methods often enforce continuity regularizations along the z-axis, which introduces sensitive hyper-parameters and may lead to over-smoothing results. In this work, we revisit the origin of stochasticity in diffusion sampling and introduce Inter-Slice Consistent Stochasticity (ISCS), a simple yet effective strategy that encourages interslice consistency during diffusion sampling. Our key idea is to control the consistency of stochastic noise components during diffusion sampling, thereby aligning their sampling trajectories without adding any new loss terms or optimization steps. Importantly, the proposed ISCS is plug-and-play and can be dropped into any 2D trained diffusion based 3D reconstruction pipeline without additional computational cost. Experiments on several medical imaging problems show that our method can effectively improve the performance of medical 3D imaging problems based on 2D diffusion models. Our findings suggest that controlling inter-slice stochasticity is a principled and practically attractive route toward high-fidelity 3D medical imaging with 2D diffusion priors. The code is available at: https://github.com/duchenhe/ISCS

  </details>



- **SuperPoint-E: local features for 3D reconstruction via tracking adaptation in endoscopy**  
  O. Leon Barbed, José M. M. Montiel, Pascal Fua, Ana C. Murillo  
  _2026-02-04_ · https://arxiv.org/abs/2602.04108v1  
  <details><summary>Abstract</summary>

  In this work, we focus on boosting the feature extraction to improve the performance of Structure-from-Motion (SfM) in endoscopy videos. We present SuperPoint-E, a new local feature extraction method that, using our proposed Tracking Adaptation supervision strategy, significantly improves the quality of feature detection and description in endoscopy. Extensive experimentation on real endoscopy recordings studies our approach's most suitable configuration and evaluates SuperPoint-E feature quality. The comparison with other baselines also shows that our 3D reconstructions are denser and cover more and longer video segments because our detector fires more densely and our features are more likely to survive (i.e. higher detection precision). In addition, our descriptor is more discriminative, making the guided matching step almost redundant. The presented approach brings significant improvements in the 3D reconstructions obtained, via SfM on endoscopy videos, compared to the original SuperPoint and the gold standard SfM COLMAP pipeline.

  </details>



- **Seeing Through Clutter: Structured 3D Scene Reconstruction via Iterative Object Removal**  
  Rio Aguina-Kang, Kevin James Blackburn-Matzen, Thibault Groueix, Vladimir Kim, Matheus Gadelha  
  _2026-02-03_ · https://arxiv.org/abs/2602.04053v1  
  <details><summary>Abstract</summary>

  We present SeeingThroughClutter, a method for reconstructing structured 3D representations from single images by segmenting and modeling objects individually. Prior approaches rely on intermediate tasks such as semantic segmentation and depth estimation, which often underperform in complex scenes, particularly in the presence of occlusion and clutter. We address this by introducing an iterative object removal and reconstruction pipeline that decomposes complex scenes into a sequence of simpler subtasks. Using VLMs as orchestrators, foreground objects are removed one at a time via detection, segmentation, object removal, and 3D fitting. We show that removing objects allows for cleaner segmentations of subsequent objects, even in highly occluded scenes. Our method requires no task-specific training and benefits directly from ongoing advances in foundation models. We demonstrate stateof-the-art robustness on 3D-Front and ADE20K datasets. Project Page: https://rioak.github.io/seeingthroughclutter/

  </details>



- **AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting**  
  Joanna Kaleta, Bartosz Świrta, Kacper Kania, Przemysław Spurek, Marek Kowalski  
  _2026-02-03_ · https://arxiv.org/abs/2602.04043v1  
  <details><summary>Abstract</summary>

  The growing demand for rapid and scalable 3D asset creation has driven interest in feed-forward 3D reconstruction methods, with 3D Gaussian Splatting (3DGS) emerging as an effective scene representation. While recent approaches have demonstrated pose-free reconstruction from unposed image collections, integrating stylization or appearance control into such pipelines remains underexplored. Existing attempts largely rely on image-based conditioning, which limits both controllability and flexibility. In this work, we introduce AnyStyle, a feed-forward 3D reconstruction and stylization framework that enables pose-free, zero-shot stylization through multimodal conditioning. Our method supports both textual and visual style inputs, allowing users to control the scene appearance using natural language descriptions or reference images. We propose a modular stylization architecture that requires only minimal architectural modifications and can be integrated into existing feed-forward 3D reconstruction backbones. Experiments demonstrate that AnyStyle improves style controllability over prior feed-forward stylization methods while preserving high-quality geometric reconstruction. A user study further confirms that AnyStyle achieves superior stylization quality compared to an existing state-of-the-art approach. Repository: https://github.com/joaxkal/AnyStyle.

  </details>



- **EventNeuS: 3D Mesh Reconstruction from a Single Event Camera**  
  Shreyas Sachan, Viktor Rudnev, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik  
  _2026-02-03_ · https://arxiv.org/abs/2602.03847v1  
  <details><summary>Abstract</summary>

  Event cameras offer a considerable alternative to RGB cameras in many scenarios. While there are recent works on event-based novel-view synthesis, dense 3D mesh reconstruction remains scarcely explored and existing event-based techniques are severely limited in their 3D reconstruction accuracy. To address this limitation, we present EventNeuS, a self-supervised neural model for learning 3D representations from monocular colour event streams. Our approach, for the first time, combines 3D signed distance function and density field learning with event-based supervision. Furthermore, we introduce spherical harmonics encodings into our model for enhanced handling of view-dependent effects. EventNeuS outperforms existing approaches by a significant margin, achieving 34% lower Chamfer distance and 31% lower mean absolute error on average compared to the best previous method.

  </details>



- **Beyond the Vehicle: Cooperative Localization by Fusing Point Clouds for GPS-Challenged Urban Scenarios**  
  Kuo-Yi Chao, Ralph Rasshofer, Alois Christian Knoll  
  _2026-02-03_ · https://arxiv.org/abs/2602.03908v1  
  <details><summary>Abstract</summary>

  Accurate vehicle localization is a critical challenge in urban environments where GPS signals are often unreliable. This paper presents a cooperative multi-sensor and multi-modal localization approach to address this issue by fusing data from vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) systems. Our approach integrates cooperative data with a point cloud registration-based simultaneous localization and mapping (SLAM) algorithm. The system processes point clouds generated from diverse sensor modalities, including vehicle-mounted LiDAR and stereo cameras, as well as sensors deployed at intersections. By leveraging shared data from infrastructure, our method significantly improves localization accuracy and robustness in complex, GPS-noisy urban scenarios.

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



- **4DPC$^2$hat: Towards Dynamic Point Cloud Understanding with Failure-Aware Bootstrapping**  
  Xindan Zhang, Weilong Yan, Yufei Shi, Xuerui Qiu, Tao He, Ying Li, Ming Li, Hehe Fan  
  _2026-02-03_ · https://arxiv.org/abs/2602.03890v1  
  <details><summary>Abstract</summary>

  Point clouds provide a compact and expressive representation of 3D objects, and have recently been integrated into multimodal large language models (MLLMs). However, existing methods primarily focus on static objects, while understanding dynamic point cloud sequences remains largely unexplored. This limitation is mainly caused by the lack of large-scale cross-modal datasets and the difficulty of modeling motions in spatio-temporal contexts. To bridge this gap, we present 4DPC$^2$hat, the first MLLM tailored for dynamic point cloud understanding. To this end, we construct a large-scale cross-modal dataset 4DPC$^2$hat-200K via a meticulous two-stage pipeline consisting of topology-consistent 4D point construction and two-level captioning. The dataset contains over 44K dynamic object sequences, 700K point cloud frames, and 200K curated question-answer (QA) pairs, supporting inquiries about counting, temporal relationship, action, spatial relationship, and appearance. At the core of the framework, we introduce a Mamba-enhanced temporal reasoning MLLM to capture long-range dependencies and dynamic patterns among a point cloud sequence. Furthermore, we propose a failure-aware bootstrapping learning strategy that iteratively identifies model deficiencies and generates targeted QA supervision to continuously strengthen corresponding reasoning capabilities. Extensive experiments demonstrate that our 4DPC$^2$hat significantly improves action understanding and temporal reasoning compared with existing models, establishing a strong foundation for 4D dynamic point cloud understanding.

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


