# 3D Reconstruction

_Updated: 2026-04-03 07:35 UTC_

Total papers shown: **26**


---

- **CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects**  
  Jingliang Li, Jindou Jia, Tuo An, Chuhao Zhou, Xiangyu Chen, Shilin Shan, Boyu Ma, Bofan Lyu, Gen Li, Jianfei Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02060v1  
  <details><summary>Abstract</summary>

  When told to "cut the apple," a robot must choose the knife over nearby scissors, despite both objects affording the same cutting function. In real-world scenes, multiple objects may share identical affordances, yet only one is appropriate under the given task context. We call such cases confusing pairs. However, existing 3D affordance methods largely sidestep this challenge by evaluating isolated single objects, often with explicit category names provided in the query. We formalize Multi-Object Affordance Grounding under Intent-Driven Instructions, a new 3D affordance setting that requires predicting a per-point affordance mask on the correct object within a cluttered multi-object point cloud, conditioned on implicit natural language intent. To study this problem, we construct CompassAD, the first benchmark centered on implicit intent in confusable multi-object scenes. It comprises 30 confusing object pairs spanning 16 affordance types, 6,422 scenes, and 88K+ query-answer pairs. Furthermore, we propose CompassNet, a framework that incorporates two dedicated modules tailored to this task. Instance-bounded Cross Injection (ICI) constrains language-geometry alignment within object boundaries to prevent cross-object semantic leakage. Bi-level Contrastive Refinement (BCR) enforces discrimination at both geometric-group and point levels, sharpening distinctions between target and confusable surfaces. Extensive experiments demonstrate state-of-the-art results on both seen and unseen queries, and deployment on a robotic manipulator confirms effective transfer to real-world grasping in confusing multi-object scenes.

  </details>



- **Test-Time Adaptation for Height Completion via Self-Supervised ViT Features and Monocular Foundation Models**  
  Osher Rafaeli, Tal Svoray, Ariel Nahlieli  
  _2026-04-02_ · https://arxiv.org/abs/2604.02009v1  
  <details><summary>Abstract</summary>

  Accurate digital surface models (DSMs) are essential for many geospatial applications, including urban monitoring, environmental analyses, infrastructure management, and change detection. However, large-scale DSMs frequently contain incomplete or outdated regions due to acquisition limitations, reconstruction artifacts, or changes in the built environment. Traditional height completion approaches primarily rely on spatial interpolation or which assume spatial continuity and therefore fail when objects are missing. Recent learning-based approaches improve reconstruction quality but typically require supervised training on sensor-specific datasets, limiting their generalization across domains and sensing conditions. We propose Prior2DSM, a training-free framework for metric DSM completion that operates entirely at test time by leveraging foundation models. Unlike previous height completion approaches that require task-specific training, the proposed method combines self-supervised Vision Transformer (ViT) features from DINOv3 with monocular depth foundation models to propagate metric information from incomplete height priors through semantic feature-space correspondence. Test-time adaptation (TTA) is performed using parameter-efficient low-rank adaptation (LoRA) together with a lightweight multilayer perceptron (MLP), which predicts spatially varying scale and shift parameters to convert relative depth estimates into metric heights. Experiments demonstrate consistent improvements over interpolation based methods, prior-based rescaling height approaches, and state-of-the-art monocular depth estimation models. Prior2DSM reduces reconstruction error while preserving structural fidelity, achieving up to a 46% reduction in RMSE compared to linear fitting of MDE, and further enables DSM updating and coupled RGB-DSM generation.

  </details>



- **Enhanced Polarization Locking in VCSELs**  
  Zifeng Yuan, Dewen Zhang, Lei Shi, Yutong Liu, Aaron Danner  
  _2026-04-02_ · https://arxiv.org/abs/2604.01857v1  
  <details><summary>Abstract</summary>

  While optical injection locking (OIL) of vertical-cavity surface-emitting lasers (VCSELs) has been widely studied in the past, the polarization dynamics of OIL have received far less attention. Recent studies suggest that polarization locking via OIL could enable novel computational applications such as polarization-encoded Ising computers. However, the inherent polarization preference and limited polarization switchability of VCSELs hinder their use for such purposes. To address these challenges, we fabricate VCSELs with tailored oxide aperture designs and combine these with bias current tuning to study the overall impact on polarization locking. Experimental results demonstrate that this approach reduces the required injection power (to as low as 3.6 μW) and expands the locking range. To investigate the impact of the approach, the spin-flip model (SFM) is used to analyze the effects of amplitude anisotropy and bias current on polarization locking, demonstrating strong coherence with experimental results.

  </details>



- **PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency**  
  Leezy Han, Seunggyu Kim, Dongseok Shim, Hyeonbeom Lee  
  _2026-04-02_ · https://arxiv.org/abs/2604.01791v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation (MDE) has been widely adopted in the perception systems of autonomous vehicles and mobile robots. However, existing approaches often struggle to maintain temporal consistency in depth estimation across consecutive frames. This inconsistency not only causes jitter but can also lead to estimation failures when the depth range changes abruptly. To address these challenges, this paper proposes a consistency-aware monocular depth estimation framework that leverages wheel odometry from a mobile robot to achieve stable and coherent depth predictions over time. Specifically, we estimate camera pose and sparse depth from triangulation using optical flow between consecutive frames. The sparse depth estimates are used to update a recursive Bayesian estimate of the metric scale, which is then applied to rescale the relative depth predicted by a pre-trained depth estimation foundation model. The proposed method is evaluated on the KITTI, TartanAir, MS2, and our own dataset, demonstrating robust and accurate depth estimation performance.

  </details>



- **Control-DINO: Feature Space Conditioning for Controllable Image-to-Video Diffusion**  
  Edoardo A. Dominici, Thomas Deixelberger, Konstantinos Vardis, Markus Steinberger  
  _2026-04-02_ · https://arxiv.org/abs/2604.01761v1  
  <details><summary>Abstract</summary>

  Video models have recently been applied with success to problems in content generation, novel view synthesis, and, more broadly, world simulation. Many applications in generation and transfer rely on conditioning these models, typically through perceptual, geometric, or simple semantic signals, fundamentally using them as generative renderers. At the same time, high-dimensional features obtained from large-scale self-supervised learning on images or point clouds are increasingly used as a general-purpose interface for vision models. The connection between the two has been explored for subject specific editing, aligning and training video diffusion models, but not in the role of a more general conditioning signal for pretrained video diffusion models. Features obtained through self-supervised learning like DINO, contain a lot of entangled information about style, lighting and semantics of the scene. This makes them great at reconstruction tasks but limits their generative capabilities. In this paper, we show how we can use the features for tasks such as video domain transfer and video-from-3D generation. We introduce a lightweight architecture and training strategy that decouples appearance from other features that we wish to preserve, enabling robust control for appearance changes such as stylization and relighting. Furthermore, we show that low spatial resolution can be compensated by higher feature dimensionality, improving controllability in generative rendering from explicit spatial representations.

  </details>



- **TOL: Textual Localization with OpenStreetMap**  
  Youqi Liao, Shuhao Kang, Jingyu Xu, Olaf Wysocki, Yan Xia, Jianping Li, Zhen Dong, Bisheng Yang, Xieyuanli Chen  
  _2026-04-02_ · https://arxiv.org/abs/2604.01644v1  
  <details><summary>Abstract</summary>

  Natural language provides an intuitive way to express spatial intent in geospatial applications. While existing localization methods often rely on dense point cloud maps or high-resolution imagery, OpenStreetMap (OSM) offers a compact and freely available map representation that encodes rich semantic and structural information, making it well suited for large-scale localization. However, text-to-OSM (T2O) localization remains largely unexplored. In this paper, we formulate the T2O global localization task, which aims to estimate accurate 2 degree-of-freedom (DoF) positions in urban environments from textual scene descriptions without relying on geometric observations or GNSS-based initial location. To support the proposed task, we introduce TOL, a large-scale benchmark spanning multiple continents and diverse urban environments. TOL contains approximately 121K textual queries paired with OSM map tiles and covers about 316 km of road trajectories across Boston, Karlsruhe, and Singapore. We further propose TOLoc, a coarse-to-fine localization framework that explicitly models the semantics of surrounding objects and their directional information. In the coarse stage, direction-aware features are extracted from both textual descriptions and OSM tiles to construct global descriptors, which are used to retrieve candidate locations for the query. In the fine stage, the query text and top-1 retrieved tile are jointly processed, where a dedicated alignment module fuses textual descriptor and local map features to regress the 2-DoF pose. Experimental results demonstrate that TOLoc achieves strong localization performance, outperforming the best existing method by 6.53%, 9.93%, and 8.31% at 5m, 10m, and 25m thresholds, respectively, and shows strong generalization to unseen environments. Dataset, code and models will be publicly available at: https://github.com/WHU-USI3DV/TOL.

  </details>



- **F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling**  
  Morui Zhu, Mohammad Dehghani Tezerjani, Mátyás Szántó, Márton Vaitkus, Song Fu, Qing Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01605v1  
  <details><summary>Abstract</summary>

  We present F3DGS, a federated 3D Gaussian Splatting framework for decentralized multi-agent 3D reconstruction. Existing 3DGS pipelines assume centralized access to all observations, which limits their applicability in distributed robotic settings where agents operate independently, and centralized data aggregation may be restricted. Directly extending centralized training to multi-agent systems introduces communication overhead and geometric inconsistency. F3DGS first constructs a shared geometric scaffold by registering locally merged LiDAR point clouds from multiple clients to initialize a global 3DGS model. During federated optimization, Gaussian positions are fixed to preserve geometric alignment, while each client updates only appearance-related attributes, including covariance, opacity, and spherical harmonic coefficients. The server aggregates these updates using visibility-aware aggregation, weighting each client's contribution by how frequently it observed each Gaussian, resolving the partial-observability challenge inherent to multi-agent exploration. To evaluate decentralized reconstruction, we collect a multi-sequence indoor dataset with synchronized LiDAR, RGB, and IMU measurements. Experiments show that F3DGS achieves reconstruction quality comparable to centralized training while enabling distributed optimization across agents. The dataset, development kit, and source code will be publicly released.

  </details>



- **Towards Minimal Focal Stack in Shape from Focus**  
  Khurram Ashfaq, Muhammad Tariq Mahmood  
  _2026-04-02_ · https://arxiv.org/abs/2604.01603v1  
  <details><summary>Abstract</summary>

  Shape from Focus (SFF) is a depth reconstruction technique that estimates scene structure from focus variations observed across a focal stack, that is, a sequence of images captured at different focus settings. A key limitation of SFF methods is their reliance on densely sampled, large focal stacks, which limits their practical applicability. In this study, we propose a focal stack augmentation that enables SFF methods to estimate depth using a reduced stack of just two images, without sacrificing precision. We introduce a simple yet effective physics-based focal stack augmentation that enriches the stack with two auxiliary cues: an all-in-focus (AiF) image estimated from two input images, and Energy-of-Difference (EOD) maps, computed as the energy of differences between the AiF and input images. Furthermore, we propose a deep network that computes a deep focus volume from the augmented focal stacks and iteratively refines depth using convolutional Gated Recurrent Units (ConvGRUs) at multiple scales. Extensive experiments on both synthetic and real-world datasets demonstrate that the proposed augmentation benefits existing state-of-the-art SFF models, enabling them to achieve comparable accuracy. The results also show that our approach maintains state-of-the-art performance with a minimal stack size.

  </details>



- **Riemannian and Symplectic Geometry for Hierarchical Text-Driven Place Recognition**  
  Tianyi Shang, Zhenyu Li  
  _2026-04-02_ · https://arxiv.org/abs/2604.01598v1  
  <details><summary>Abstract</summary>

  Text-to-point-cloud localization enables robots to understand spatial positions through natural language descriptions, which is crucial for human-robot collaboration in applications such as autonomous driving and last-mile delivery. However, existing methods employ pooled global descriptors for similarity retrieval, which suffer from severe information loss and fail to capture discriminative scene structures. To address these issues, we propose SympLoc, a novel coarse-to-fine localization framework with multi-level alignment in the coarse stage. Different from previous methods that rely solely on global descriptors, our coarse stage consists of three complementary alignment levels: 1) Instance-level alignment establishes direct correspondence between individual object instances in point clouds and textual hints through Riemannian self-attention in hyperbolic space; 2) Relation-level alignment explicitly models pairwise spatial relationships between objects using the Information-Symplectic Relation Encoder (ISRE), which reformulates relation features through Fisher-Rao metric and Hamiltonian dynamics for uncertainty-aware geometrically consistent propagation; 3) Global-level alignment synthesizes discriminative global descriptors via the Spectral Manifold Transform (SMT) that extracts structural invariants through graph spectral analysis. This hierarchical alignment strategy progressively captures fine-grained to coarse-grained scene semantics, enabling robust cross-modal retrieval. Extensive experiments on the KITTI360Pose dataset demonstrate that SympLoc achieves a 19% improvement in Top-1 recall@10m compared to existing state-of-the-art approaches.

  </details>



- **UniRecGen: Unifying Multi-View 3D Reconstruction and Generation**  
  Zhisheng Huang, Jiahao Chen, Cheng Lin, Chenyu Hu, Hanzhuo Huang, Zhengming Yu, Mengfei Li, Yuheng Liu, Zekai Gu, Zibo Zhao, et al.  
  _2026-04-01_ · https://arxiv.org/abs/2604.01479v1  
  <details><summary>Abstract</summary>

  Sparse-view 3D modeling represents a fundamental tension between reconstruction fidelity and generative plausibility. While feed-forward reconstruction excels in efficiency and input alignment, it often lacks the global priors needed for structural completeness. Conversely, diffusion-based generation provides rich geometric details but struggles with multi-view consistency. We present UniRecGen, a unified framework that integrates these two paradigms into a single cooperative system. To overcome inherent conflicts in coordinate spaces, 3D representations, and training objectives, we align both models within a shared canonical space. We employ disentangled cooperative learning, which maintains stable training while enabling seamless collaboration during inference. Specifically, the reconstruction module is adapted to provide canonical geometric anchors, while the diffusion generator leverages latent-augmented conditioning to refine and complete the geometric structure. Experimental results demonstrate that UniRecGen achieves superior fidelity and robustness, outperforming existing methods in creating complete and consistent 3D models from sparse observations.

  </details>



- **LESV: Language Embedded Sparse Voxel Fusion for Open-Vocabulary 3D Scene Understanding**  
  Fusang Wang, Nathan Piasco, Moussab Bennehar, Luis Roldão, Dzmitry Tsishkou, Fabien Moutarde  
  _2026-04-01_ · https://arxiv.org/abs/2604.01388v1  
  <details><summary>Abstract</summary>

  Recent advancements in open-vocabulary 3D scene understanding heavily rely on 3D Gaussian Splatting (3DGS) to register vision-language features into 3D space. However, we identify two critical limitations in these approaches: the spatial ambiguity arising from unstructured, overlapping Gaussians which necessitates probabilistic feature registration, and the multi-level semantic ambiguity caused by pooling features over object-level masks, which dilutes fine-grained details. To address these challenges, we present a novel framework that leverages Sparse Voxel Rasterization (SVRaster) as a structured, disjoint geometry representation. By regularizing SVRaster with monocular depth and normal priors, we establish a stable geometric foundation. This enables a deterministic, confidence-aware feature registration process and suppresses the semantic bleeding artifact common in 3DGS. Furthermore, we resolve multi-level ambiguity by exploiting the emerging dense alignment properties of foundation model AM-RADIO, avoiding the computational overhead of hierarchical training methods. Our approach achieves state-of-the-art performance on Open Vocabulary 3D Object Retrieval and Point Cloud Understanding benchmarks, particularly excelling on fine-grained queries where registration methods typically fail.

  </details>



- **IGLOSS: Image Generation for Lidar Open-vocabulary Semantic Segmentation**  
  Nermin Samet, Gilles Puy, Renaud Marlet  
  _2026-04-01_ · https://arxiv.org/abs/2604.01361v1  
  <details><summary>Abstract</summary>

  This paper presents a new method for the zero-shot open-vocabulary semantic segmentation (OVSS) of 3D automotive lidar data. To circumvent the recognized image-text modality gap that is intrinsic to approaches based on Vision Language Models (VLMs) such as CLIP, our method relies instead on image generation from text, to create prototype images. Given a 3D network distilled from a 2D Vision Foundation Model (VFM), we then label a point cloud by matching 3D point features with 2D image features of these prototypes. Our method is state-of-the-art for OVSS on nuScenes and SemanticKITTI. Code, pre-trained models, and generated images are available at https://github.com/valeoai/IGLOSS.

  </details>



- **Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction**  
  Jorge Condor, Nicolas Moenne-Loccoz, Merlin Nimier-David, Piotr Didyk, Zan Gojcic, Qi Wu  
  _2026-04-01_ · https://arxiv.org/abs/2604.01204v1  
  <details><summary>Abstract</summary>

  Primitive-based methods such as 3D Gaussian Splatting have recently become the state-of-the-art for novel-view synthesis and related reconstruction tasks. Compared to neural fields, these representations are more flexible, adaptive, and scale better to large scenes. However, the limited expressivity of individual primitives makes modeling high-frequency detail challenging. We introduce Neural Harmonic Textures, a neural representation approach that anchors latent feature vectors on a virtual scaffold surrounding each primitive. These features are interpolated within the primitive at ray intersection points. Inspired by Fourier analysis, we apply periodic activations to the interpolated features, turning alpha blending into a weighted sum of harmonic components. The resulting signal is then decoded in a single deferred pass using a small neural network, significantly reducing computational cost. Neural Harmonic Textures yield state-of-the-art results in real-time novel view synthesis while bridging the gap between primitive- and neural-field-based reconstruction. Our method integrates seamlessly into existing primitive-based pipelines such as 3DGUT, Triangle Splatting, and 2DGS. We further demonstrate its generality with applications to 2D image fitting and semantic reconstruction.

  </details>



- **Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects**  
  Hanzhe Liang, Luocheng Zhang, Junyang Xia, HanLiang Zhou, Bingyang Guo, Yingxi Xie, Can Gao, Ruiyun Yu, Jinbao Wang, Pan Li  
  _2026-04-01_ · https://arxiv.org/abs/2604.01171v1  
  <details><summary>Abstract</summary>

  Although self-supervised 3D anomaly detection assumes that acquiring high-precision point clouds is computationally expensive, in real manufacturing scenarios it is often feasible to collect a limited number of anomalous samples. Therefore, we study open-set supervised 3D anomaly detection, where the model is trained with only normal samples and a small number of known anomalous samples, aiming to identify unknown anomalies at test time. We present Open-Industry, a high-quality industrial dataset containing 15 categories, each with five real anomaly types collected from production lines. We first adapt general open-set anomaly detection methods to accommodate 3D point cloud inputs better. Building upon this, we propose Open3D-AD, a point-cloud-oriented approach that leverages normal samples, simulated anomalies, and partially observed real anomalies to model the probability density distributions of normal and anomalous data. Then, we introduce a simple Correspondence Distributions Subsampling to reduce the overlap between normal and non-normal distributions, enabling stronger dual distributions modeling. Based on these contributions, we establish a comprehensive benchmark and evaluate the proposed method extensively on Open-Industry as well as established datasets including Real3D-AD and Anomaly-ShapeNet. Benchmark results and ablation studies demonstrate the effectiveness of Open3D-AD and further reveal the potential of open-set supervised 3D anomaly detection.

  </details>



- **ReinDriveGen: Reinforcement Post-Training for Out-of-Distribution Driving Scene Generation**  
  Hao Zhang, Lue Fan, Weikang Bian, Zehuan Wu, Lewei Lu, Zhaoxiang Zhang, Hongsheng Li  
  _2026-04-01_ · https://arxiv.org/abs/2604.01129v1  
  <details><summary>Abstract</summary>

  We present ReinDriveGen, a framework that enables full controllability over dynamic driving scenes, allowing users to freely edit actor trajectories to simulate safety-critical corner cases such as front-vehicle collisions, drifting cars, vehicles spinning out of control, pedestrians jaywalking, and cyclists cutting across lanes. Our approach constructs a dynamic 3D point cloud scene from multi-frame LiDAR data, introduces a vehicle completion module to reconstruct full 360° geometry from partial observations, and renders the edited scene into 2D condition images that guide a video diffusion model to synthesize realistic driving videos. Since such edited scenarios inevitably fall outside the training distribution, we further propose an RL-based post-training strategy with a pairwise preference model and a pairwise reward mechanism, enabling robust quality improvement under out-of-distribution conditions without ground-truth supervision. Extensive experiments demonstrate that ReinDriveGen outperforms existing approaches on edited driving scenarios and achieves state-of-the-art results on novel ego viewpoint synthesis.

  </details>



- **Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation**  
  Reyhaneh Ahani Manghotay, Jie Liang  
  _2026-04-01_ · https://arxiv.org/abs/2604.01118v1  
  <details><summary>Abstract</summary>

  Leveraging the rich semantic features of vision-language models (VLMs) like CLIP for monocular depth estimation tasks is a promising direction, yet often requires extensive fine-tuning or lacks geometric precision. We present a parameter-efficient framework, named MoA-DepthCLIP, that adapts pretrained CLIP representations for monocular depth estimation with minimal supervision. Our method integrates a lightweight Mixture-of-Adapters (MoA) module into the pretrained Vision Transformer (ViT-B/32) backbone combined with selective fine-tuning of the final layers. This design enables spatially-aware adaptation, guided by a global semantic context vector and a hybrid prediction architecture that synergizes depth bin classification with direct regression. To enhance structural accuracy, we employ a composite loss function that enforces geometric constraints. On the NYU Depth V2 benchmark, MoA-DepthCLIP achieves competitive results, significantly outperforming the DepthCLIP baseline by improving the $δ_1$ accuracy from 0.390 to 0.745 and reducing the RMSE from 1.176 to 0.520. These results are achieved while requiring substantially few trainable parameters, demonstrating that lightweight, prompt-guided MoA is a highly effective strategy for transferring VLM knowledge to fine-grained monocular depth estimation tasks.

  </details>



- **Sub-metre Lunar DEM Generation and Validation from Chandrayaan-2 OHRC Multi-View Imagery Using Open-Source Photogrammetry**  
  Aaranay Aadi, Jai Singla, Nitant Dube, Oleg Alexandrov  
  _2026-04-01_ · https://arxiv.org/abs/2604.01032v1  
  <details><summary>Abstract</summary>

  High-resolution digital elevation models (DEMs) of the lunar surface are essential for surface mobility planning, landing site characterization, and planetary science. The Orbiter High Resolution Camera (OHRC) on board Chandrayaan-2 has the best ground sampling capabilities of any lunar orbital imaging currently in use by acquiring panchromatic imagery at a resolution of roughly 20-30 cm per pixel. This work presents, for the first time, the generation of sub-metre DEMs from OHRC multi-view imagery using an exclusively open-source pipeline. Candidate stereo pairs are identified from non-paired OHRC archives through geometric analysis of image metadata, employing baseline-to-height (B/H) ratio computation and convergence angle estimation. Dense stereo correspondence and ray triangulation are then applied to generate point clouds, which are gridded into DEMs at effective spatial resolutions between approximately 24 and 54 cm across five geographically distributed lunar sites. Absolute elevation consistency is established through Iterative Closest Point (ICP) alignment against Lunar Reconnaissance Orbiter Narrow Angle Camera (NAC) Digital Terrain Models, followed by constant-bias offset correction. Validation against NAC reference terrain yields a vertical RMSE of 5.85 m (at native OHRC resolution), and a horizontal accuracy of less than 30 cm assessed by planimetric feature matching.

  </details>



- **EgoSim: Egocentric World Simulator for Embodied Interaction Generation**  
  Jinkun Hao, Mingda Jia, Ruiyan Wang, Xihui Liu, Ran Yi, Lizhuang Ma, Jiangmiao Pang, Xudong Xu  
  _2026-04-01_ · https://arxiv.org/abs/2604.01001v1  
  <details><summary>Abstract</summary>

  We introduce EgoSim, a closed-loop egocentric world simulator that generates spatially consistent interaction videos and persistently updates the underlying 3D scene state for continuous simulation. Existing egocentric simulators either lack explicit 3D grounding, causing structural drift under viewpoint changes, or treat the scene as static, failing to update world states across multi-stage interactions. EgoSim addresses both limitations by modeling 3D scenes as updatable world states. We generate embodiment interactions via a Geometry-action-aware Observation Simulation model, with spatial consistency from an Interaction-aware State Updating module. To overcome the critical data bottleneck posed by the difficulty in acquiring densely aligned scene-interaction training pairs, we design a scalable pipeline that extracts static point clouds, camera trajectories, and embodiment actions from in-the-wild large-scale monocular egocentric videos. We further introduce EgoCap, a capture system that enables low-cost real-world data collection with uncalibrated smartphones. Extensive experiments demonstrate that EgoSim significantly outperforms existing methods in terms of visual quality, spatial consistency, and generalization to complex scenes and in-the-wild dexterous interactions, while supporting cross-embodiment transfer to robotic manipulation. Codes and datasets will be open soon. The project page is at egosimulator.github.io.

  </details>



- **Shape Representation using Gaussian Process mixture models**  
  Panagiotis Sapoutzoglou, George Terzakis, Georgios Floros, Maria Pateraki  
  _2026-04-01_ · https://arxiv.org/abs/2604.00862v1  
  <details><summary>Abstract</summary>

  Traditional explicit 3D representations, such as point clouds and meshes, demand significant storage to capture fine geometric details and require complex indexing systems for surface lookups, making functional representations an efficient, compact, and continuous alternative. In this work, we propose a novel, object-specific functional shape representation that models surface geometry with Gaussian Process (GP) mixture models. Rather than relying on computationally heavy neural architectures, our method is lightweight, leveraging GPs to learn continuous directional distance fields from sparsely sampled point clouds. We capture complex topologies by anchoring local GP priors at strategic reference points, which can be flexibly extracted using any structural decomposition method (e.g. skeletonization, distance-based clustering). Extensive evaluations on the ShapeNetCore and IndustryShapes datasets demonstrate that our method can efficiently and accurately represent complex geometries.

  </details>



- **Sparkle: A Robust and Versatile Representation for Point Cloud based Human Motion Capture**  
  Yiming Ren, Yujing Sun, Aoru Xue, Kwok-Yan Lam, Yuexin Ma  
  _2026-04-01_ · https://arxiv.org/abs/2604.00857v1  
  <details><summary>Abstract</summary>

  Point cloud-based motion capture leverages rich spatial geometry and privacy-preserving sensing, but learning robust representations from noisy, unstructured point clouds remains challenging. Existing approaches face a struggle trade-off between point-based methods (geometrically detailed but noisy) and skeleton-based ones (robust but oversimplified). We address the fundamental challenge: how to construct an effective representation for human motion capture that can balance expressiveness and robustness. In this paper, we propose Sparkle, a structured representation unifying skeletal joints and surface anchors with explicit kinematic-geometric factorization. Our framework, SparkleMotion, learns this representation through hierarchical modules embedding geometric continuity and kinematic constraints. By explicitly disentangling internal kinematic structure from external surface geometry, SparkleMotion achieves state-of-the-art performance not only in accuracy but crucially in robustness and generalization under severe domain shifts, noise, and occlusion. Extensive experiments demonstrate our superiority across diverse sensor types and challenging real-world scenarios.

  </details>



- **MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data**  
  Clémentine Grethen, Yuang Shi, Simone Gasparini, Géraldine Morin  
  _2026-04-01_ · https://arxiv.org/abs/2604.00682v1  
  <details><summary>Abstract</summary>

  Accurate perception of lunar surfaces is critical for modern lunar exploration missions. However, developing robust learning-based perception systems is hindered by the lack of datasets that provide both geometric and photometric supervision. Existing lunar datasets typically lack either geometric ground truth, photometric realism, illumination diversity, or large-scale coverage. In this paper, we introduce MoonAnything, a unified benchmark built on real lunar topography with physically-based rendering, providing the first comprehensive geometric and photometric supervision under diverse illumination with large scale. The benchmark comprises two complementary sub-datasets : i) LunarGeo provides stereo images with corresponding dense depth maps and camera calibration enabling 3D reconstruction and pose estimation; ii) LunarPhoto provides photorealistic images using a spatially-varying BRDF model, along with multi-illumination renderings under real solar configurations, enabling reflectance estimation and illumination-robust perception. Together, these datasets offer over 130K samples with comprehensive supervision. Beyond lunar applications, MoonAnything offers a unique setting and challenging testbed for algorithms under low-textured, high-contrast conditions and applies to other airless celestial bodies and could generalize beyond. We establish baselines using state-of-the-art methods and release the complete dataset along with generation tools to support community extension: https://github.com/clementinegrethen/MoonAnything.

  </details>



- **Reliev3R: Relieving Feed-forward Reconstruction from Multi-View Geometric Annotations**  
  Youyu Chen, Junjun Jiang, Yueru Luo, Kui Jiang, Xianming Liu, Xu Yan, Dave Zhenyu Chen  
  _2026-04-01_ · https://arxiv.org/abs/2604.00548v1  
  <details><summary>Abstract</summary>

  With recent advances, Feed-forward Reconstruction Models (FFRMs) have demonstrated great potential in reconstruction quality and adaptiveness to multiple downstream tasks. However, the excessive reliance on multi-view geometric annotations, e.g. 3D point maps and camera poses, makes the fully-supervised training scheme of FFRMs difficult to scale up. In this paper, we propose Reliev3R, a weakly-supervised paradigm for training FFRMs from scratch without cost-prohibitive multi-view geometric annotations. Relieving the reliance on geometric sensory data and compute-exhaustive structure-from-motion preprocessing, our method draws 3D knowledge directly from monocular relative depths and image sparse correspondences given by zero-shot predictions of pretrained models. At the core of Reliev3R, we design an ambiguity-aware relative depth loss and a trigonometry-based reprojection loss to facilitate supervision for multi-view geometric consistency. Training from scratch with the less data, Reliev3R catches up with its fully-supervised sibling models, taking a step towards low-cost 3D reconstruction supervisions and scalable FFRMs.

  </details>



- **Think, Act, Build: An Agentic Framework with Vision Language Models for Zero-Shot 3D Visual Grounding**  
  Haibo Wang, Zihao Lin, Zhiyang Xu, Lifu Huang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00528v2  
  <details><summary>Abstract</summary>

  3D Visual Grounding (3D-VG) aims to localize objects in 3D scenes via natural language descriptions. While recent advancements leveraging Vision-Language Models (VLMs) have explored zero-shot possibilities, they typically suffer from a static workflow relying on preprocessed 3D point clouds, essentially degrading grounding into proposal matching. To bypass this reliance, our core motivation is to decouple the task: leveraging 2D VLMs to resolve complex spatial semantics, while relying on deterministic multi-view geometry to instantiate the 3D structure. Driven by this insight, we propose "Think, Act, Build (TAB)", a dynamic agentic framework that reformulates 3D-VG tasks as a generative 2D-to-3D reconstruction paradigm operating directly on raw RGB-D streams. Specifically, guided by a specialized 3D-VG skill, our VLM agent dynamically invokes visual tools to track and reconstruct the target across 2D frames. Crucially, to overcome the multi-view coverage deficit caused by strict VLM semantic tracking, we introduce the Semantic-Anchored Geometric Expansion, a mechanism that first anchors the target in a reference video clip and then leverages multi-view geometry to propagate its spatial location across unobserved frames. This enables the agent to "Build" the target's 3D representation by aggregating these multi-view features via camera parameters, directly mapping 2D visual cues to 3D coordinates. Furthermore, to ensure rigorous assessment, we identify flaws such as reference ambiguity and category errors in existing benchmarks and manually refine the incorrect queries. Extensive experiments on ScanRefer and Nr3D demonstrate that our framework, relying entirely on open-source models, significantly outperforms previous zero-shot methods and even surpasses fully supervised baselines.

  </details>



- **Neural Reconstruction of LiDAR Point Clouds under Jamming Attacks via Full-Waveform Representation and Simultaneous Laser Sensing**  
  Ryo Yoshida, Takami Sato, Wenlun Zhang, Yuki Hayakawa, Shota Nagai, Takahiro Kado, Taro Beppu, Ibuki Fujioka, Yunshan Zhong, Kentaro Yoshioka  
  _2026-04-01_ · https://arxiv.org/abs/2604.00371v1  
  <details><summary>Abstract</summary>

  LiDAR sensors are critical for autonomous driving perception, yet remain vulnerable to spoofing attacks. Jamming attacks inject high-frequency laser pulses that completely blind LiDAR sensors by overwhelming authentic returns with malicious signals. We discover that while point clouds become randomized, the underlying full-waveform data retains distinguishable signatures between attack and legitimate signals. In this work, we propose PULSAR-Net, capable of reconstructing authentic point clouds under jamming attacks by leveraging previously underutilized intermediate full-waveform representations and simultaneous laser sensing in modern LiDAR systems. PULSAR-Net adopts a novel U-Net architecture with axial spatial attention mechanisms specifically designed to identify attack-induced signals from authentic object returns in the full-waveform representation. To address the lack of full-waveform representations in existing LiDAR datasets under jamming attacks, we introduce a physics-aware dataset generation pipeline that synthesizes realistic full-waveform representations under jamming attacks. Despite being trained exclusively on synthetic data, PULSAR-Net achieves reconstruction rates of 92% and 73% for vehicles obscured by jamming attacks in real-world static and driving scenarios, respectively.

  </details>



- **Dual Contouring of Signed Distance Data**  
  Xiana Carrera, Ningna Wang, Christopher Batty, Oded Stein, Silvia Sellán  
  _2026-03-31_ · https://arxiv.org/abs/2604.00157v1  
  <details><summary>Abstract</summary>

  We propose an algorithm to reconstruct explicit polygonal meshes from discretely sampled Signed Distance Function (SDF) data, which is especially effective at recovering sharp features. Building on the traditional Dual Contouring of Hermite Data method, we design and solve a quadratic optimization problem to decide the optimal placement of the mesh's vertices within each cell of a regular grid. Critically, this optimization relies solely on discretely sampled SDF data, without requiring arbitrary access to the function, gradient information, or training on large-scale datasets. Our method sets a new state of the art in surface reconstruction from SDFs at medium and high resolutions, and opens the door for applications in 3D modeling and design.

  </details>



- **OmniRoam: World Wandering via Long-Horizon Panoramic Video Generation**  
  Yuheng Liu, Xin Lin, Xinke Li, Baihan Yang, Chen Wang, Kalyan Sunkavalli, Yannick Hold-Geoffroy, Hao Tan, Kai Zhang, Xiaohui Xie, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2603.30045v1  
  <details><summary>Abstract</summary>

  Modeling scenes using video generation models has garnered growing research interest in recent years. However, most existing approaches rely on perspective video models that synthesize only limited observations of a scene, leading to issues of completeness and global consistency. We propose OmniRoam, a controllable panoramic video generation framework that exploits the rich per-frame scene coverage and inherent long-term spatial and temporal consistency of panoramic representation, enabling long-horizon scene wandering. Our framework begins with a preview stage, where a trajectory-controlled video generation model creates a quick overview of the scene from a given input image or video. Then, in the refine stage, this video is temporally extended and spatially upsampled to produce long-range, high-resolution videos, thus enabling high-fidelity world wandering. To train our model, we introduce two panoramic video datasets that incorporate both synthetic and real-world captured videos. Experiments show that our framework consistently outperforms state-of-the-art methods in terms of visual quality, controllability, and long-term scene consistency, both qualitatively and quantitatively. We further showcase several extensions of this framework, including real-time video generation and 3D reconstruction. Code is available at https://github.com/yuhengliu02/OmniRoam.

  </details>


