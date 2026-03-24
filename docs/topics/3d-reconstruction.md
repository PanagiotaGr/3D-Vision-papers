# 3D Reconstruction

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **16**


---

- **GenOpticalFlow: A Generative Approach to Unsupervised Optical Flow Learning**  
  Yixuan Luo, Feng Qiao, Zhexiao Xiong, Yanjing Li, Nathan Jacobs  
  _2026-03-23_ · https://arxiv.org/abs/2603.22270v1  
  <details><summary>Abstract</summary>

  Optical flow estimation is a fundamental problem in computer vision, yet the reliance on expensive ground-truth annotations limits the scalability of supervised approaches. Although unsupervised and semi-supervised methods alleviate this issue, they often suffer from unreliable supervision signals based on brightness constancy and smoothness assumptions, leading to inaccurate motion estimation in complex real-world scenarios. To overcome these limitations, we introduce \textbf{\modelname}, a novel framework that synthesizes large-scale, perfectly aligned frame--flow data pairs for supervised optical flow training without human annotations. Specifically, our method leverages a pre-trained depth estimation network to generate pseudo optical flows, which serve as conditioning inputs for a next-frame generation model trained to produce high-fidelity, pixel-aligned subsequent frames. This process enables the creation of abundant, high-quality synthetic data with precise motion correspondence. Furthermore, we propose an \textit{inconsistent pixel filtering} strategy that identifies and removes unreliable pixels in generated frames, effectively enhancing fine-tuning performance on real-world datasets. Extensive experiments on KITTI2012, KITTI2015, and Sintel demonstrate that \textbf{\modelname} achieves competitive or superior results compared to existing unsupervised and semi-supervised approaches, highlighting its potential as a scalable and annotation-free solution for optical flow learning. We will release our code upon acceptance.

  </details>



- **Riverine Land Cover Mapping through Semantic Segmentation of Multispectral Point Clouds**  
  Sopitta Thurachen, Josef Taher, Matti Lehtomäki, Leena Matikainen, Linnea Blåfield, Mikel Calle Navarro, Antero Kukko, Tomi Westerlund, Harri Kaartinen  
  _2026-03-23_ · https://arxiv.org/abs/2603.22230v1  
  <details><summary>Abstract</summary>

  Accurate land cover mapping in riverine environments is essential for effective river management, ecological understanding, and geomorphic change monitoring. This study explores the use of Point Transformer v2 (PTv2), an advanced deep neural network architecture designed for point cloud data, for land cover mapping through semantic segmentation of multispectral LiDAR data in real-world riverine environments. We utilize the geometric and spectral information from the 3-channel LiDAR point cloud to map land cover classes, including sand, gravel, low vegetation, high vegetation, forest floor, and water. The PTv2 model was trained and evaluated on point cloud data from the Oulanka river in northern Finland using both geometry and spectral features. To improve the model's generalization in new riverine environments, we additionally investigate multi-dataset training that adds sparsely annotated data from an additional river dataset. Results demonstrated that using the full-feature configuration resulted in performance with a mean Intersection over Union (mIoU) of 0.950, significantly outperforming the geometry baseline. Other ablation studies revealed that intensity and reflectance features were the key for accurate land cover mapping. The multi-dataset training experiment showed improved generalization performance, suggesting potential for developing more robust models despite limited high-quality annotated data. Our work demonstrates the potential of applying transformer-based architectures to multispectral point clouds in riverine environments. The approach offers new capabilities for monitoring sediment transport and other river management applications.

  </details>



- **Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models**  
  Meiqi Wu, Zhixin Cai, Fufangchen Zhao, Xiaokun Feng, Rujing Dang, Bingze Song, Ruitian Tian, Jiashu Zhu, Jiachen Lei, Hao Dou, et al.  
  _2026-03-23_ · https://arxiv.org/abs/2603.22212v1  
  <details><summary>Abstract</summary>

  Video--based world models have emerged along two dominant paradigms: video generation and 3D reconstruction. However, existing evaluation benchmarks either focus narrowly on visual fidelity and text--video alignment for generative models, or rely on static 3D reconstruction metrics that fundamentally neglect temporal dynamics. We argue that the future of world modeling lies in 4D generation, which jointly models spatial structure and temporal evolution. In this paradigm, the core capability is interactive response: the ability to faithfully reflect how interaction actions drive state transitions across space and time. Yet no existing benchmark systematically evaluates this critical dimension. To address this gap, we propose Omni--WorldBench, a comprehensive benchmark specifically designed to evaluate the interactive response capabilities of world models in 4D settings. Omni--WorldBench comprises two key components: Omni--WorldSuite, a systematic prompt suite spanning diverse interaction levels and scene types; and Omni--Metrics, an agent-based evaluation framework that quantifies world modeling capabilities by measuring the causal impact of interaction actions on both final outcomes and intermediate state evolution trajectories. We conduct extensive evaluations of 18 representative world models across multiple paradigms. Our analysis reveals critical limitations of current world models in interactive response, providing actionable insights for future research. Omni-WorldBench will be publicly released to foster progress in interactive 4D world modeling.

  </details>



- **Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning**  
  Xingyu Zhu, Liang Yi, Shuo Wang, Wenbo Zhu, Yonglinag Wu, Beier Zhu, Hanwang Zhang  
  _2026-03-23_ · https://arxiv.org/abs/2603.22070v1  
  <details><summary>Abstract</summary>

  Multimodal 3D vision-language models show strong generalization across diverse 3D tasks, but their performance still degrades notably under domain shifts. This has motivated recent studies on test-time adaptation (TTA), which enables models to adapt online using test-time data. Among existing TTA methods, cache-based mechanisms are widely adopted for leveraging previously observed samples in online prediction refinement. However, they store only limited historical information, leading to progressive information loss as the test stream evolves. In addition, their prediction logits are fused heuristically, making adaptation unstable. To address these limitations, we propose BayesMM, a Multimodal Bayesian Distribution Learning framework for test-time point cloud analysis. BayesMM models textual priors and streaming visual features of each class as Gaussian distributions: textual parameters are derived from semantic prompts, while visual parameters are updated online with arriving samples. The two modalities are fused via Bayesian model averaging, which automatically adjusts their contributions based on posterior evidence, yielding a unified prediction that adapts continually to evolving test-time data without training. Extensive experiments on multiple point cloud benchmarks demonstrate that BayesMM maintains robustness under distributional shifts, yielding over 4% average improvement.

  </details>



- **GTSR: Subsurface Scattering Awared 3D Gaussians for Translucent Surface Reconstruction**  
  Youwen Yuan, Xi Zhao  
  _2026-03-23_ · https://arxiv.org/abs/2603.22036v1  
  <details><summary>Abstract</summary>

  Reconstructing translucent objects from multi-view images is a difficult problem. Previously, researchers have used differentiable path tracing and the neural implicit field, which require relatively large computational costs. Recently, many works have achieved good reconstruction results for opaque objects based on a 3DGS pipeline with much higher efficiency. However, such methods have difficulty dealing with translucent objects, because they do not consider the optical properties of translucent objects. In this paper, we propose a novel 3DGS-based pipeline (GTSR) to reconstruct the surface geometry of translucent objects. GTSR combines two sets of Gaussians, surface and interior Gaussians, which are used to model the surface and scattering color when lights pass translucent objects. To render the appearance of translucent objects, we introduce a method that uses the Fresnel term to blend two sets of Gaussians. Furthermore, to improve the reconstructed details of non-contour areas, we introduce the Disney BSDF model with deferred rendering to enhance constraints of the normal and depth. Experimental results demonstrate that our method outperforms baseline reconstruction methods on the NeuralTO Syn dataset while showing great real-time rendering performance. We also extend the dataset with new translucent objects of varying material properties and demonstrate our method can adapt to different translucent materials.

  </details>



- **SARe: Structure-Aware Large-Scale 3D Fragment Reassembly**  
  Hanze Jia, Chunshi Wang, Yuxiao Yang, Zhonghua Jiang, Yawei Luo, Shuainan Ye, Tan Tang  
  _2026-03-23_ · https://arxiv.org/abs/2603.21611v1  
  <details><summary>Abstract</summary>

  3D fragment reassembly aims to recover the rigid poses of unordered fragment point clouds or meshes in a common object coordinate system to reconstruct the complete shape. The problem becomes particularly challenging as the number of fragments grows, since the target shape is unknown and fragments provide weak semantic cues. Existing end-to-end approaches are prone to cascading failures due to unreliable contact reasoning, most notably inaccurate fragment adjacencies. To address this, we propose Structure-Aware Reassembly (SARe), a generative framework with SARe-Gen for Euclidean-space assembly generation and SARe-Refine for inference-time refinement, with explicit contact modeling. SARe-Gen jointly predicts fracture-surface token probabilities and an inter-fragment contact graph to localize contact regions and infer candidate adjacencies. It adopts a query-point-based conditioning scheme and extracts aligned local geometric tokens at query locations from a frozen geometry encoder, yielding queryable structural representations without additional structural pretraining. We further introduce an inference-time refinement stage, SARe-Refine. By verifying candidate contact edges with geometric-consistency checks, it selects reliable substructures and resamples the remaining uncertain regions while keeping verified parts fixed, leading to more stable and consistent assemblies in the many-fragment regime. We evaluate SARe across three settings, including synthetic fractures, simulated fractures from scanned real objects, and real physically fractured scans. The results demonstrate state-of-the-art performance, with more graceful degradation and higher success rates as the fragment count increases in challenging large-scale reassembly.

  </details>



- **HACMatch Semi-Supervised Rotation Regression with Hardness-Aware Curriculum Pseudo Labeling**  
  Mei Li, Huayi Zhou, Suizhi Huang, Yuxiang Lu, Yue Ding, Hongtao Lu  
  _2026-03-23_ · https://arxiv.org/abs/2603.21583v1  
  <details><summary>Abstract</summary>

  Regressing 3D rotations of objects from 2D images is a crucial yet challenging task, with broad applications in autonomous driving, virtual reality, and robotic control. Existing rotation regression models often rely on large amounts of labeled data for training or require additional information beyond 2D images, such as point clouds or CAD models. Therefore, exploring semi-supervised rotation regression using only a limited number of labeled 2D images is highly valuable. While recent work FisherMatch introduces semi-supervised learning to rotation regression, it suffers from rigid entropy-based pseudo-label filtering that fails to effectively distinguish between reliable and unreliable unlabeled samples. To address this limitation, we propose a hardness-aware curriculum learning framework that dynamically selects pseudo-labeled samples based on their difficulty, progressing from easy to complex examples. We introduce both multi-stage and adaptive curriculum strategies to replace fixed-threshold filtering with more flexible, hardness-aware mechanisms. Additionally, we present a novel structured data augmentation strategy specifically tailored for rotation estimation, which assembles composite images from augmented patches to introduce feature diversity while preserving critical geometric integrity. Comprehensive experiments on PASCAL3D+ and ObjectNet3D demonstrate that our method outperforms existing supervised and semi-supervised baselines, particularly in low-data regimes, validating the effectiveness of our curriculum learning framework and structured augmentation approach.

  </details>



- **Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection**  
  Kaiqiang Li, Gang Li, Mingle Zhou, Min Li, Delong Han, Jin Wan  
  _2026-03-23_ · https://arxiv.org/abs/2603.21511v1  
  <details><summary>Abstract</summary>

  Zero-shot (ZS) 3D anomaly detection is crucial for reliable industrial inspection, as it enables detecting and localizing defects without requiring any target-category training data. Existing approaches render 3D point clouds into 2D images and leverage pre-trained Vision-Language Models (VLMs) for anomaly detection. However, such strategies inevitably discard geometric details and exhibit limited sensitivity to local anomalies. In this paper, we revisit intrinsic 3D representations and explore the potential of pre-trained Point-Language Models (PLMs) for ZS 3D anomaly detection. We propose BTP (Back To Point), a novel framework that effectively aligns 3D point cloud and textual embeddings. Specifically, BTP aligns multi-granularity patch features with textual representations for localized anomaly detection, while incorporating geometric descriptors to enhance sensitivity to structural anomalies. Furthermore, we introduce a joint representation learning strategy that leverages auxiliary point cloud data to improve robustness and enrich anomaly semantics. Extensive experiments on Real3D-AD and Anomaly-ShapeNet demonstrate that BTP achieves superior performance in ZS 3D anomaly detection. Code will be available at \href{https://github.com/wistful-8029/BTP-3DAD}{https://github.com/wistful-8029/BTP-3DAD}.

  </details>



- **PAS3R: Pose-Adaptive Streaming 3D Reconstruction for Long Video Sequences**  
  Lanbo Xu, Liang Guo, Caigui Jiang, Cheng Wang  
  _2026-03-22_ · https://arxiv.org/abs/2603.21436v1  
  <details><summary>Abstract</summary>

  Online monocular 3D reconstruction enables dense scene recovery from streaming video but remains fundamentally limited by the stability-adaptation dilemma: the reconstruction model must rapidly incorporate novel viewpoints while preserving previously accumulated scene structure. Existing streaming approaches rely on uniform or attention-based update mechanisms that often fail to account for abrupt viewpoint transitions, leading to trajectory drift and geometric inconsistencies over long sequences. We introduce PAS3R, a pose-adaptive streaming reconstruction framework that dynamically modulates state updates according to camera motion and scene structure. Our key insight is that frames contributing significant geometric novelty should exert stronger influence on the reconstruction state, while frames with minor viewpoint variation should prioritize preserving historical context. PAS3R operationalizes this principle through a motion-aware update mechanism that jointly leverages inter-frame pose variation and image frequency cues to estimate frame importance. To further stabilize long-horizon reconstruction, we introduce trajectory-consistent training objectives that incorporate relative pose constraints and acceleration regularization. A lightweight online stabilization module further suppresses high-frequency trajectory jitter and geometric artifacts without increasing memory consumption. Extensive experiments across multiple benchmarks demonstrate that PAS3R significantly improves trajectory accuracy, depth estimation, and point cloud reconstruction quality in long video sequences while maintaining competitive performance on shorter sequences.

  </details>



- **FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction**  
  Yuqiu Liu, Jialin Song, Marissa Ramirez de Chanlatte, Rochishnu Chowdhury, Rushil Paresh Desai, Wuyang Chen, Daniel Martin, Michael Mahoney  
  _2026-03-22_ · https://arxiv.org/abs/2603.21356v1  
  <details><summary>Abstract</summary>

  Real objects that inhabit the physical world follow physical laws and thus behave plausibly during interaction with other physical objects. However, current methods that perform 3D reconstructions of real-world scenes from multi-view 2D images optimize primarily for visual fidelity, i.e., they train with photometric losses and reason about uncertainty in the image or representation space. This appearance-centric view overlooks body contacts and couplings, conflates function-critical regions (e.g., aerodynamic or hydrodynamic surfaces) with ornamentation, and reconstructs structures suboptimally, even when physical regularizers are added. All these can lead to unphysical and implausible interactions. To address this, we consider the question: How can 3D reconstruction become aware of real-world interactions and underlying object functionality, beyond visual cues? To answer this question, we propose FluidGaussian, a plug-and-play method that tightly couples geometry reconstruction with ubiquitous fluid-structure interactions to assess surface quality at high granularity. We define a simulation-based uncertainty metric induced by fluid simulations and integrate it with active learning to prioritize views that improve both visual and physical fidelity. In an empirical evaluation on NeRF Synthetic (Blender), Mip-NeRF 360, and DrivAerNet++, our FluidGaussian method yields up to +8.6% visual PSNR (Peak Signal-to-Noise Ratio) and -62.3% velocity divergence during fluid simulations. Our code is available at https://github.com/delta-lab-ai/FluidGaussian.

  </details>



- **Training-Free Instance-Aware 3D Scene Reconstruction and Diffusion-Based View Synthesis from Sparse Images**  
  Jiatong Xia, Lingqiao Liu  
  _2026-03-22_ · https://arxiv.org/abs/2603.21166v1  
  <details><summary>Abstract</summary>

  We introduce a novel, training-free system for reconstructing, understanding, and rendering 3D indoor scenes from a sparse set of unposed RGB images. Unlike traditional radiance field approaches that require dense views and per-scene optimization, our pipeline achieves high-fidelity results without any training or pose preprocessing. The system integrates three key innovations: (1) A robust point cloud reconstruction module that filters unreliable geometry using a warping-based anomaly removal strategy; (2) A warping-guided 2D-to-3D instance lifting mechanism that propagates 2D segmentation masks into a consistent, instance-aware 3D representation; and (3) A novel rendering approach that projects the point cloud into new views and refines the renderings with a 3D-aware diffusion model. Our method leverages the generative power of diffusion to compensate for missing geometry and enhances realism, especially under sparse input conditions. We further demonstrate that object-level scene editing such as instance removal can be naturally supported in our pipeline by modifying only the point cloud, enabling the synthesis of consistent, edited views without retraining. Our results establish a new direction for efficient, editable 3D content generation without relying on scene-specific optimization. Project page: https://jiatongxia.github.io/TID3R/

  </details>



- **Single-Eye View: Monocular Real-time Perception Package for Autonomous Driving**  
  Haixi Zhang, Aiyinsi Zuo, Zirui Li, Chunshu Wu, Tong Geng, Zhiyao Duan  
  _2026-03-22_ · https://arxiv.org/abs/2603.21061v1  
  <details><summary>Abstract</summary>

  Amidst the rapid advancement of camera-based autonomous driving technology, effectiveness is often prioritized with limited attention to computational efficiency. To address this issue, this paper introduces LRHPerception, a real-time monocular perception package for autonomous driving that uses single-view camera video to interpret the surrounding environment. The proposed system combines the computational efficiency of end-to-end learning with the rich representational detail of local mapping methodologies. With significant improvements in object tracking and prediction, road segmentation, and depth estimation integrated into a unified framework, LRHPerception processes monocular image data into a five-channel tensor consisting of RGB, road segmentation, and pixel-level depth estimation, augmented with object detection and trajectory prediction. Experimental results demonstrate strong performance, achieving real-time processing at 29 FPS on a single GPU, representing a 555% speedup over the fastest mapping-based approach.

  </details>



- **SpatialFly: Geometry-Guided Representation Alignment for UAV Vision-and-Language Navigation in Urban Environments**  
  Wen Jiang, Kangyao Huang, Li Wang, Wang Xu, Wei Fan, Jinyuan Liu, Shaoyu Liu, Hanfang Liang, Hongwei Duan, Bin Xu, et al.  
  _2026-03-22_ · https://arxiv.org/abs/2603.21046v1  
  <details><summary>Abstract</summary>

  UAVs play an important role in applications such as autonomous exploration, disaster response, and infrastructure inspection. However, UAV VLN in complex 3D environments remains challenging. A key difficulty is the structural representation mismatch between 2D visual perception and the 3D trajectory decision space, which limits spatial reasoning. To this end, we propose SpatialFly, a geometry-guided spatial representation framework for UAV VLN. Operating on RGB observations without explicit 3D reconstruction, SpatialFly introduces a geometry-guided 2D representation alignment mechanism. Specifically, the geometric prior injection module injects global structural cues into 2D semantic tokens to provide scene-level geometric guidance. The geometry-aware reparameterization module then aligns 2D semantic tokens with 3D geometric tokens through cross-modal attention, followed by gated residual fusion to preserve semantic discrimination. Experimental results show that SpatialFly consistently outperforms state-of-the-art UAV VLN baselines across both seen and unseen environments, reducing NE by 4.03m and improving SR by 1.27% over the strongest baseline on the unseen Full split. Additional trajectory-level analysis shows that SpatialFly produces trajectories with better path alignment and smoother, more stable motion.

  </details>



- **Fast and Robust Deformable 3D Gaussian Splatting**  
  Han Jiao, Jiakai Sun, Lei Zhao, Zhanjie Zhang, Wei Xing, Huaizhong Lin  
  _2026-03-21_ · https://arxiv.org/abs/2603.20857v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has demonstrated remarkable real-time rendering capabilities and superior visual quality in novel view synthesis for static scenes. Building upon these advantages, researchers have progressively extended 3D Gaussians to dynamic scene reconstruction. Deformation field-based methods have emerged as a promising approach among various techniques. These methods maintain 3D Gaussian attributes in a canonical field and employ the deformation field to transform this field across temporal sequences. Nevertheless, these approaches frequently encounter challenges such as suboptimal rendering speeds, significant dependence on initial point clouds, and vulnerability to local optima in dim scenes. To overcome these limitations, we present FRoG, an efficient and robust framework for high-quality dynamic scene reconstruction. FRoG integrates per-Gaussian embedding with a coarse-to-fine temporal embedding strategy, accelerating rendering through the early fusion of temporal embeddings. Moreover, to enhance robustness against sparse initializations, we introduce a novel depth- and error-guided sampling strategy. This strategy populates the canonical field with new 3D Gaussians at low-deviation initial positions, significantly reducing the optimization burden on the deformation field and improving detail reconstruction in both static and dynamic regions. Furthermore, by modulating opacity variations, we mitigate the local optima problem in dim scenes, improving color fidelity. Comprehensive experimental results validate that our method achieves accelerated rendering speeds while maintaining state-of-the-art visual quality.

  </details>



- **Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding**  
  Jincen Jiang, Qianyu Zhou, Yuhang Li, Kui Su, Meili Wang, Jian Chang, Jian Jun Zhang, Xuequan Lu  
  _2026-03-21_ · https://arxiv.org/abs/2603.20739v1  
  <details><summary>Abstract</summary>

  While recent Transformer and Mamba architectures have advanced point cloud representation learning, they are typically developed for single-task or single-domain settings. Directly applying them to multi-task domain generalization (DG) leads to degraded performance. Transformers effectively model global dependencies but suffer from quadratic attention cost and lack explicit structural ordering, whereas Mamba offers linear-time recurrence yet often depends on coordinate-driven serialization, which is sensitive to viewpoint changes and missing regions, causing structural drift and unstable sequential modeling. In this paper, we propose Structure-Aware Domain Generalization (SADG), a Mamba-based In-Context Learning framework that preserves structural hierarchy across domains and tasks. We design structure-aware serialization (SAS) that generates transformation-invariant sequences using centroid-based topology and geodesic curvature continuity. We further devise hierarchical domain-aware modeling (HDM) that stabilizes cross-domain reasoning by consolidating intra-domain structure and fusing inter-domain relations. At test time, we introduce a lightweight spectral graph alignment (SGA) that shifts target features toward source prototypes in the spectral domain without updating model parameters, ensuring structure-preserving test-time feature shifting. In addition, we introduce MP3DObject, a real-scan object dataset for multi-task DG evaluation. Comprehensive experiments demonstrate that the proposed approach improves structural fidelity and consistently outperforms state-of-the-art methods across multiple tasks including reconstruction, denoising, and registration.

  </details>



- **The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting**  
  Ivan Desiatov, Torsten Sattler  
  _2026-03-21_ · https://arxiv.org/abs/2603.20714v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has become the method of choice for photo-realistic 3D reconstruction of scenes, due to being able to efficiently and accurately recover the scene appearance and geometry from images. 3DGS represents the scene through a set of 3D Gaussians, parameterized by their position, spatial extent, and view-dependent color. Starting from an initial point cloud, 3DGS refines the Gaussians' parameters as to reconstruct a set of training images as accurately as possible. Typically, a sparse Structure-from-Motion point cloud is used as initialization. In order to obtain dense Gaussian clouds, 3DGS methods thus rely on a densification stage. In this paper, we systematically study the relation between densification and initialization. Proposing a new benchmark, we study combinations of different types of initializations (dense laser scans, dense (multi-view) stereo point clouds, dense monocular depth estimates, sparse SfM point clouds) and different densification schemes. We show that current densification approaches are not able to take full advantage of dense initialization as they are often unable to (significantly) improve over sparse SfM-based initialization. We will make our benchmark publicly available.

  </details>


