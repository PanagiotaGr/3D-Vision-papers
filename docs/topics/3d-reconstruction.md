# 3D Reconstruction

_Updated: 2026-03-03 07:08 UTC_

Total papers shown: **21**


---

- **3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems**  
  Namhoon Kim, Narges Moeini, Justin Romberg, Sara Fridovich-Keil  
  _2026-03-02_ · https://arxiv.org/abs/2603.02149v1  
  <details><summary>Abstract</summary>

  Volume denoising is a foundational problem in computational imaging, as many 3D imaging inverse problems face high levels of measurement noise. Inspired by the strong 2D image denoising properties of Field of Junctions (ICCV 2021), we propose a novel, fully volumetric 3D Field of Junctions (3D FoJ) representation that optimizes a junction of 3D wedges that best explain each 3D patch of a full volume, while encouraging consistency between overlapping patches. In addition to direct volume denoising, we leverage our 3D FoJ representation as a structural prior that: (i) requires no training data, and thus precludes the risk of hallucination, (ii) preserves and enhances sharp edge and corner structures in 3D, even under low signal to noise ratio (SNR), and (iii) can be used as a drop-in denoising representation via projected or proximal gradient descent for any volumetric inverse problem with low SNR. We demonstrate successful volume reconstruction and denoising with 3D FoJ across three diverse 3D imaging tasks with low-SNR measurements: low-dose X-ray computed tomography (CT), cryogenic electron tomography (cryo-ET), and denoising point clouds such as those from lidar in adverse weather. Across these challenging low-SNR volumetric imaging problems, 3D FoJ outperforms a mixture of classical and neural methods.

  </details>



- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution**  
  Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02134v1  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

  </details>



- **WorldStereo: Bridging Camera-Guided Video Generation and Scene Reconstruction via 3D Geometric Memories**  
  Yisu Zhang, Chenjie Cao, Tengfei Wang, Xuhui Zuo, Junta Wu, Jianke Zhu, Chunchao Guo  
  _2026-03-02_ · https://arxiv.org/abs/2603.02049v1  
  <details><summary>Abstract</summary>

  Recent advances in foundational Video Diffusion Models (VDMs) have yielded significant progress. Yet, despite the remarkable visual quality of generated videos, reconstructing consistent 3D scenes from these outputs remains challenging, due to limited camera controllability and inconsistent generated content when viewed from distinct camera trajectories. In this paper, we propose WorldStereo, a novel framework that bridges camera-guided video generation and 3D reconstruction via two dedicated geometric memory modules. Formally, the global-geometric memory enables precise camera control while injecting coarse structural priors through incrementally updated point clouds. Moreover, the spatial-stereo memory constrains the model's attention receptive fields with 3D correspondence to focus on fine-grained details from the memory bank. These components enable WorldStereo to generate multi-view-consistent videos under precise camera control, facilitating high-quality 3D reconstruction. Furthermore, the flexible control branch-based WorldStereo shows impressive efficiency, benefiting from the distribution matching distilled VDM backbone without joint training. Extensive experiments across both camera-guided video generation and 3D reconstruction benchmarks demonstrate the effectiveness of our approach. Notably, we show that WorldStereo acts as a powerful world model, tackling diverse scene generation tasks (whether starting from perspective or panoramic images) with high-fidelity 3D results. Models will be released.

  </details>



- **Learning Vision-Based Omnidirectional Navigation: A Teacher-Student Approach Using Monocular Depth Estimation**  
  Jan Finke, Wayne Paul Martis, Adrian Schmelter, Lars Erbach, Christian Jestel, Marvin Wiedemann  
  _2026-03-02_ · https://arxiv.org/abs/2603.01999v1  
  <details><summary>Abstract</summary>

  Reliable obstacle avoidance in industrial settings demands 3D scene understanding, but widely used 2D LiDAR sensors perceive only a single horizontal slice of the environment, missing critical obstacles above or below the scan plane. We present a teacher-student framework for vision-based mobile robot navigation that eliminates the need for LiDAR sensors. A teacher policy trained via Proximal Policy Optimization (PPO) in NVIDIA Isaac Lab leverages privileged 2D LiDAR observations that account for the full robot footprint to learn robust navigation. The learned behavior is distilled into a student policy that relies solely on monocular depth maps predicted by a fine-tuned Depth Anything V2 model from four RGB cameras. The complete inference pipeline, comprising monocular depth estimation (MDE), policy execution, and motor control, runs entirely onboard an NVIDIA Jetson Orin AGX mounted on a DJI RoboMaster platform, requiring no external computation for inference. In simulation, the student achieves success rates of 82-96.5%, consistently outperforming the standard 2D LiDAR teacher (50-89%). In real-world experiments, the MDE-based student outperforms the 2D LiDAR teacher when navigating around obstacles with complex 3D geometries, such as overhanging structures and low-profile objects, that fall outside the single scan plane of a 2D LiDAR.

  </details>



- **physfusion: A Transformer-based Dual-Stream Radar and Vision Fusion Framework for Open Water Surface Object Detection**  
  Yuting Wan, Liguo Sun, Jiuwu Hao, Zao Zhang, Pin LV  
  _2026-03-02_ · https://arxiv.org/abs/2603.01947v1  
  <details><summary>Abstract</summary>

  Detecting water-surface targets for Unmanned Surface Vehicles (USVs) is challenging due to wave clutter, specular reflections, and weak appearance cues in long-range observations. Although 4D millimeter-wave radar complements cameras under degraded illumination, maritime radar point clouds are sparse and intermittent, with reflectivity attributes exhibiting heavy-tailed variations under scattering and multipath, making conventional fusion designs struggle to exploit radar cues effectively. We propose PhysFusion, a physics-informed radar-image detection framework for water-surface perception. The framework integrates: (1) a Physics-Informed Radar Encoder (PIR Encoder) with an RCS Mapper and Quality Gate, transforming per-point radar attributes into compact scattering priors and predicting point-wise reliability for robust feature learning under clutter; (2) a Radar-guided Interactive Fusion Module (RIFM) performing query-level radar-image fusion between semantically enriched radar features and multi-scale visual features, with the radar branch modeled by a dual-stream backbone including a point-based local stream and a transformer-based global stream using Scattering-Aware Self-Attention (SASA); and (3) a Temporal Query Aggregation module (TQA) aggregating frame-wise fused queries over a short temporal window for temporally consistent representations. Experiments on WaterScenes and FLOW demonstrate that PhysFusion achieves 59.7% mAP50:95 and 90.3% mAP50 on WaterScenes (T=5 radar history) using 5.6M parameters and 12.5G FLOPs, and reaches 94.8% mAP50 and 46.2% mAP50:95 on FLOW under radar+camera setting. Ablation studies quantify the contributions of PIR Encoder, SASA-based global reasoning, and RIFM.

  </details>



- **Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones**  
  Ilenia Carboni, Elia Cereda, Lorenzo Lamberti, Daniele Malpetti, Francesco Conti, Daniele Palossi  
  _2026-03-02_ · https://arxiv.org/abs/2603.01850v1  
  <details><summary>Abstract</summary>

  Sub-30g nano-sized aerial robots can leverage their agility and form factor to autonomously explore cluttered and narrow environments, like in industrial inspection and search and rescue missions. However, the price for their tiny size is a strong limit in their resources, i.e., sub-100 mW microcontroller units (MCUs) delivering $\sim$100 GOps/s at best, and memory budgets well below 100 MB. Despite these strict constraints, we aim to enable complex vision-based tasks aboard nano-drones, such as dense 3D scene reconstruction: a key robotic task underlying fundamental capabilities like spatial awareness and motion planning. Top-performing 3D reconstruction methods leverage neural radiance fields (NeRF) models, which require GBs of memory and massive computation, usually delivered by high-end GPUs consuming 100s of Watts. Our work introduces Tiny-DroNeRF, a lightweight NeRF model, based on Instant-NGP, and optimized for running on a GAP9 ultra-low-power (ULP) MCU aboard our nano-drones. Then, we further empower our Tiny-DroNeRF by leveraging a collaborative federated learning scheme, which distributes the model training among multiple nano-drones. Our experimental results show a 96% reduction in Tiny-DroNeRF's memory footprint compared to Instant-NGP, with only a 5.7 dB drop in reconstruction accuracy. Finally, our federated learning scheme allows Tiny-DroNeRF to train with an amount of data otherwise impossible to keep in a single drone's memory, increasing the overall reconstruction accuracy. Ultimately, our work combines, for the first time, NeRF training on an ULP MCU with federated learning on nano-drones.

  </details>



- **LEAR: Learning Edge-Aware Representations for Event-to-LiDAR Localization**  
  Kuangyi Chen, Jun Zhang, Yuxi Hu, Yi Zhou, Friedrich Fraundorfer  
  _2026-03-02_ · https://arxiv.org/abs/2603.01839v1  
  <details><summary>Abstract</summary>

  Event cameras offer high-temporal-resolution sensing that remains reliable under high-speed motion and challenging lighting, making them promising for localization from LiDAR point clouds in GPS-denied and visually degraded environments. However, aligning sparse, asynchronous events with dense LiDAR maps is fundamentally ill-posed, as direct correspondence estimation suffers from modality gaps. We propose LEAR, a dual-task learning framework that jointly estimates edge structures and dense event-depth flow fields to bridge the sensing-modality divide. Instead of treating edges as a post-hoc aid, LEAR couples them with flow estimation through a cross-modal fusion mechanism that injects modality-invariant geometric cues into the motion representation, and an iterative refinement strategy that enforces mutual consistency between the two tasks over multiple update steps. This synergy produces edge-aware, depth-aligned flow fields that enable more robust and accurate pose recovery via Perspective-n-Point (PnP) solvers. On several popular and challenging datasets, LEAR achieves superior performance over the best prior method. The source code, trained models, and demo videos are made publicly available online.

  </details>



- **Affine Correspondences in Stereo Vision: Theory, Practice, and Limitations**  
  Levente Hajder  
  _2026-03-02_ · https://arxiv.org/abs/2603.01836v1  
  <details><summary>Abstract</summary>

  Affine transformations have been recently used for stereo vision. They can be exploited in various computer vision application, e.g., when estimating surface normals, homographies, fundamental and essential matrices. Even full 3D reconstruction can be obtained by using affine correspondences. First, this paper overviews the fundamental statements for affine transformations and epipolar geometry. Then it is investigated how the transformation accuracy influences the quality of the 3D reconstruction. Besides, we propose novel techniques for estimating the local affine transformation from corresponding image directions; moreover, the fundamental matrix, related to the processed image pair, can also be exploited. Both synthetic and real quantitative evaluations are implemented based on the accuracy of the reconstructed surface normals. For the latter one, a special object, containing three perpendicular planes with chessboard patterns, is constructed. The quantitative evaluations are based on the accuracy of the reconstructed surface normals and it is concluded that the estimation accuracy is around a few degrees for realistic test cases. Special stereo poses and plane orientations are also evaluated in detail.

  </details>



- **Neural Operator-Grounded Continuous Tensor Function Representation and Its Applications**  
  Ruoyang Su, Xi-Le Zhao, Sheng Liu, Wei-Hao Wu, Yisi Luo, Michael K. Ng  
  _2026-03-02_ · https://arxiv.org/abs/2603.01812v1  
  <details><summary>Abstract</summary>

  Recently, continuous tensor functions have attracted increasing attention, because they can unifiedly represent data both on mesh grids and beyond mesh grids. However, since mode-$n$ product is essentially discrete and linear, the potential of current continuous tensor function representations is still locked. To break this bottleneck, we suggest neural operator-grounded mode-$n$ operators as a continuous and nonlinear alternative of discrete and linear mode-$n$ product. Instead of mapping the discrete core tensor to the discrete target tensor, proposed mode-$n$ operator directly maps the continuous core tensor function to the continuous target tensor function, which provides a genuine continuous representation of real-world data and can ameliorate discretization artifacts. Empowering with continuous and nonlinear mode-$n$ operators, we propose a neural operator-grounded continuous tensor function representation (abbreviated as NO-CTR), which can more faithfully represent complex real-world data compared with classic discrete tensor representations and continuous tensor function representations. Theoretically, we also prove that any continuous tensor function can be approximated by NO-CTR. To examine the capability of NO-CTR, we suggest an NO-CTR-based multi-dimensional data completion model. Extensive experiments across various data on regular mesh grids (multi-spectral images and color videos), on mesh girds with different resolutions (Sentinel-2 images) and beyond mesh grids (point clouds) demonstrate the superiority of NO-CTR.

  </details>



- **Dehallu3D: Hallucination-Mitigated 3D Generation from Single Image via Cyclic View Consistency Refinement**  
  Xiwen Wang, Shichao Zhang, Hailun Zhang, Ruowei Wang, Mao Li, Chenyu Zhou, Qijun Zhao, Ji-Zhe Zhou  
  _2026-03-02_ · https://arxiv.org/abs/2603.01601v1  
  <details><summary>Abstract</summary>

  Large 3D reconstruction models have revolutionized the 3D content generation field, enabling broad applications in virtual reality and gaming. Just like other large models, large 3D reconstruction models suffer from hallucinations as well, introducing structural outliers (e.g., odd holes or protrusions) that deviate from the input data. However, unlike other large models, hallucinations in large 3D reconstruction models remain severely underexplored, leading to malformed 3D-printed objects or insufficient immersion in virtual scenes. Such hallucinations majorly originate from that existing methods reconstruct 3D content from sparsely generated multi-view images which suffer from large viewpoint gaps and discontinuities. To mitigate hallucinations by eliminating the outliers, we propose Dehallu3D for 3D mesh generation. Our key idea is to design a balanced multi-view continuity constraint to enforce smooth transitions across dense intermediate viewpoints, while avoiding over-smoothing that could erase sharp geometric features. Therefore, Dehallu3D employs a plug-and-play optimization module with two key constraints: (i) adjacent consistency to ensure geometric continuity across views, and (ii) adaptive smoothness to retain fine details.We further propose the Outlier Risk Measure (ORM) metric to quantify geometric fidelity in 3D generation from the perspective of outliers. Extensive experiments show that Dehallu3D achieves high-fidelity 3D generation by effectively preserving structural details while removing hallucinated outliers.

  </details>



- **WildCross: A Cross-Modal Large Scale Benchmark for Place Recognition and Metric Depth Estimation in Natural Environments**  
  Joshua Knights, Joseph Reid, Kaushik Roy, David Hall, Mark Cox, Peyman Moghadam  
  _2026-03-02_ · https://arxiv.org/abs/2603.01475v1  
  <details><summary>Abstract</summary>

  Recent years have seen a significant increase in demand for robotic solutions in unstructured natural environments, alongside growing interest in bridging 2D and 3D scene understanding. However, existing robotics datasets are predominantly captured in structured urban environments, making them inadequate for addressing the challenges posed by complex, unstructured natural settings. To address this gap, we propose WildCross, a cross-modal benchmark for place recognition and metric depth estimation in large-scale natural environments. WildCross comprises over 476K sequential RGB frames with semi-dense depth and surface normal annotations, each aligned with accurate 6DoF poses and synchronized dense lidar submaps. We conduct comprehensive experiments on visual, lidar, and cross-modal place recognition, as well as metric depth estimation, demonstrating the value of WildCross as a challenging benchmark for multi-modal robotic perception tasks. We provide access to the code repository and dataset at https://csiro-robotics.github.io/WildCross.

  </details>



- **RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations**  
  Mochu Xiang, Zhelun Shen, Xuesong Li, Jiahui Ren, Jing Zhang, Chen Zhao, Shanshan Liu, Haocheng Feng, Jingdong Wang, Yuchao Dai  
  _2026-03-01_ · https://arxiv.org/abs/2603.01194v1  
  <details><summary>Abstract</summary>

  Human perceive the 3D world through 2D observations from limited viewpoints. While recent feed-forward generalizable 3D reconstruction models excel at recovering 3D structures from sparse images, their representations are often confined to observed regions, leaving unseen geometry un-modeled. This raises a key, fundamental challenge: Can we infer a complete 3D structure from partial 2D observations? We present RnG (Reconstruction and Generation), a novel feed-forward Transformer that unifies these two tasks by predicting an implicit, complete 3D representation. At the core of RnG, we propose a reconstruction-guided causal attention mechanism that separates reconstruction and generation at the attention level, and treats the KV-cache as an implicit 3D representation. Then, arbitrary poses can efficiently query this cache to render high-fidelity, novel-view RGBD outputs. As a result, RnG not only accurately reconstructs visible geometry but also generates plausible, coherent unseen geometry and appearance. Our method achieves state-of-the-art performance in both generalizable 3D reconstruction and novel view generation, while operating efficiently enough for real-time interactive applications. Project page: https://npucvr.github.io/RnG

  </details>



- **ArtLLM: Generating Articulated Assets via 3D LLM**  
  Penghao Wang, Siyuan Xie, Hongyu Yan, Xianghui Yang, Jingwei Huang, Chunchao Guo, Jiayuan Gu  
  _2026-03-01_ · https://arxiv.org/abs/2603.01142v1  
  <details><summary>Abstract</summary>

  Creating interactive digital environments for gaming, robotics, and simulation relies on articulated 3D objects whose functionality emerges from their part geometry and kinematic structure. However, existing approaches remain fundamentally limited: optimization-based reconstruction methods require slow, per-object joint fitting and typically handle only simple, single-joint objects, while retrieval-based methods assemble parts from a fixed library, leading to repetitive geometry and poor generalization. To address these challenges, we introduce ArtLLM, a novel framework for generating high-quality articulated assets directly from complete 3D meshes. At its core is a 3D multimodal large language model trained on a large-scale articulation dataset curated from both existing articulation datasets and procedurally generated objects. Unlike prior work, ArtLLM autoregressively predicts a variable number of parts and joints, inferring their kinematic structure in a unified manner from the object's point cloud. This articulation-aware layout then conditions a 3D generative model to synthesize high-fidelity part geometries. Experiments on the PartNet-Mobility dataset show that ArtLLM significantly outperforms state-of-the-art methods in both part layout accuracy and joint prediction, while generalizing robustly to real-world objects. Finally, we demonstrate its utility in constructing digital twins, highlighting its potential for scalable robot learning.

  </details>



- **Adaptive Augmentation-Aware Latent Learning for Robust LiDAR Semantic Segmentation**  
  Wangkai Li, Zhaoyang Li, Yuwen Pan, Rui Sun, Yujia Chen, Tianzhu Zhang  
  _2026-03-01_ · https://arxiv.org/abs/2603.01074v1  
  <details><summary>Abstract</summary>

  Adverse weather conditions significantly degrade the performance of LiDAR point cloud semantic segmentation networks by introducing large distribution shifts. Existing augmentation-based methods attempt to enhance robustness by simulating weather interference during training. However, they struggle to fully exploit the potential of augmentations due to the trade-off between minor and aggressive augmentations. To address this, we propose A3Point, an adaptive augmentation-aware latent learning framework that effectively utilizes a diverse range of augmentations while mitigating the semantic shift, which refers to the change in the semantic meaning caused by augmentations. A3Point consists of two key components: semantic confusion prior (SCP) latent learning, which captures the model's inherent semantic confusion information, and semantic shift region (SSR) localization, which decouples semantic confusion and semantic shift, enabling adaptive optimization strategies for different disturbance levels. Extensive experiments on multiple standard generalized LiDAR segmentation benchmarks under adverse weather demonstrate the effectiveness of our method, setting new state-of-the-art results.

  </details>



- **Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery**  
  Yangyang Xu, Junbo Ke, You-Wei Wen, Chao Wang  
  _2026-03-01_ · https://arxiv.org/abs/2603.01034v1  
  <details><summary>Abstract</summary>

  Tensor Ring (TR) decomposition is a powerful tool for high-order data modeling, but is inherently restricted to discrete forms defined on fixed meshgrids. In this work, we propose a TR functional decomposition for both meshgrid and non-meshgrid data, where factors are parameterized by Implicit Neural Representations (INRs). However, optimizing this continuous framework to capture fine-scale details is intrinsically difficult. Through a frequency-domain analysis, we demonstrate that the spectral structure of TR factors determines the frequency composition of the reconstructed tensor and limits the high-frequency modeling capacity. To mitigate this, we propose a reparameterized TR functional decomposition, in which each TR factor is a structured combination of a learnable latent tensor and a fixed basis. This reparameterization is theoretically shown to improve the training dynamics of TR factor learning. We further derive a principled initialization scheme for the fixed basis and prove the Lipschitz continuity of our proposed model. Extensive experiments on image inpainting, denoising, super-resolution, and point cloud recovery demonstrate that our method achieves consistently superior performance over existing approaches. Code is available at https://github.com/YangyangXu2002/RepTRFD.

  </details>



- **Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving**  
  Xubo Zhu, Haoyang Zhang, Fei He, Rui Wu, Yanhu Shan, Wen Yang, Huai Yu  
  _2026-03-01_ · https://arxiv.org/abs/2603.01007v1  
  <details><summary>Abstract</summary>

  3D semantic occupancy prediction is crucial for autonomous driving perception, offering comprehensive geometric scene understanding and semantic recognition. However, existing methods struggle with geometric misalignment in view transformation due to the lack of pixel-level accurate depth estimation, and severe spatial class imbalance where semantic categories exhibit strong spatial anisotropy. To address these challenges, we propose Dr.Occ, a depth- and region-guided occupancy prediction framework. Specifically, we introduce a depth-guided 2D-to-3D View Transformer (D$^2$-VFormer) that effectively leverages high-quality dense depth cues from MoGe-2 to construct reliable geometric priors, thereby enabling precise geometric alignment of voxel features. Moreover, inspired by the Mixture-of-Experts (MoE) framework, we propose a region-guided Expert Transformer (R/R$^2$-EFormer) that adaptively allocates region-specific experts to focus on different spatial regions, effectively addressing spatial semantic variations. Thus, the two components make complementary contributions: depth guidance ensures geometric alignment, while region experts enhance semantic learning. Experiments on the Occ3D-nuScenes benchmark demonstrate that \textbf{Dr.Occ} improves the strong baseline BEVDet4D by 7.43\% mIoU and 3.09\% IoU under the full vision-only setting.

  </details>



- **MLRecon: Robust Markerless Freehand 3D Ultrasound Reconstruction via Coarse-to-Fine Pose Estimation**  
  Yi Zhang, Puxun Tu, Kun Wang, Yulin Yan, Tao Ying, Xiaojun Chen  
  _2026-03-01_ · https://arxiv.org/abs/2603.00990v1  
  <details><summary>Abstract</summary>

  Freehand 3D ultrasound (US) reconstruction promises volumetric imaging with the flexibility of standard 2D probes, yet existing tracking paradigms face a restrictive trilemma: marker-based systems demand prohibitive costs, inside-out methods require intrusive sensor attachment, and sensorless approaches suffer from severe cumulative drift. To overcome these limitations, we present MLRecon, a robust markerless 3D US reconstruction framework delivering drift-resilient 6D probe pose tracking using a single commodity RGB-D camera. Leveraging the generalization power of vision foundation models, our pipeline enables continuous markerless tracking of the probe, augmented by a vision-guided divergence detector that autonomously monitors tracking integrity and triggers failure recovery to ensure uninterrupted scanning. Crucially, we further propose a dual-stage pose refinement network that explicitly disentangles high-frequency jitter from low-frequency bias, effectively denoising the trajectory while maintaining the kinematic fidelity of operator maneuvers. Experiments demonstrate that MLRecon significantly outperforms competing sensorless and sensor-aided methods, achieving average position errors as low as 0.88 mm on complex trajectories and yielding high-quality 3D reconstructions with sub-millimeter mean surface accuracy. This establishes a new benchmark for low-cost, accessible volumetric US imaging in resource-limited clinical settings.

  </details>



- **UD-SfPNet: An Underwater Descattering Shape-from-Polarization Network for 3D Normal Reconstruction**  
  Puyun Wang, Kaimin Yu, Huayang He, Feng Huang, Xianyu Wu, Yating Chen  
  _2026-03-01_ · https://arxiv.org/abs/2603.00908v1  
  <details><summary>Abstract</summary>

  Underwater optical imaging is severely hindered by scattering, but polarization imaging offers the unique dual advantages of descattering and shape-from-polarization (SfP) 3D reconstruction. To exploit these advantages, this paper proposes UD-SfPNet, an underwater descattering shape-from-polarization network that leverages polarization cues for improved 3D surface normal prediction. The framework jointly models polarization-based image descattering and SfP normal estimation in a unified pipeline, avoiding error accumulation from sequential processing and enabling global optimization across both tasks. UD-SfPNet further incorporates a novel color embedding module to enhance geometric consistency by exploiting the relationship between color encodings and surface orientation. A detail enhancement convolution module is also included to better preserve high-frequency geometric details that are lost under scattering. Experiments on the MuS-Polar3D dataset show that the proposed method significantly improves reconstruction accuracy, achieving a mean surface normal angular error of 15.12$^\circ$ (the lowest among compared methods). These results confirm the efficacy of combining descattering with polarization-based shape inference, and highlight the practical significance and potential applications of UD-SfPNet for optical 3D imaging in challenging underwater environments. The code is available at https://github.com/WangPuyun/UD-SfPNet.

  </details>



- **pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning**  
  Zhanpeng Luo, Ce Zhang, Silong Yong, Cunxi Dai, Qianwei Wang, Haoxi Ran, Guanya Shi, Katia Sycara, Yaqi Xie  
  _2026-03-01_ · https://arxiv.org/abs/2603.00905v1  
  <details><summary>Abstract</summary>

  Multi-modal Large Language Models (MLLMs) have demonstrated strong capabilities in general-purpose perception and reasoning, but they still struggle with tasks that require spatial understanding of the 3D world. To address this, we introduce pySpatial, a visual programming framework that equips MLLMs with the ability to interface with spatial tools via Python code generation. Given an image sequence and a natural-language query, the model composes function calls to spatial tools including 3D reconstruction, camera-pose recovery, novel-view rendering, etc. These operations convert raw 2D inputs into an explorable 3D scene, enabling MLLMs to reason explicitly over structured spatial representations. Notably, pySpatial requires no gradient-based fine-tuning and operates in a fully zero-shot setting. Experimental evaluations on the challenging MindCube and Omni3D-Bench benchmarks demonstrate that our framework pySpatial consistently surpasses strong MLLM baselines; for instance, it outperforms GPT-4.1-mini by 12.94% on MindCube. Furthermore, we conduct real-world indoor navigation experiments where the robot can successfully traverse complex environments using route plans generated by pySpatial, highlighting the practical effectiveness of our approach.

  </details>



- **PPC-MT: Parallel Point Cloud Completion with Mamba-Transformer Hybrid Architecture**  
  Jie Li, Shengwei Tian, Long Yu, Xin Ning  
  _2026-03-01_ · https://arxiv.org/abs/2603.00870v1  
  <details><summary>Abstract</summary>

  Existing point cloud completion methods struggle to balance high-quality reconstruction with computational efficiency. To address this, we propose PPC-MT, a novel parallel framework for point cloud completion leveraging a hybrid Mamba-Transformer architecture. Our approach introduces an innovative parallel completion strategy guided by Principal Component Analysis (PCA), which imposes a geometrically meaningful structure on unordered point clouds, transforming them into ordered sets and decomposing them into multiple subsets. These subsets are reconstructed in parallel using a multi-head reconstructor. This structured parallel synthesis paradigm significantly enhances the uniformity of point distribution and detail fidelity, while preserving computational efficiency. By integrating Mamba's linear complexity for efficient feature extraction during encoding with the Transformer's capability to model fine-grained multi-sequence relationships during decoding, PPC-MT effectively balances efficiency and reconstruction accuracy. Extensive quantitative and qualitative experiments on benchmark datasets, including PCN, ShapeNet-55/34, and KITTI, demonstrate that PPC-MT outperforms state-of-the-art methods across multiple metrics, validating the efficacy of our proposed framework.

  </details>



- **STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification**  
  Xingguo Xu, Zhanyu Liu, Weixiang Zhou, Yuansheng Gao, Junjie Cao, Yuhao Wang, Jixiang Luo, Dell Zhang  
  _2026-02-28_ · https://arxiv.org/abs/2603.00695v1  
  <details><summary>Abstract</summary>

  Multi-modal object Re-Identification (ReID) aims to exploit complementary information from different modalities to retrieve specific objects. However, existing methods often rely on hard token filtering or simple fusion strategies, which can lead to the loss of discriminative cues and increased background interference. To address these challenges, we propose STMI, a novel multi-modal learning framework consisting of three key components: (1) Segmentation-Guided Feature Modulation (SFM) module leverages SAM-generated masks to enhance foreground representations and suppress background noise through learnable attention modulation; (2) Semantic Token Reallocation (STR) module employs learnable query tokens and an adaptive reallocation mechanism to extract compact and informative representations without discarding any tokens; (3) Cross-Modal Hypergraph Interaction (CHI) module constructs a unified hypergraph across modalities to capture high-order semantic relationships. Extensive experiments on public benchmarks (i.e., RGBNT201, RGBNT100, and MSVR310) demonstrate the effectiveness and robustness of our proposed STMI framework in multi-modal ReID scenarios.

  </details>


