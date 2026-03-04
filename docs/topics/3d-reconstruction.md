# 3D Reconstruction

_Updated: 2026-03-04 07:04 UTC_

Total papers shown: **23**


---

- **Utonia: Toward One Encoder for All Point Clouds**  
  Yujia Zhang, Xiaoyang Wu, Yunhan Yang, Xianzhe Fan, Han Li, Yuechen Zhang, Zehao Huang, Naiyan Wang, Hengshuang Zhao  
  _2026-03-03_ · https://arxiv.org/abs/2603.03283v1  
  <details><summary>Abstract</summary>

  We dream of a future where point clouds from all domains can come together to shape a single model that benefits them all. Toward this goal, we present Utonia, a first step toward training a single self-supervised point transformer encoder across diverse domains, spanning remote sensing, outdoor LiDAR, indoor RGB-D sequences, object-centric CAD models, and point clouds lifted from RGB-only videos. Despite their distinct sensing geometries, densities, and priors, Utonia learns a consistent representation space that transfers across domains. This unification improves perception capability while revealing intriguing emergent behaviors that arise only when domains are trained jointly. Beyond perception, we observe that Utonia representations can also benefit embodied and multimodal reasoning: conditioning vision-language-action policies on Utonia features improves robotic manipulation, and integrating them into vision-language models yields gains on spatial reasoning. We hope Utonia can serve as a step toward foundation models for sparse 3D data, and support downstream applications in AR/VR, robotics, and autonomous driving.

  </details>



- **LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory**  
  Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, Deqing Sun  
  _2026-03-03_ · https://arxiv.org/abs/2603.03269v1  
  <details><summary>Abstract</summary>

  Feedforward geometric foundation models achieve strong short-window reconstruction, yet scaling them to minutes-long videos is bottlenecked by quadratic attention complexity or limited effective memory in recurrent designs. We present LoGeR (Long-context Geometric Reconstruction), a novel architecture that scales dense 3D reconstruction to extremely long sequences without post-optimization. LoGeR processes video streams in chunks, leveraging strong bidirectional priors for high-fidelity intra-chunk reasoning. To manage the critical challenge of coherence across chunk boundaries, we propose a learning-based hybrid memory module. This dual-component system combines a parametric Test-Time Training (TTT) memory to anchor the global coordinate frame and prevent scale drift, alongside a non-parametric Sliding Window Attention (SWA) mechanism to preserve uncompressed context for high-precision adjacent alignment. Remarkably, this memory architecture enables LoGeR to be trained on sequences of 128 frames, and generalize up to thousands of frames during inference. Evaluated across standard benchmarks and a newly repurposed VBR dataset with sequences of up to 19k frames, LoGeR substantially outperforms prior state-of-the-art feedforward methods--reducing ATE on KITTI by over 74%--and achieves robust, globally consistent reconstruction over unprecedented horizons.

  </details>



- **VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats**  
  Alessio Mazzucchelli, Ivan Ojeda-Martin, Fernando Rivas-Manzaneque, Elena Garces, Adrian Penate-Sanchez, Francesc Moreno-Noguer  
  _2026-03-03_ · https://arxiv.org/abs/2603.02986v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.

  </details>



- **The Dresden Dataset for 4D Reconstruction of Non-Rigid Abdominal Surgical Scenes**  
  Reuben Docea, Rayan Younis, Yonghao Long, Maxime Fleury, Jinjing Xu, Chenyang Li, André Schulze, Ann Wierick, Johannes Bender, Micha Pfeiffer, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02985v1  
  <details><summary>Abstract</summary>

  The D4D Dataset provides paired endoscopic video and high-quality structured-light geometry for evaluating 3D reconstruction of deforming abdominal soft tissue in realistic surgical conditions. Data were acquired from six porcine cadaver sessions using a da Vinci Xi stereo endoscope and a Zivid structured-light camera, registered via optical tracking and manually curated iterative alignment methods. Three sequence types - whole deformations, incremental deformations, and moved-camera clips - probe algorithm robustness to non-rigid motion, deformation magnitude, and out-of-view updates. Each clip provides rectified stereo images, per-frame instrument masks, stereo depth, start/end structured-light point clouds, curated camera poses and camera intrinsics. In postprocessing, ICP and semi-automatic registration techniques are used to register data, and instrument masks are created. The dataset enables quantitative geometric evaluation in both visible and occluded regions, alongside photometric view-synthesis baselines. Comprising over 300,000 frames and 369 point clouds across 98 curated recordings, this resource can serve as a comprehensive benchmark for developing and evaluating non-rigid SLAM, 4D reconstruction, and depth estimation methods.

  </details>



- **R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild**  
  Margherita Lea Corona, Wieland Morgenstern, Peter Eisert, Anna Hilsmann  
  _2026-03-03_ · https://arxiv.org/abs/2603.02801v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS representation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW models view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction artifacts, improving rendering quality at the sky-foreground boundary

  </details>



- **DREAM: Where Visual Understanding Meets Text-to-Image Generation**  
  Chao Li, Tianhong Li, Sai Vidyaranya Nuthalapati, Hong-You Chen, Satya Narayan Shukla, Yonghuan Yang, Jun Xiao, Xiangjun Fan, Aashu Singh, Dina Katabi, et al.  
  _2026-03-03_ · https://arxiv.org/abs/2603.02667v1  
  <details><summary>Abstract</summary>

  Unifying visual representation learning and text-to-image (T2I) generation within a single model remains a central challenge in multimodal learning. We introduce DREAM, a unified framework that jointly optimizes discriminative and generative objectives, while learning strong visual representations. DREAM is built on two key techniques: During training, Masking Warmup, a progressive masking schedule, begins with minimal masking to establish the contrastive alignment necessary for representation learning, then gradually transitions to full masking for stable generative training. At inference, DREAM employs Semantically Aligned Decoding to align partially masked image candidates with the target text and select the best one for further decoding, improving text-image fidelity (+6.3%) without external rerankers. Trained solely on CC12M, DREAM achieves 72.7% ImageNet linear-probing accuracy (+1.1% over CLIP) and an FID of 4.25 (+6.2% over FLUID), with consistent gains in few-shot classification, semantic segmentation, and depth estimation. These results demonstrate that discriminative and generative objectives can be synergistic, allowing unified multimodal models that excel at both visual understanding and generation.

  </details>



- **VLMFusionOcc3D: VLM Assisted Multi-Modal 3D Semantic Occupancy Prediction**  
  A. Enes Doruk, Hasan F. Ates  
  _2026-03-03_ · https://arxiv.org/abs/2603.02609v1  
  <details><summary>Abstract</summary>

  This paper introduces VLMFusionOcc3D, a robust multimodal framework for dense 3D semantic occupancy prediction in autonomous driving. Current voxel-based occupancy models often struggle with semantic ambiguity in sparse geometric grids and performance degradation under adverse weather conditions. To address these challenges, we leverage the rich linguistic priors of Vision-Language Models (VLMs) to anchor ambiguous voxel features to stable semantic concepts. Our framework initiates with a dual-branch feature extraction pipeline that projects multi-view images and LiDAR point clouds into a unified voxel space. We propose Instance-driven VLM Attention (InstVLM), which utilizes gated cross-attention and LoRA-adapted CLIP embeddings to inject high-level semantic and geographic priors directly into the 3D voxels. Furthermore, we introduce Weather-Aware Adaptive Fusion (WeathFusion), a dynamic gating mechanism that utilizes vehicle metadata and weather-conditioned prompts to re-weight sensor contributions based on real-time environmental reliability. To ensure structural consistency, a Depth-Aware Geometric Alignment (DAGA) loss is employed to align dense camera-derived geometry with sparse, spatially accurate LiDAR returns. Extensive experiments on the nuScenes and SemanticKITTI datasets demonstrate that our plug-and-play modules consistently enhance the performance of state-of-the-art voxel-based baselines. Notably, our approach achieves significant improvements in challenging weather scenarios, offering a scalable and robust solution for complex urban navigation.

  </details>



- **TruckDrive: Long-Range Autonomous Highway Driving Dataset**  
  Filippo Ghilotti, Edoardo Palladin, Samuel Brucker, Adam Sigal, Mario Bijelic, Felix Heide  
  _2026-03-02_ · https://arxiv.org/abs/2603.02413v1  
  <details><summary>Abstract</summary>

  Safe highway autonomy for heavy trucks remains an open and unsolved challenge: due to long braking distances, scene understanding of hundreds of meters is required for anticipatory planning and to allow safe braking margins. However, existing driving datasets primarily cover urban scenes, with perception effectively limited to short ranges of only up to 100 meters. To address this gap, we introduce TruckDrive, a highway-scale multimodal driving dataset, captured with a sensor suite purpose-built for long range sensing: seven long-range FMCW LiDARs measuring range and radial velocity, three high-resolution short-range LiDARs, eleven 8MP surround cameras with varying focal lengths and ten 4D FMCW radars. The dataset offers 475 thousands samples with 165 thousands densely annotated frames for driving perception benchmarking up to 1,000 meters for 2D detection and 400 meters for 3D detection, depth estimation, tracking, planning and end to end driving over 20 seconds sequences at highway speeds. We find that state-of-the-art autonomous driving models do not generalize to ranges beyond 150 meters, with drops between 31% and 99% in 3D perception tasks, exposing a systematic long-range gap that current architectures and training signals cannot close.

  </details>



- **MERG3R: A Divide-and-Conquer Approach to Large-Scale Neural Visual Geometry**  
  Leo Kaixuan Cheng, Abdus Shaikh, Ruofan Liang, Zhijie Wu, Yushi Guan, Nandita Vijaykumar  
  _2026-03-02_ · https://arxiv.org/abs/2603.02351v1  
  <details><summary>Abstract</summary>

  Recent advancements in neural visual geometry, including transformer-based models such as VGGT and Pi3, have achieved impressive accuracy on 3D reconstruction tasks. However, their reliance on full attention makes them fundamentally limited by GPU memory capacity, preventing them from scaling to large, unordered image collections. We introduce MERG3R, a training-free divide-and-conquer framework that enables geometric foundation models to operate far beyond their native memory limits. MERG3R first reorders and partitions unordered images into overlapping, geometrically diverse subsets that can be reconstructed independently. It then merges the resulting local reconstructions through an efficient global alignment and confidence-weighted bundle adjustment procedure, producing a globally consistent 3D model. Our framework is model-agnostic and can be paired with existing neural geometry models. Across large-scale datasets, including 7-Scenes, NRGBD, Tanks & Temples, and Cambridge Landmarks, MERG3R consistently improves reconstruction accuracy, memory efficiency, and scalability, enabling high-quality reconstruction when the dataset exceeds memory capacity limits.

  </details>



- **3D Field of Junctions: A Noise-Robust, Training-Free Structural Prior for Volumetric Inverse Problems**  
  Namhoon Kim, Narges Moeini, Justin Romberg, Sara Fridovich-Keil  
  _2026-03-02_ · https://arxiv.org/abs/2603.02149v1  
  <details><summary>Abstract</summary>

  Volume denoising is a foundational problem in computational imaging, as many 3D imaging inverse problems face high levels of measurement noise. Inspired by the strong 2D image denoising properties of Field of Junctions (ICCV 2021), we propose a novel, fully volumetric 3D Field of Junctions (3D FoJ) representation that optimizes a junction of 3D wedges that best explain each 3D patch of a full volume, while encouraging consistency between overlapping patches. In addition to direct volume denoising, we leverage our 3D FoJ representation as a structural prior that: (i) requires no training data, and thus precludes the risk of hallucination, (ii) preserves and enhances sharp edge and corner structures in 3D, even under low signal to noise ratio (SNR), and (iii) can be used as a drop-in denoising representation via projected or proximal gradient descent for any volumetric inverse problem with low SNR. We demonstrate successful volume reconstruction and denoising with 3D FoJ across three diverse 3D imaging tasks with low-SNR measurements: low-dose X-ray computed tomography (CT), cryogenic electron tomography (cryo-ET), and denoising point clouds such as those from lidar in adverse weather. Across these challenging low-SNR volumetric imaging problems, 3D FoJ outperforms a mixture of classical and neural methods.

  </details>



- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution**  
  Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02134v2  
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


