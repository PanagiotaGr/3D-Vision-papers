# Gaussian Splatting & 3DGS

_Updated: 2026-03-17 07:17 UTC_

Total papers shown: **11**


---

- **IRIS: Intersection-aware Ray-based Implicit Editable Scenes**  
  Grzegorz Wilczyński, Mikołaj Zieliński, Krzysztof Byrski, Joanna Waczyńska, Dominik Belter, Przemysław Spurek  
  _2026-03-16_ · https://arxiv.org/abs/2603.15368v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields achieve high-fidelity scene representation but suffer from costly training and rendering, while 3D Gaussian splatting offers real-time performance with strong empirical results. Recently, solutions that harness the best of both worlds by using Gaussians as proxies to guide neural field evaluations, still suffer from significant computational inefficiencies. They typically rely on stochastic volumetric sampling to aggregate features, which severely limits rendering performance. To address this issue, a novel framework named IRIS (Intersection-aware Ray-based Implicit Editable Scenes) is introduced as a method designed for efficient and interactive scene editing. To overcome the limitations of standard ray marching, an analytical sampling strategy is employed that precisely identifies interaction points between rays and scene primitives, effectively eliminating empty space processing. Furthermore, to address the computational bottleneck of spatial neighbor lookups, a continuous feature aggregation mechanism is introduced that operates directly along the ray. By interpolating latent attributes from sorted intersections, costly 3D searches are bypassed, ensuring geometric consistency, enabling high-fidelity, real-time rendering, and flexible shape editing. Code can be found at https://github.com/gwilczynski95/iris.

  </details>



- **GeoNVS: Geometry Grounded Video Diffusion for Novel View Synthesis**  
  Minjun Kang, Inkyu Shin, Taeyeop Lee, Myungchul Kim, In So Kweon, Kuk-Jin Yoon  
  _2026-03-16_ · https://arxiv.org/abs/2603.14965v1  
  <details><summary>Abstract</summary>

  Novel view synthesis requires strong 3D geometric consistency and the ability to generate visually coherent images across diverse viewpoints. While recent camera-controlled video diffusion models show promising results, they often suffer from geometric distortions and limited camera controllability. To overcome these challenges, we introduce GeoNVS, a geometry-grounded novel-view synthesizer that enhances both geometric fidelity and camera controllability through explicit 3D geometric guidance. Our key innovation is the Gaussian Splat Feature Adapter (GS-Adapter), which lifts input-view diffusion features into 3D Gaussian representations, renders geometry-constrained novel-view features, and adaptively fuses them with diffusion features to correct geometrically inconsistent representations. Unlike prior methods that inject geometry at the input level, GS-Adapter operates in feature space, avoiding view-dependent color noise that degrades structural consistency. Its plug-and-play design enables zero-shot compatibility with diverse feed-forward geometry models without additional training, and can be adapted to other video diffusion backbones. Experiments across 9 scenes and 18 settings demonstrate state-of-the-art performance, achieving 11.3% and 14.9% improvements over SEVA and CameraCtrl, with up to 2x reduction in translation error and 7x in Chamfer Distance.

  </details>



- **Zero-Shot Reconstruction of Animatable 3D Avatars with Cloth Dynamics from a Single Image**  
  Joohyun Kwon, Geonhee Sim, Gyeongsik Moon  
  _2026-03-16_ · https://arxiv.org/abs/2603.14772v1  
  <details><summary>Abstract</summary>

  Existing single-image 3D human avatar methods primarily rely on rigid joint transformations, limiting their ability to model realistic cloth dynamics. We present DynaAvatar, a zero-shot framework that reconstructs animatable 3D human avatars with motion-dependent cloth dynamics from a single image. Trained on large-scale multi-person motion datasets, DynaAvatar employs a Transformer-based feed-forward architecture that directly predicts dynamic 3D Gaussian deformations without subject-specific optimization. To overcome the scarcity of dynamic captures, we introduce a static-to-dynamic knowledge transfer strategy: a Transformer pretrained on large-scale static captures provides strong geometric and appearance priors, which are efficiently adapted to motion-dependent deformations through lightweight LoRA fine-tuning on dynamic captures. We further propose the DynaFlow loss, an optical flow-guided objective that provides reliable motion-direction geometric cues for cloth dynamics in rendered space. Finally, we reannotate the missing or noisy SMPL-X fittings in existing dynamic capture datasets, as most public dynamic capture datasets contain incomplete or unreliable fittings that are unsuitable for training high-quality 3D avatar reconstruction models. Experiments demonstrate that DynaAvatar produces visually rich and generalizable animations, outperforming prior methods.

  </details>



- **LiDAR-EVS: Enhance Extrapolated View Synthesis for 3D Gaussian Splatting with Pseudo-LiDAR Supervision**  
  Yiming Huang, Xin Kang, Sipeng Zhang, Hongliang Ren, Weihua Zhang, Junjie Lai  
  _2026-03-16_ · https://arxiv.org/abs/2603.14763v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time LiDAR and camera synthesis in autonomous driving simulation. However, simulating LiDAR with 3DGS remains challenging for extrapolated views beyond the training trajectory, as existing methods are typically trained on single-traversal sensor scans, suffer from severe overfitting and poor generalization to novel ego-vehicle paths. To enable reliable simulation of LiDAR along unseen driving trajectories without external multi-pass data, we present LiDAR-EVS, a lightweight framework for robust extrapolated-view LiDAR simulation in autonomous driving. Designed to be plug-and-play, LiDAR-EVS readily extends to diverse LiDAR sensors and neural rendering baselines with minimal modification. Our framework comprises two key components: (1) pseudo extrapolated-view point cloud supervision with multi-frame LiDAR fusion, view transformation, occlusion curling, and intensity adjustment; (2) spatially-constrained dropout regularization that promotes robustness to diverse trajectory variations encountered in real-world driving. Extensive experiments demonstrate that LiDAR-EVS achieves SOTA performance on extrapolated-view LiDAR synthesis across three datasets, making it a promising tool for data-driven simulation, closed-loop evaluation, and synthetic data generation in autonomous driving systems.

  </details>



- **E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction**  
  Yunsoo Kim, Changki Sung, Dasol Hong, Hyun Myung  
  _2026-03-16_ · https://arxiv.org/abs/2603.14684v1  
  <details><summary>Abstract</summary>

  The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS). These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting. Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution. However, existing event-based NVS methods either assume known poses or rely on depth estimation models that are bounded by their initial observations, failing to generalize as the camera traverses previously unseen regions. We present E2EGS, a pose-free framework operating solely on event streams. Our key insight is that edge information provides rich structural cues essential for accurate trajectory estimation and high-quality NVS. To extract edges from noisy event streams, we exploit the distinct spatio-temporal characteristics of edges and non-edge regions. The event camera's movement induces consistent events along edges, while non-edge regions produce sparse noise. We leverage this through a patch-based temporal coherence analysis that measures local variance to extract edges while robustly suppressing noise. The extracted edges guide structure-aware Gaussian initialization and enable edge-weighted losses throughout initialization, tracking, and bundle adjustment. Extensive experiments on both synthetic and real datasets demonstrate that E2EGS achieves superior reconstruction quality and trajectory accuracy, establishing a fully pose-free paradigm for event-based 3D reconstruction.

  </details>



- **Direct Object-Level Reconstruction via Probabilistic Gaussian Splatting**  
  Shuai Guo, Ao Guo, Junchao Zhao, Qi Chen, Yuxiang Qi, Zechuan Li, Dong Chen, Tianjia Shao, Mingliang Xu  
  _2026-03-15_ · https://arxiv.org/abs/2603.14316v1  
  <details><summary>Abstract</summary>

  Object-level 3D reconstruction play important roles across domains such as cultural heritage digitization, industrial manufacturing, and virtual reality. However, existing Gaussian Splatting-based approaches generally rely on full-scene reconstruction, in which substantial redundant background information is introduced, leading to increased computational and storage overhead. To address this limitation, we propose an efficient single-object 3D reconstruction method based on 2D Gaussian Splatting. By directly integrating foreground-background probability cues into Gaussian primitives and dynamically pruning low-probability Gaussians during training, the proposed method fundamentally focuses on an object of interest and improves the memory and computational efficiency. Our pipeline leverages probability masks generated by YOLO and SAM to supervise probabilistic Gaussian attributes, replacing binary masks with continuous probability values to mitigate boundary ambiguity. Additionally, we propose a dual-stage filtering strategy for training's startup to suppress background Gaussians. And, during training, rendered probability masks are conversely employed to refine supervision and enhance boundary consistency across views. Experiments conducted on the MIP-360, T&T, and NVOS datasets demonstrate that our method exhibits strong self-correction capability in the presence of mask errors and achieves reconstruction quality comparable to standard 3DGS approaches, while requiring only approximately 1/10 of their Gaussian amount. These results validate the efficiency and robustness of our method for single-object reconstruction and highlight its potential for applications requiring both high fidelity and computational efficiency.

  </details>



- **In-Field 3D Wheat Head Instance Segmentation From TLS Point Clouds Using Deep Learning Without Manual Labels**  
  Tomislav Medic, Liangliang Nan  
  _2026-03-15_ · https://arxiv.org/abs/2603.14309v1  
  <details><summary>Abstract</summary>

  3D instance segmentation for laser scanning (LiDAR) point clouds remains a challenge in many remote sensing-related domains. Successful solutions typically rely on supervised deep learning and manual annotations, and consequently focus on objects that can be well delineated through visual inspection and manual labeling of point clouds. However, for tasks with more complex and cluttered scenes, such as in-field plant phenotyping in agriculture, such approaches are often infeasible. In this study, we tackle the task of in-field wheat head instance segmentation directly from terrestrial laser scanning (TLS) point clouds. To address the problem and circumvent the need for manual annotations, we propose a novel two-stage pipeline. To obtain the initial 3D instance proposals, the first stage uses 3D-to-2D multi-view projections, the Grounded SAM pipeline for zero-shot 2D object-centric segmentation, and multi-view label fusion. The second stage uses these initial proposals as noisy pseudo-labels to train a supervised 3D panoptic-style segmentation neural network. Our results demonstrate the feasibility of the proposed approach and show performance improvementsrelative to Wheat3DGS, a recent alternative solution for in-field wheat head instance segmentation without manual 3D annotations based on multi-view RGB images and 3D Gaussian Splatting, showcasing TLS as a competitive sensing alternative. Moreover, the results show that both stages of the proposed pipeline can deliver usable 3D instance segmentation without manual annotations, indicating promising, low-effort transferability to other comparable TLS-based point cloud segmentation tasks.

  </details>



- **4D Synchronized Fields: Motion-Language Gaussian Splatting for Temporal Scene Understanding**  
  Mohamed Rayan Barhdadi, Samir Abdaljalil, Rasul Khanbayov, Erchin Serpedin, Hasan Kurban  
  _2026-03-15_ · https://arxiv.org/abs/2603.14301v1  
  <details><summary>Abstract</summary>

  Current 4D representations decouple geometry, motion, and semantics: reconstruction methods discard interpretable motion structure; language-grounded methods attach semantics after motion is learned, blind to how objects move; and motion-aware methods encode dynamics as opaque per-point residuals without object-level organization. We propose 4D Synchronized Fields, a 4D Gaussian representation that learns object-factored motion in-loop during reconstruction and synchronizes language to the resulting kinematics through a per-object conditioned field. Each Gaussian trajectory is decomposed into shared object motion plus an implicit residual, and a kinematic-conditioned ridge map predicts temporal semantic variation, yielding a single representation in which reconstruction, motion, and semantics are structurally coupled and enabling open-vocabulary temporal queries that retrieve both objects and moments. On HyperNeRF, 4D Synchronized Fields achieves 28.52 dB mean PSNR, the highest among all language-grounded and motion-aware baselines, within 1.5 dB of reconstruction-only methods. On targeted temporal-state retrieval, the kinematic-conditioned field attains 0.884 mean accuracy, 0.815 mean vIoU, and 0.733 mean tIoU, surpassing 4D LangSplat (0.620, 0.433, and 0.439 respectively) and LangSplat (0.415, 0.304, and 0.262). Ablation confirms that kinematic conditioning is the primary driver, accounting for +0.45 tIoU over a static-embedding-only baseline. 4D Synchronized Fields is the only method that jointly exposes interpretable motion primitives and temporally grounded language fields from a single trained representation. Code will be released.

  </details>



- **S2GS: Streaming Semantic Gaussian Splatting for Online Scene Understanding and Reconstruction**  
  Renhe Zhang, Yuyang Tan, Jingyu Gong, Zhizhong Zhang, Lizhuang Ma, Yuan Xie, Xin Tan  
  _2026-03-15_ · https://arxiv.org/abs/2603.14232v1  
  <details><summary>Abstract</summary>

  Existing offline feed-forward methods for joint scene understanding and reconstruction on long image streams often repeatedly perform global computation over an ever-growing set of past observations, causing runtime and GPU memory to increase rapidly with sequence length and limiting scalability. We propose Streaming Semantic Gaussian Splatting (S2GS), a strictly causal, incremental 3D Gaussian semantic field framework: it does not leverage future frames and continuously updates scene geometry, appearance, and instance-level semantics without reprocessing historical frames, enabling scalable online joint reconstruction and understanding. S2GS adopts a geometry-semantic decoupled dual-backbone design: the geometry branch performs causal modeling to drive incremental Gaussian updates, while the semantic branch leverages a 2D foundation vision model and a query-driven decoder to predict segmentation masks and identity embeddings, further stabilized by query-level contrastive alignment and lightweight online association with an instance memory. Experiments show that S2GS matches or outperforms strong offline baselines on joint reconstruction-and-understanding benchmarks, while significantly improving long-horizon scalability: it processes 1,000+ frames with much slower growth in runtime and GPU memory, whereas offline global-processing baselines typically run out of memory at around 80 frames under the same setting.

  </details>



- **PhyGaP: Physically-Grounded Gaussians with Polarization Cues**  
  Jiale Wu, Xiaoyang Bai, Zongqi He, Weiwei Xu, Yifan Peng  
  _2026-03-14_ · https://arxiv.org/abs/2603.14001v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated great success in modeling reflective 3D objects and their interaction with the environment via deferred rendering (DR). However, existing methods often struggle with correctly reconstructing physical attributes such as albedo and reflectance, and therefore they do not support high-fidelity relighting. Observing that this limitation stems from the lack of shape and material information in RGB images, we present PhyGaP, a physically-grounded 3DGS method that leverages polarization cues to facilitate precise reflection decomposition and visually consistent relighting of reconstructed objects. Specifically, we design a polarimetric deferred rendering (PolarDR) process to model polarization by reflection, and a self-occlusion-aware environment map building technique (GridMap) to resolve indirect lighting of non-convex objects. We validate on multiple synthetic and real-world scenes, including those featuring only partial polarization cues, that PhyGaP not only excels in reconstructing the appearance and surface normal of reflective 3D objects (~2 dB in PSNR and 45.7% in Cosine Distance better than existing RGB-based methods on average), but also achieves state-of-the-art inverse rendering and relighting capability. Our code will be released soon.

  </details>



- **Scene Generation at Absolute Scale: Utilizing Semantic and Geometric Guidance From Text for Accurate and Interpretable 3D Indoor Scene Generation**  
  Stefan Ainetter, Thomas Deixelberger, Edoardo A. Dominici, Philipp Drescher, Konstantinos Vardis, Markus Steinberger  
  _2026-03-14_ · https://arxiv.org/abs/2603.13910v1  
  <details><summary>Abstract</summary>

  We present GuidedSceneGen, a text-to-3D generation framework that produces metrically accurate, globally consistent, and semantically interpretable indoor scenes. Unlike prior text-driven methods that often suffer from geometric drift or scale ambiguity, our approach maintains an absolute world coordinate frame throughout the entire generation process. Starting from a textual scene description, we predict a global 3D layout encoding both semantic and geometric structure, which serves as a guiding proxy for downstream stages. A semantics- and depth-conditioned panoramic diffusion model then synthesizes 360° imagery aligned with the global layout, substantially improving spatial coherence. To explore unobserved regions, we employ a video diffusion model guided by optimized camera trajectories that balances coverage and collision avoidance, achieving up to 10x faster sampling compared to exhaustive path exploration. The generated views are fused using 3D Gaussian Splatting, yielding a consistent and fully navigable 3D scene in absolute scale. GuidedSceneGen enables accurate transfer of object poses and semantic labels from layout to reconstruction, and supports progressive scene expansion without re-alignment. Quantitative results and a user study demonstrate greater 3D consistency and layout plausibility compared to recent panoramic text-to-3D baselines.

  </details>


