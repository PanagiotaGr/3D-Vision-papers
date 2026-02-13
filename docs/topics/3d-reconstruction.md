# 3D Reconstruction

_Updated: 2026-02-13 07:13 UTC_

Total papers shown: **22**


---

- **UPDA: Unsupervised Progressive Domain Adaptation for No-Reference Point Cloud Quality Assessment**  
  Bingxu Xie, Fang Zhou, Jincan Wu, Yonghui Liu, Weiqing Li, Zhiyong Su  
  _2026-02-12_ · https://arxiv.org/abs/2602.11969v1  
  <details><summary>Abstract</summary>

  While no-reference point cloud quality assessment (NR-PCQA) approaches have achieved significant progress over the past decade, their performance often degrades substantially when a distribution gap exists between the training (source domain) and testing (target domain) data. However, to date, limited attention has been paid to transferring NR-PCQA models across domains. To address this challenge, we propose the first unsupervised progressive domain adaptation (UPDA) framework for NR-PCQA, which introduces a two-stage coarse-to-fine alignment paradigm to address domain shifts. At the coarse-grained stage, a discrepancy-aware coarse-grained alignment method is designed to capture relative quality relationships between cross-domain samples through a novel quality-discrepancy-aware hybrid loss, circumventing the challenges of direct absolute feature alignment. At the fine-grained stage, a perception fusion fine-grained alignment approach with symmetric feature fusion is developed to identify domain-invariant features, while a conditional discriminator selectively enhances the transfer of quality-relevant features. Extensive experiments demonstrate that the proposed UPDA effectively enhances the performance of NR-PCQA methods in cross-domain scenarios, validating its practical applicability. The code is available at https://github.com/yokeno1/UPDA-main.

  </details>



- **RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval**  
  Khanh Nguyen, Dasith de Silva Edirimuni, Ghulam Mubashar Hassan, Ajmal Mian  
  _2026-02-12_ · https://arxiv.org/abs/2602.11673v1  
  <details><summary>Abstract</summary>

  3D assets have rapidly expanded in quantity and diversity due to the growing popularity of virtual reality and gaming. As a result, text-to-shape retrieval has become essential in facilitating intuitive search within large repositories. However, existing methods require canonical poses and support few object categories, limiting their real-world applicability where objects can belong to diverse classes and appear in random orientations. To address this challenge, we propose RI-Mamba, the first rotation-invariant state-space model for point clouds. RI-Mamba defines global and local reference frames to disentangle pose from geometry and uses Hilbert sorting to construct token sequences with meaningful geometric structure while maintaining rotation invariance. We further introduce a novel strategy to compute orientational embeddings and reintegrate them via feature-wise linear modulation, effectively recovering spatial context and enhancing model expressiveness. Our strategy is inherently compatible with state-space models and operates in linear time. To scale up retrieval, we adopt cross-modal contrastive learning with automated triplet generation, allowing training on diverse datasets without manual annotation. Extensive experiments demonstrate RI-Mamba's superior representational capacity and robustness, achieving state-of-the-art performance on the OmniObject3D benchmark across more than 200 object categories under arbitrary orientations. Our code will be made available at https://github.com/ndkhanh360/RI-Mamba.git.

  </details>



- **Electrostatics-Inspired Surface Reconstruction (EISR): Recovering 3D Shapes as a Superposition of Poisson's PDE Solutions**  
  Diego Patiño, Knut Peterson, Kostas Daniilidis, David K. Han  
  _2026-02-12_ · https://arxiv.org/abs/2602.11642v1  
  <details><summary>Abstract</summary>

  Implicit shape representation, such as SDFs, is a popular approach to recover the surface of a 3D shape as the level sets of a scalar field. Several methods approximate SDFs using machine learning strategies that exploit the knowledge that SDFs are solutions of the Eikonal partial differential equation (PDEs). In this work, we present a novel approach to surface reconstruction by encoding it as a solution to a proxy PDE, namely Poisson's equation. Then, we explore the connection between Poisson's equation and physics, e.g., the electrostatic potential due to a positive charge density. We employ Green's functions to obtain a closed-form parametric expression for the PDE's solution, and leverage the linearity of our proxy PDE to find the target shape's implicit field as a superposition of solutions. Our method shows improved results in approximating high-frequency details, even with a small number of shape priors.

  </details>



- **HyperDet: 3D Object Detection with Hyper 4D Radar Point Clouds**  
  Yichun Xiao, Runwei Guan, Fangqiang Ding  
  _2026-02-12_ · https://arxiv.org/abs/2602.11554v1  
  <details><summary>Abstract</summary>

  4D mmWave radar provides weather-robust, velocity-aware measurements and is more cost-effective than LiDAR. However, radar-only 3D detection still trails LiDAR-based systems because radar point clouds are sparse, irregular, and often corrupted by multipath noise, yielding weak and unstable geometry. We present HyperDet, a detector-agnostic radar-only 3D detection framework that constructs a task-aware hyper 4D radar point cloud for standard LiDAR-oriented detectors. HyperDet aggregates returns from multiple surround-view 4D radars over consecutive frames to improve coverage and density, then applies geometry-aware cross-sensor consensus validation with a lightweight self-consistency check outside overlap regions to suppress inconsistent returns. It further integrates a foreground-focused diffusion module with training-time mixed radar-LiDAR supervision to densify object structures while lifting radar attributes (e.g., Doppler, RCS); the model is distilled into a consistency model for single-step inference. On MAN TruckScenes, HyperDet consistently improves over raw radar inputs with VoxelNeXt and CenterPoint, partially narrowing the radar-LiDAR gap. These results show that input-level refinement enables radar to better leverage LiDAR-oriented detectors without architectural modifications.

  </details>



- **Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation**  
  Penghui Ruan, Bojia Zi, Xianbiao Qi, Youze Huang, Rong Xiao, Pichao Wang, Jiannong Cao, Yuhui Shi  
  _2026-02-11_ · https://arxiv.org/abs/2602.11440v1  
  <details><summary>Abstract</summary>

  Object-level manipulation, relocating or reorienting objects in images or videos while preserving scene realism, is central to film post-production, AR, and creative editing. Yet existing methods struggle to jointly achieve three core goals: background preservation, geometric consistency under viewpoint shifts, and user-controllable transformations. Geometry-based approaches offer precise control but require explicit 3D reconstruction and generalize poorly; diffusion-based methods generalize better but lack fine-grained geometric control. We present Ctrl&Shift, an end-to-end diffusion framework to achieve geometry-consistent object manipulation without explicit 3D representations. Our key insight is to decompose manipulation into two stages, object removal and reference-guided inpainting under explicit camera pose control, and encode both within a unified diffusion process. To enable precise, disentangled control, we design a multi-task, multi-stage training strategy that separates background, identity, and pose signals across tasks. To improve generalization, we introduce a scalable real-world dataset construction pipeline that generates paired image and video samples with estimated relative camera poses. Extensive experiments demonstrate that Ctrl&Shift achieves state-of-the-art results in fidelity, viewpoint consistency, and controllability. To our knowledge, this is the first framework to unify fine-grained geometric control and real-world generalization for object manipulation, without relying on any explicit 3D modeling.

  </details>



- **Filmsticking++: Rapid Film Sticking for Explicit Surface Reconstruction**  
  Pengfei Wang, Jian Liu, Qiujie Dong, Shiqing Xin, Yuanfeng Zhou, Changhe Tu, Caiming Zhang, Wenping Wang  
  _2026-02-11_ · https://arxiv.org/abs/2602.11433v1  
  <details><summary>Abstract</summary>

  Explicit surface reconstruction aims to generate a surface mesh that exactly interpolates a given point cloud. This requirement is crucial when the point cloud must lie non-negotiably on the final surface to preserve sharp features and fine geometric details. However, the task becomes substantially challenging with low-quality point clouds, due to inherent reconstruction ambiguities compounded by combinatorial complexity. A previous method using filmsticking technique by iteratively compute restricted Voronoi diagram to address these issues, ensures to produce a watertight manifold, setting a new benchmark as the state-of-the-art (SOTA) technique. Unfortunately, RVD-based filmsticking is inability to interpolate all points in the case of deep internal cavities, resulting in very likely is the generation of faulty topology. The cause of this issue is that RVD-based filmsticking has inherent limitations due to Euclidean distance metrics. In this paper, we extend the filmsticking technique, named Filmsticking++. Filmsticking++ reconstructing an explicit surface from points without normals. On one hand, Filmsticking++ break through the inherent limitations of Euclidean distance by employing a weighted-distance-based Restricted Power Diagram, which guarantees that all points are interpolated. On the other hand, we observe that as the guiding surface increasingly approximates the target shape, the external medial axis is gradually expelled outside the guiding surface. Building on this observation, we propose placing virtual sites inside the guiding surface to accelerate the expulsion of the external medial axis from its interior. To summarize, contrary to the SOTA method, Filmsticking++ demonstrates multiple benefits, including decreases computational cost, improved robustness and scalability.

  </details>



- **MDE-VIO: Enhancing Visual-Inertial Odometry Using Learned Depth Priors**  
  Arda Alniak, Sinan Kalkan, Mustafa Mert Ankarali, Afsar Saranli, Abdullah Aydin Alatan  
  _2026-02-11_ · https://arxiv.org/abs/2602.11323v1  
  <details><summary>Abstract</summary>

  Traditional monocular Visual-Inertial Odometry (VIO) systems struggle in low-texture environments where sparse visual features are insufficient for accurate pose estimation. To address this, dense Monocular Depth Estimation (MDE) has been widely explored as a complementary information source. While recent Vision Transformer (ViT) based complex foundational models offer dense, geometrically consistent depth, their computational demands typically preclude them from real-time edge deployment. Our work bridges this gap by integrating learned depth priors directly into the VINS-Mono optimization backend. We propose a novel framework that enforces affine-invariant depth consistency and pairwise ordinal constraints, explicitly filtering unstable artifacts via variance-based gating. This approach strictly adheres to the computational limits of edge devices while robustly recovering metric scale. Extensive experiments on the TartanGround and M3ED datasets demonstrate that our method prevents divergence in challenging scenarios and delivers significant accuracy gains, reducing Absolute Trajectory Error (ATE) by up to 28.3%. Code will be made available.

  </details>



- **From Circuits to Dynamics: Understanding and Stabilizing Failure in 3D Diffusion Transformers**  
  Maximilian Plattner, Fabian Paischer, Johannes Brandstetter, Arturs Berzins  
  _2026-02-11_ · https://arxiv.org/abs/2602.11130v1  
  <details><summary>Abstract</summary>

  Reliable surface completion from sparse point clouds underpins many applications spanning content creation and robotics. While 3D diffusion transformers attain state-of-the-art results on this task, we uncover that they exhibit a catastrophic mode of failure: arbitrarily small on-surface perturbations to the input point cloud can fracture the output into multiple disconnected pieces -- a phenomenon we call Meltdown. Using activation-patching from mechanistic interpretability, we localize Meltdown to a single early denoising cross-attention activation. We find that the singular-value spectrum of this activation provides a scalar proxy: its spectral entropy rises when fragmentation occurs and returns to baseline when patched. Interpreted through diffusion dynamics, we show that this proxy tracks a symmetry-breaking bifurcation of the reverse process. Guided by this insight, we introduce PowerRemap, a test-time control that stabilizes sparse point-cloud conditioning. We demonstrate that Meltdown persists across state-of-the-art architectures (WaLa, Make-a-Shape), datasets (GSO, SimJEB) and denoising strategies (DDPM, DDIM), and that PowerRemap effectively counters this failure with stabilization rates of up to 98.3%. Overall, this work is a case study on how diffusion model behavior can be understood and guided based on mechanistic analysis, linking a circuit-level cross-attention mechanism to diffusion-dynamics accounts of trajectory bifurcations.

  </details>



- **PuriLight: A Lightweight Shuffle and Purification Framework for Monocular Depth Estimation**  
  Yujie Chen, Li Zhang, Xiaomeng Chu, Tian Zhang  
  _2026-02-11_ · https://arxiv.org/abs/2602.11066v1  
  <details><summary>Abstract</summary>

  We propose PuriLight, a lightweight and efficient framework for self-supervised monocular depth estimation, to address the dual challenges of computational efficiency and detail preservation. While recent advances in self-supervised depth estimation have reduced reliance on ground truth supervision, existing approaches remain constrained by either bulky architectures compromising practicality or lightweight models sacrificing structural precision. These dual limitations underscore the critical need to develop lightweight yet structurally precise architectures. Our framework addresses these limitations through a three-stage architecture incorporating three novel modules: the Shuffle-Dilation Convolution (SDC) module for local feature extraction, the Rotation-Adaptive Kernel Attention (RAKA) module for hierarchical feature enhancement, and the Deep Frequency Signal Purification (DFSP) module for global feature purification. Through effective collaboration, these modules enable PuriLight to achieve both lightweight and accurate feature extraction and processing. Extensive experiments demonstrate that PuriLight achieves state-of-the-art performance with minimal training parameters while maintaining exceptional computational efficiency. Codes will be available at https://github.com/ishrouder/PuriLight.

  </details>



- **ReTracing: An Archaeological Approach Through Body, Machine, and Generative Systems**  
  Yitong Wang, Yue Yao  
  _2026-02-11_ · https://arxiv.org/abs/2602.11242v1  
  <details><summary>Abstract</summary>

  We present ReTracing, a multi-agent embodied performance art that adopts an archaeological approach to examine how artificial intelligence shapes, constrains, and produces bodily movement. Drawing from science-fiction novels, the project extracts sentences that describe human-machine interaction. We use large language models (LLMs) to generate paired prompts "what to do" and "what not to do" for each excerpt. A diffusion-based text-to-video model transforms these prompts into choreographic guides for a human performer and motor commands for a quadruped robot. Both agents enact the actions on a mirrored floor, captured by multi-camera motion tracking and reconstructed into 3D point clouds and motion trails, forming a digital archive of motion traces. Through this process, ReTracing serves as a novel approach to reveal how generative systems encode socio-cultural biases through choreographed movements. Through an immersive interplay of AI, human, and robot, ReTracing confronts a critical question of our time: What does it mean to be human among AIs that also move, think, and leave traces behind?

  </details>



- **LaSSM: Efficient Semantic-Spatial Query Decoding via Local Aggregation and State Space Models for 3D Instance Segmentation**  
  Lei Yao, Yi Wang, Yawen Cui, Moyun Liu, Lap-Pui Chau  
  _2026-02-11_ · https://arxiv.org/abs/2602.11007v1  
  <details><summary>Abstract</summary>

  Query-based 3D scene instance segmentation from point clouds has attained notable performance. However, existing methods suffer from the query initialization dilemma due to the sparse nature of point clouds and rely on computationally intensive attention mechanisms in query decoders. We accordingly introduce LaSSM, prioritizing simplicity and efficiency while maintaining competitive performance. Specifically, we propose a hierarchical semantic-spatial query initializer to derive the query set from superpoints by considering both semantic cues and spatial distribution, achieving comprehensive scene coverage and accelerated convergence. We further present a coordinate-guided state space model (SSM) decoder that progressively refines queries. The novel decoder features a local aggregation scheme that restricts the model to focus on geometrically coherent regions and a spatial dual-path SSM block to capture underlying dependencies within the query set by integrating associated coordinates information. Our design enables efficient instance prediction, avoiding the incorporation of noisy information and reducing redundant computation. LaSSM ranks first place on the latest ScanNet++ V2 leaderboard, outperforming the previous best method by 2.5% mAP with only 1/3 FLOPs, demonstrating its superiority in challenging large-scale scene instance segmentation. LaSSM also achieves competitive performance on ScanNet, ScanNet200, S3DIS and ScanNet++ V1 benchmarks with less computational cost. Extensive ablation studies and qualitative results validate the effectiveness of our design. The code and weights are available at https://github.com/RayYoh/LaSSM.

  </details>



- **Interpretable Vision Transformers in Monocular Depth Estimation via SVDA**  
  Vasileios Arampatzakis, George Pavlidis, Nikolaos Mitianoudis, Nikos Papamarkos  
  _2026-02-11_ · https://arxiv.org/abs/2602.11005v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation is a central problem in computer vision with applications in robotics, AR, and autonomous driving, yet the self-attention mechanisms that drive modern Transformer architectures remain opaque. We introduce SVD-Inspired Attention (SVDA) into the Dense Prediction Transformer (DPT), providing the first spectrally structured formulation of attention for dense prediction tasks. SVDA decouples directional alignment from spectral modulation by embedding a learnable diagonal matrix into normalized query-key interactions, enabling attention maps that are intrinsically interpretable rather than post-hoc approximations. Experiments on KITTI and NYU-v2 show that SVDA preserves or slightly improves predictive accuracy while adding only minor computational overhead. More importantly, SVDA unlocks six spectral indicators that quantify entropy, rank, sparsity, alignment, selectivity, and robustness. These reveal consistent cross-dataset and depth-wise patterns in how attention organizes during training, insights that remain inaccessible in standard Transformers. By shifting the role of attention from opaque mechanism to quantifiable descriptor, SVDA redefines interpretability in monocular depth estimation and opens a principled avenue toward transparent dense prediction models.

  </details>



- **Viewpoint Recommendation for Point Cloud Labeling through Interaction Cost Modeling**  
  Yu Zhang, Xinyi Zhao, Chongke Bi, Siming Chen  
  _2026-02-11_ · https://arxiv.org/abs/2602.10871v1  
  <details><summary>Abstract</summary>

  Semantic segmentation of 3D point clouds is important for many applications, such as autonomous driving. To train semantic segmentation models, labeled point cloud segmentation datasets are essential. Meanwhile, point cloud labeling is time-consuming for annotators, which typically involves tuning the camera viewpoint and selecting points by lasso. To reduce the time cost of point cloud labeling, we propose a viewpoint recommendation approach to reduce annotators' labeling time costs. We adapt Fitts' law to model the time cost of lasso selection in point clouds. Using the modeled time cost, the viewpoint that minimizes the lasso selection time cost is recommended to the annotator. We build a data labeling system for semantic segmentation of 3D point clouds that integrates our viewpoint recommendation approach. The system enables users to navigate to recommended viewpoints for efficient annotation. Through an ablation study, we observed that our approach effectively reduced the data labeling time cost. We also qualitatively compare our approach with previous viewpoint selection approaches on different datasets.

  </details>



- **DMP-3DAD: Cross-Category 3D Anomaly Detection via Realistic Depth Map Projection with Few Normal Samples**  
  Zi Wang, Katsuya Hotta, Koichiro Kamide, Yawen Zou, Jianjian Qin, Chao Zhang, Jun Yu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10806v1  
  <details><summary>Abstract</summary>

  Cross-category anomaly detection for 3D point clouds aims to determine whether an unseen object belongs to a target category using only a few normal examples. Most existing methods rely on category-specific training, which limits their flexibility in few-shot scenarios. In this paper, we propose DMP-3DAD, a training-free framework for cross-category 3D anomaly detection based on multi-view realistic depth map projection. Specifically, by converting point clouds into a fixed set of realistic depth images, our method leverages a frozen CLIP visual encoder to extract multi-view representations and performs anomaly detection via weighted feature similarity, which does not require any fine-tuning or category-dependent adaptation. Extensive experiments on the ShapeNetPart dataset demonstrate that DMP-3DAD achieves state-of-the-art performance under few-shot setting. The results show that the proposed approach provides a simple yet effective solution for practical cross-category 3D anomaly detection.

  </details>



- **AugVLA-3D: Depth-Driven Feature Augmentation for Vision-Language-Action Models**  
  Zhifeng Rao, Wenlong Chen, Lei Xie, Xia Hua, Dongfu Yin, Zhen Tian, F. Richard Yu  
  _2026-02-11_ · https://arxiv.org/abs/2602.10698v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have recently achieved remarkable progress in robotic perception and control, yet most existing approaches primarily rely on VLM trained using 2D images, which limits their spatial understanding and action grounding in complex 3D environments. To address this limitation, we propose a novel framework that integrates depth estimation into VLA models to enrich 3D feature representations. Specifically, we employ a depth estimation baseline called VGGT to extract geometry-aware 3D cues from standard RGB inputs, enabling efficient utilization of existing large-scale 2D datasets while implicitly recovering 3D structural information. To further enhance the reliability of these depth-derived features, we introduce a new module called action assistant, which constrains the learned 3D representations with action priors and ensures their consistency with downstream control tasks. By fusing the enhanced 3D features with conventional 2D visual tokens, our approach significantly improves the generalization ability and robustness of VLA models. Experimental results demonstrate that the proposed method not only strengthens perception in geometrically ambiguous scenarios but also leads to superior action prediction accuracy. This work highlights the potential of depth-driven data augmentation and auxiliary expert supervision for bridging the gap between 2D observations and 3D-aware decision-making in robotic systems.

  </details>



- **End-to-End LiDAR optimization for 3D point cloud registration**  
  Siddhant Katyan, Marc-André Gardner, Jean-François Lalonde  
  _2026-02-11_ · https://arxiv.org/abs/2602.10492v1  
  <details><summary>Abstract</summary>

  LiDAR sensors are a key modality for 3D perception, yet they are typically designed independently of downstream tasks such as point cloud registration. Conventional registration operates on pre-acquired datasets with fixed LiDAR configurations, leading to suboptimal data collection and significant computational overhead for sampling, noise filtering, and parameter tuning. In this work, we propose an adaptive LiDAR sensing framework that dynamically adjusts sensor parameters, jointly optimizing LiDAR acquisition and registration hyperparameters. By integrating registration feedback into the sensing loop, our approach optimally balances point density, noise, and sparsity, improving registration accuracy and efficiency. Evaluations in the CARLA simulation demonstrate that our method outperforms fixed-parameter baselines while retaining generalization abilities, highlighting the potential of adaptive LiDAR for autonomous perception and robotic applications.

  </details>



- **ERGO: Excess-Risk-Guided Optimization for High-Fidelity Monocular 3D Gaussian Splatting**  
  Zehua Ma, Hanhui Li, Zhenyu Xie, Xiaonan Luo, Michael Kampffmeyer, Feng Gao, Xiaodan Liang  
  _2026-02-10_ · https://arxiv.org/abs/2602.10278v1  
  <details><summary>Abstract</summary>

  Generating 3D content from a single image remains a fundamentally challenging and ill-posed problem due to the inherent absence of geometric and textural information in occluded regions. While state-of-the-art generative models can synthesize auxiliary views to provide additional supervision, these views inevitably contain geometric inconsistencies and textural misalignments that propagate and amplify artifacts during 3D reconstruction. To effectively harness these imperfect supervisory signals, we propose an adaptive optimization framework guided by excess risk decomposition, termed ERGO. Specifically, ERGO decomposes the optimization losses in 3D Gaussian splatting into two components, i.e., excess risk that quantifies the suboptimality gap between current and optimal parameters, and Bayes error that models the irreducible noise inherent in synthesized views. This decomposition enables ERGO to dynamically estimate the view-specific excess risk and adaptively adjust loss weights during optimization. Furthermore, we introduce geometry-aware and texture-aware objectives that complement the excess-risk-derived weighting mechanism, establishing a synergistic global-local optimization paradigm. Consequently, ERGO demonstrates robustness against supervision noise while consistently enhancing both geometric fidelity and textural quality of the reconstructed 3D content. Extensive experiments on the Google Scanned Objects dataset and the OmniObject3D dataset demonstrate the superiority of ERGO over existing state-of-the-art methods.

  </details>



- **XSPLAIN: XAI-enabling Splat-based Prototype Learning for Attribute-aware INterpretability**  
  Dominik Galus, Julia Farganus, Tymoteusz Zapala, Mikołaj Czachorowski, Piotr Borycki, Przemysław Spurek, Piotr Syga  
  _2026-02-10_ · https://arxiv.org/abs/2602.10239v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has rapidly become a standard for high-fidelity 3D reconstruction, yet its adoption in multiple critical domains is hindered by the lack of interpretability of the generation models as well as classification of the Splats. While explainability methods exist for other 3D representations, like point clouds, they typically rely on ambiguous saliency maps that fail to capture the volumetric coherence of Gaussian primitives. We introduce XSPLAIN, the first ante-hoc, prototype-based interpretability framework designed specifically for 3DGS classification. Our approach leverages a voxel-aggregated PointNet backbone and a novel, invertible orthogonal transformation that disentangles feature channels for interpretability while strictly preserving the original decision boundaries. Explanations are grounded in representative training examples, enabling intuitive ``this looks like that'' reasoning without any degradation in classification performance. A rigorous user study (N=51) demonstrates a decisive preference for our approach: participants selected XSPLAIN explanations 48.4\% of the time as the best, significantly outperforming baselines $(p<0.001)$, showing that XSPLAIN provides transparency and user trust. The source code for this work is available at: https://github.com/Solvro/ml-splat-xai

  </details>



- **VersaViT: Enhancing MLLM Vision Backbones via Task-Guided Optimization**  
  Yikun Liu, Yuan Liu, Shangzhe Di, Haicheng Wang, Zhongyin Zhao, Le Tian, Xiao Zhou, Jie Zhou, Jiangchao Yao, Yanfeng Wang, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09934v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently achieved remarkable success in visual-language understanding, demonstrating superior high-level semantic alignment within their vision encoders. An important question thus arises: Can these encoders serve as versatile vision backbones, capable of reliably performing classic vision-centric tasks as well? To address the question, we make the following contributions: (i) we identify that the vision encoders within MLLMs exhibit deficiencies in their dense feature representations, as evidenced by their suboptimal performance on dense prediction tasks (e.g., semantic segmentation, depth estimation); (ii) we propose VersaViT, a well-rounded vision transformer that instantiates a novel multi-task framework for collaborative post-training. This framework facilitates the optimization of the vision backbone via lightweight task heads with multi-granularity supervision; (iii) extensive experiments across various downstream tasks demonstrate the effectiveness of our method, yielding a versatile vision backbone suited for both language-mediated reasoning and pixel-level understanding.

  </details>



- **SARS: A Novel Face and Body Shape and Appearance Aware 3D Reconstruction System extends Morphable Models**  
  Gulraiz Khan, Kenneth Y. Wertheim, Kevin Pimbblet, Waqas Ahmed  
  _2026-02-10_ · https://arxiv.org/abs/2602.09918v1  
  <details><summary>Abstract</summary>

  Morphable Models (3DMMs) are a type of morphable model that takes 2D images as inputs and recreates the structure and physical appearance of 3D objects, especially human faces and bodies. 3DMM combines identity and expression blendshapes with a basic face mesh to create a detailed 3D model. The variability in the 3D Morphable models can be controlled by tuning diverse parameters. They are high-level image descriptors, such as shape, texture, illumination, and camera parameters. Previous research in 3D human reconstruction concentrated solely on global face structure or geometry, ignoring face semantic features such as age, gender, and facial landmarks characterizing facial boundaries, curves, dips, and wrinkles. In order to accommodate changes in these high-level facial characteristics, this work introduces a shape and appearance-aware 3D reconstruction system (named SARS by us), a c modular pipeline that extracts body and face information from a single image to properly rebuild the 3D model of the human full body.

  </details>



- **VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model**  
  Hanqing Wang, Mingyu Liu, Xiaoyu Chen, Chengwei MA, Yiming Zhong, Wenti Yin, Yuhao Liu, Zhiqing Cui, Jiahao Yuan, Lu Dai, et al.  
  _2026-02-10_ · https://arxiv.org/abs/2602.09638v1  
  <details><summary>Abstract</summary>

  3D affordance grounding aims to highlight the actionable regions on 3D objects, which is crucial for robotic manipulation. Previous research primarily focused on learning affordance knowledge from static cues such as language and images, which struggle to provide sufficient dynamic interaction context that can reveal temporal and causal cues. To alleviate this predicament, we collect a comprehensive video-based 3D affordance dataset, \textit{VIDA}, which contains 38K human-object-interaction videos covering 16 affordance types, 38 object categories, and 22K point clouds. Based on \textit{VIDA}, we propose a strong baseline: VideoAfford, which activates multimodal large language models with additional affordance segmentation capabilities, enabling both world knowledge reasoning and fine-grained affordance grounding within a unified framework. To enhance action understanding capability, we leverage a latent action encoder to extract dynamic interaction priors from HOI videos. Moreover, we introduce a \textit{spatial-aware} loss function to enable VideoAfford to obtain comprehensive 3D spatial knowledge. Extensive experimental evaluations demonstrate that our model significantly outperforms well-established methods and exhibits strong open-world generalization with affordance reasoning abilities. All datasets and code will be publicly released to advance research in this area.

  </details>



- **RAD: Retrieval-Augmented Monocular Metric Depth Estimation for Underrepresented Classes**  
  Michael Baltaxe, Dan Levi, Sagie Benaim  
  _2026-02-10_ · https://arxiv.org/abs/2602.09532v1  
  <details><summary>Abstract</summary>

  Monocular Metric Depth Estimation (MMDE) is essential for physically intelligent systems, yet accurate depth estimation for underrepresented classes in complex scenes remains a persistent challenge. To address this, we propose RAD, a retrieval-augmented framework that approximates the benefits of multi-view stereo by utilizing retrieved neighbors as structural geometric proxies. Our method first employs an uncertainty-aware retrieval mechanism to identify low-confidence regions in the input and retrieve RGB-D context samples containing semantically similar content. We then process both the input and retrieved context via a dual-stream network and fuse them using a matched cross-attention module, which transfers geometric information only at reliable point correspondences. Evaluations on NYU Depth v2, KITTI, and Cityscapes demonstrate that RAD significantly outperforms state-of-the-art baselines on underrepresented classes, reducing relative absolute error by 29.2% on NYU Depth v2, 13.3% on KITTI, and 7.2% on Cityscapes, while maintaining competitive performance on standard in-domain benchmarks.

  </details>


