# 3D Reconstruction

_Updated: 2026-02-12 07:16 UTC_

Total papers shown: **20**


---

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



- **Bridging the Modality Gap in Roadside LiDAR: A Training-Free Vision-Language Model Framework for Vehicle Classification**  
  Yiqiao Li, Bo Shang, Jie Wei  
  _2026-02-10_ · https://arxiv.org/abs/2602.09425v1  
  <details><summary>Abstract</summary>

  Fine-grained truck classification is critical for intelligent transportation systems (ITS), yet current LiDAR-based methods face scalability challenges due to their reliance on supervised deep learning and labor-intensive manual annotation. Vision-Language Models (VLMs) offer promising few-shot generalization, but their application to roadside LiDAR is limited by a modality gap between sparse 3D point clouds and dense 2D imagery. We propose a framework that bridges this gap by adapting off-the-shelf VLMs for fine-grained truck classification without parameter fine-tuning. Our new depth-aware image generation pipeline applies noise removal, spatial and temporal registration, orientation rectification, morphological operations, and anisotropic smoothing to transform sparse, occluded LiDAR scans into depth-encoded 2D visual proxies. Validated on a real-world dataset of 20 vehicle classes, our approach achieves competitive classification accuracy with as few as 16-30 examples per class, offering a scalable alternative to data-intensive supervised baselines. We further observe a "Semantic Anchor" effect: text-based guidance regularizes performance in ultra-low-shot regimes $k < 4$, but degrades accuracy in more-shot settings due to semantic mismatch. Furthermore, we demonstrate the efficacy of this framework as a Cold Start strategy, using VLM-generated labels to bootstrap lightweight supervised models. Notably, the few-shot VLM-based model achieves over correct classification rate of 75 percent for specific drayage categories (20ft, 40ft, and 53ft containers) entirely without the costly training or fine-tuning, significantly reducing the intensive demands of initial manual labeling, thus achieving a method of practical use in ITS applications.

  </details>



- **Single-Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D**  
  Yan Luo, Advaith Ravishankar, Serena Liu, Yutong Yang, Mengyu Wang  
  _2026-02-10_ · https://arxiv.org/abs/2602.09407v1  
  <details><summary>Abstract</summary>

  A 3D understanding of anatomy is central to diagnosis and treatment planning, yet volumetric imaging remains costly with long wait times. Image-to-3D foundations models can solve this issue by reconstructing 3D data from 2D modalites. Current foundation models are trained on natural image distributions to reconstruct naturalistic objects from a single image by leveraging geometric priors across pixels. However, it is unclear whether these learned geometric priors transfer to medical data. In this study, we present a controlled zero-shot benchmark of single slice medical image-to-3D reconstruction across five state-of-the-art image-to-3D models: SAM3D, Hunyuan3D-2.1, Direct3D, Hi3DGen, and TripoSG. These are evaluated across six medical datasets spanning anatomical and pathological structures and two natrual datasets, using voxel based metrics and point cloud distance metrics. Across medical datasets, voxel based overlap remains moderate for all models, consistent with a depth reconstruction failure mode when inferring volume from a single slice. In contrast, global distance metrics show more separation between methods: SAM3D achieves the strongest overall topological similarity to ground truth medical 3D data, while alternative models are more prone to over-simplication of reconstruction. Our results quantify the limits of single-slice medical reconstruction and highlight depth ambiguity caused by the planar nature of 2D medical data, motivating multi-view image-to-3D reconstruction to enable reliable medical 3D inference.

  </details>



- **Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit**  
  Zhendong Wang, Cihan Ruan, Jingchuan Xiao, Chuqing Shi, Wei Jiang, Wei Wang, Wenjie Liu, Nam Ling  
  _2026-02-09_ · https://arxiv.org/abs/2602.08909v1  
  <details><summary>Abstract</summary>

  We investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multi-view optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patterns: mixture-structured scales and bimodal radiance across diverse scenes. To understand what determines these parameters, we apply learnability probes by training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental density-stratification. Dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals the dual character of RORs: geometric primitives where point clouds suffice, and view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.

  </details>



- **Overview and Comparison of AVS Point Cloud Compression Standard**  
  Wei Gao, Wenxu Gao, Xingming Mu, Changhao Peng, Ge Li  
  _2026-02-09_ · https://arxiv.org/abs/2602.08613v1  
  <details><summary>Abstract</summary>

  Point cloud is a prevalent 3D data representation format with significant application values in immersive media, autonomous driving, digital heritage protection, etc. However, the large data size of point clouds poses challenges to transmission and storage, which influences the wide deployments. Therefore, point cloud compression plays a crucial role in practical applications for both human and machine perception optimization. To this end, the Moving Picture Experts Group (MPEG) has established two standards for point cloud compression, including Geometry-based Point Cloud Compression (G-PCC) and Video-based Point Cloud Compression (V-PCC). In the meantime, the Audio Video coding Standard (AVS) Workgroup of China also have launched and completed the development for its first generation point cloud compression standard, namely AVS PCC. This new standardization effort has adopted many new coding tools and techniques, which are different from the other counterpart standards. This paper reviews the AVS PCC standard from two perspectives, i.e., the related technologies and performance comparisons.

  </details>



- **TIBR4D: Tracing-Guided Iterative Boundary Refinement for Efficient 4D Gaussian Segmentation**  
  He Wu, Xia Yan, Yanghui Xu, Liegang Xia, Jiazhou Chen  
  _2026-02-09_ · https://arxiv.org/abs/2602.08540v1  
  <details><summary>Abstract</summary>

  Object-level segmentation in dynamic 4D Gaussian scenes remains challenging due to complex motion, occlusions, and ambiguous boundaries. In this paper, we present an efficient learning-free 4D Gaussian segmentation framework that lifts video segmentation masks to 4D spaces, whose core is a two-stage iterative boundary refinement, TIBR4D. The first stage is an Iterative Gaussian Instance Tracing (IGIT) at the temporal segment level. It progressively refines Gaussian-to-instance probabilities through iterative tracing, and extracts corresponding Gaussian point clouds that better handle occlusions and preserve completeness of object structures compared to existing one-shot threshold-based methods. The second stage is a frame-wise Gaussian Rendering Range Control (RCC) via suppressing highly uncertain Gaussians near object boundaries while retaining their core contributions for more accurate boundaries. Furthermore, a temporal segmentation merging strategy is proposed for IGIT to balance identity consistency and dynamic awareness. Longer segments enforce stronger multi-frame constraints for stable identities, while shorter segments allow identity changes to be captured promptly. Experiments on HyperNeRF and Neu3D demonstrate that our method produces accurate object Gaussian point clouds with clearer boundaries and higher efficiency compared to SOTA methods.

  </details>



- **RealSynCol: a high-fidelity synthetic colon dataset for 3D reconstruction applications**  
  Chiara Lena, Davide Milesi, Alessandro Casella, Luca Carlini, Joseph C. Norton, James Martin, Bruno Scaglioni, Keith L. Obstein, Roberto De Sire, Marco Spadaccini, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.08397v1  
  <details><summary>Abstract</summary>

  Deep learning has the potential to improve colonoscopy by enabling 3D reconstruction of the colon, providing a comprehensive view of mucosal surfaces and lesions, and facilitating the identification of unexplored areas. However, the development of robust methods is limited by the scarcity of large-scale ground truth data. We propose RealSynCol, a highly realistic synthetic dataset designed to replicate the endoscopic environment. Colon geometries extracted from 10 CT scans were imported into a virtual environment that closely mimics intraoperative conditions and rendered with realistic vascular textures. The resulting dataset comprises 28\,130 frames, paired with ground truth depth maps, optical flow, 3D meshes, and camera trajectories. A benchmark study was conducted to evaluate the available synthetic colon datasets for the tasks of depth and pose estimation. Results demonstrate that the high realism and variability of RealSynCol significantly enhance generalization performance on clinical images, proving it to be a powerful tool for developing deep learning algorithms to support endoscopic diagnosis.

  </details>


