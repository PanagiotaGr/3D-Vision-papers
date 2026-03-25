# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-25 07:16 UTC_

Total papers shown: **50**


---

- **MedObvious: Exposing the Medical Moravec's Paradox in VLMs via Clinical Triage**  
  Ufaq Khan, Umair Nawaz, L D M S S Teja, Numaan Saeed, Muhammad Bilal, Yutong Xie, Mohammad Yaqub, Muhammad Haris Khan  
  _2026-03-24_ · https://arxiv.org/abs/2603.23501v1  
  <details><summary>Abstract</summary>

  Vision Language Models (VLMs) are increasingly used for tasks like medical report generation and visual question answering. However, fluent diagnostic text does not guarantee safe visual understanding. In clinical practice, interpretation begins with pre-diagnostic sanity checks: verifying that the input is valid to read (correct modality and anatomy, plausible viewpoint and orientation, and no obvious integrity violations). Existing benchmarks largely assume this step is solved, and therefore miss a critical failure mode: a model can produce plausible narratives even when the input is inconsistent or invalid. We introduce MedObvious, a 1,880-task benchmark that isolates input validation as a set-level consistency capability over small multi-panel image sets: the model must identify whether any panel violates expected coherence. MedObvious spans five progressive tiers, from basic orientation/modality mismatches to clinically motivated anatomy/viewpoint verification and triage-style cues, and includes five evaluation formats to test robustness across interfaces. Evaluating 17 different VLMs, we find that sanity checking remains unreliable: several models hallucinate anomalies on normal (negative-control) inputs, performance degrades when scaling to larger image sets, and measured accuracy varies substantially between multiple-choice and open-ended settings. These results show that pre-diagnostic verification remains unsolved for medical VLMs and should be treated as a distinct, safety-critical capability before deployment.

  </details>



- **WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG**  
  Zhen Li, Zian Meng, Shuwei Shi, Wenshuo Peng, Yuwei Wu, Bo Zheng, Chuanhao Li, Kaipeng Zhang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23497v1  
  <details><summary>Abstract</summary>

  Dynamical systems theory and reinforcement learning view world evolution as latent-state dynamics driven by actions, with visual observations providing partial information about the state. Recent video world models attempt to learn this action-conditioned dynamics from data. However, existing datasets rarely match the requirement: they typically lack diverse and semantically meaningful action spaces, and actions are directly tied to visual observations rather than mediated by underlying states. As a result, actions are often entangled with pixel-level changes, making it difficult for models to learn structured world dynamics and maintain consistent evolution over long horizons. In this paper, we propose WildWorld, a large-scale action-conditioned world modeling dataset with explicit state annotations, automatically collected from a photorealistic AAA action role-playing game (Monster Hunter: Wilds). WildWorld contains over 108 million frames and features more than 450 actions, including movement, attacks, and skill casting, together with synchronized per-frame annotations of character skeletons, world states, camera poses, and depth maps. We further derive WildBench to evaluate models through Action Following and State Alignment. Extensive experiments reveal persistent challenges in modeling semantically rich actions and maintaining long-horizon state consistency, highlighting the need for state-aware video generation. The project page is https://shandaai.github.io/wildworld-project/.

  </details>



- **RealMaster: Lifting Rendered Scenes into Photorealistic Video**  
  Dana Cohen-Bar, Ido Sobol, Raphael Bensadoun, Shelly Sheynin, Oran Gafni, Or Patashnik, Daniel Cohen-Or, Amit Zohar  
  _2026-03-24_ · https://arxiv.org/abs/2603.23462v1  
  <details><summary>Abstract</summary>

  State-of-the-art video generation models produce remarkable photorealism, but they lack the precise control required to align generated content with specific scene requirements. Furthermore, without an underlying explicit geometry, these models cannot guarantee 3D consistency. Conversely, 3D engines offer granular control over every scene element and provide native 3D consistency by design, yet their output often remains trapped in the "uncanny valley". Bridging this sim-to-real gap requires both structural precision, where the output must exactly preserve the geometry and dynamics of the input, and global semantic transformation, where materials, lighting, and textures must be holistically transformed to achieve photorealism. We present RealMaster, a method that leverages video diffusion models to lift rendered video into photorealistic video while maintaining full alignment with the output of the 3D engine. To train this model, we generate a paired dataset via an anchor-based propagation strategy, where the first and last frames are enhanced for realism and propagated across the intermediate frames using geometric conditioning cues. We then train an IC-LoRA on these paired videos to distill the high-quality outputs of the pipeline into a model that generalizes beyond the pipeline's constraints, handling objects and characters that appear mid-sequence and enabling inference without requiring anchor frames. Evaluated on complex GTA-V sequences, RealMaster significantly outperforms existing video editing baselines, improving photorealism while preserving the geometry, dynamics, and identity specified by the original 3D control.

  </details>



- **3DCity-LLM: Empowering Multi-modality Large Language Models for 3D City-scale Perception and Understanding**  
  Yiping Chen, Jinpeng Li, Wenyu Ke, Yang Luo, Jie Ouyang, Zhongjie He, Li Liu, Hongchao Fan, Hao Wu  
  _2026-03-24_ · https://arxiv.org/abs/2603.23447v1  
  <details><summary>Abstract</summary>

  While multi-modality large language models excel in object-centric or indoor scenarios, scaling them to 3D city-scale environments remains a formidable challenge. To bridge this gap, we propose 3DCity-LLM, a unified framework designed for 3D city-scale vision-language perception and understanding. 3DCity-LLM employs a coarse-to-fine feature encoding strategy comprising three parallel branches for target object, inter-object relationship, and global scene. To facilitate large-scale training, we introduce 3DCity-LLM-1.2M dataset that comprises approximately 1.2 million high-quality samples across seven representative task categories, ranging from fine-grained object analysis to multi-faceted scene planning. This strictly quality-controlled dataset integrates explicit 3D numerical information and diverse user-oriented simulations, enriching the question-answering diversity and realism of urban scenarios. Furthermore, we apply a multi-dimensional protocol based on text-similarity metrics and LLM-based semantic assessment to ensure faithful and comprehensive evaluations for all methods. Extensive experiments on two benchmarks demonstrate that 3DCity-LLM significantly outperforms existing state-of-the-art methods, offering a promising and meaningful direction for advancing spatial reasoning and urban intelligence. The source code and dataset are available at https://github.com/SYSU-3DSTAILab/3D-City-LLM.

  </details>



- **SIGMA: A Physics-Based Benchmark for Gas Chimney Understanding in Seismic Images**  
  Bao Truong, Quang Nguyen, Baoru Huang, Jinpei Han, Van Nguyen, Ngan Le, Minh-Tan Pham, Doan Huy Hien, Anh Nguyen  
  _2026-03-24_ · https://arxiv.org/abs/2603.23439v1  
  <details><summary>Abstract</summary>

  Seismic images reconstruct subsurface reflectivity from field recordings, guiding exploration and reservoir monitoring. Gas chimneys are vertical anomalies caused by subsurface fluid migration. Understanding these phenomena is crucial for assessing hydrocarbon potential and avoiding drilling hazards. However, accurate detection is challenging due to strong seismic attenuation and scattering. Traditional physics-based methods are computationally expensive and sensitive to model errors, while deep learning offers efficient alternatives, yet lacks labeled datasets. In this work, we introduce \textbf{SIGMA}, a new physics-based dataset for gas chimney understanding in seismic images, featuring (i) pixel-level gas-chimney mask for detection and (ii) paired degraded and ground-truth image for enhancement. We employed physics-based methods that cover a wide range of geological settings and data acquisition conditions. Comprehensive experiments demonstrate that SIGMA serves as a challenging benchmark for gas chimney interpretation and benefits general seismic understanding.

  </details>



- **Harnessing Lightweight Transformer with Contextual Synergic Enhancement for Efficient 3D Medical Image Segmentation**  
  Xinyu Liu, Zhen Chen, Wuyang Li, Chenxin Li, Yixuan Yuan  
  _2026-03-24_ · https://arxiv.org/abs/2603.23390v1  
  <details><summary>Abstract</summary>

  Transformers have shown remarkable performance in 3D medical image segmentation, but their high computational requirements and need for large amounts of labeled data limit their applicability. To address these challenges, we consider two crucial aspects: model efficiency and data efficiency. Specifically, we propose Light-UNETR, a lightweight transformer designed to achieve model efficiency. Light-UNETR features a Lightweight Dimension Reductive Attention (LIDR) module, which reduces spatial and channel dimensions while capturing both global and local features via multi-branch attention. Additionally, we introduce a Compact Gated Linear Unit (CGLU) to selectively control channel interaction with minimal parameters. Furthermore, we introduce a Contextual Synergic Enhancement (CSE) learning strategy, which aims to boost the data efficiency of Transformers. It first leverages the extrinsic contextual information to support the learning of unlabeled data with Attention-Guided Replacement, then applies Spatial Masking Consistency that utilizes intrinsic contextual information to enhance the spatial context reasoning for unlabeled data. Extensive experiments on various benchmarks demonstrate the superiority of our approach in both performance and efficiency. For example, with only 10% labeled data on the Left Atrial Segmentation dataset, our method surpasses BCP by 1.43% Jaccard while drastically reducing the FLOPs by 90.8% and parameters by 85.8%. Code is released at https://github.com/CUHK-AIM-Group/Light-UNETR.

  </details>



- **ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment**  
  Yuzhi Chen, Ronghan Chen, Dongjie Huo, Yandan Yang, Dekang Qi, Haoyun Liu, Tong Lin, Shuang Zeng, Junjin Xiao, Xinyuan Chang, et al.  
  _2026-03-24_ · https://arxiv.org/abs/2603.23376v1  
  <details><summary>Abstract</summary>

  Video-based world models offer a powerful paradigm for embodied simulation and planning, yet state-of-the-art models often generate physically implausible manipulations - such as object penetration and anti-gravity motion - due to training on generic visual data and likelihood-based objectives that ignore physical laws. We present ABot-PhysWorld, a 14B Diffusion Transformer model that generates visually realistic, physically plausible, and action-controllable videos. Built on a curated dataset of three million manipulation clips with physics-aware annotation, it uses a novel DPO-based post-training framework with decoupled discriminators to suppress unphysical behaviors while preserving visual quality. A parallel context block enables precise spatial action injection for cross-embodiment control. To better evaluate generalization, we introduce EZSbench, the first training-independent embodied zero-shot benchmark combining real and synthetic unseen robot-task-scene combinations. It employs a decoupled protocol to separately assess physical realism and action alignment. ABot-PhysWorld achieves new state-of-the-art performance on PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and trajectory consistency. We will release EZSbench to promote standardized evaluation in embodied video generation.

  </details>



- **FHAvatar: Fast and High-Fidelity Reconstruction of Face-and-Hair Composable 3D Head Avatar from Few Casual Captures**  
  Yujie Sun, Zhuoqiang Cai, Chaoyue Niu, Jianchuan Chen, Zhiwen Chen, Chengfei Lv, Fan Wu  
  _2026-03-24_ · https://arxiv.org/abs/2603.23345v1  
  <details><summary>Abstract</summary>

  We present FHAvatar, a novel framework for reconstructing 3D Gaussian avatars with composable face and hair components from an arbitrary number of views. Unlike previous approaches that couple facial and hair representations within a unified modeling process, we explicitly decouple two components in texture space by representing the face with planar Gaussians and the hair with strand-based Gaussians. To overcome the limitations of existing methods that rely on dense multi-view captures or costly per-identity optimization, we propose an aggregated transformer backbone to learn geometry-aware cross-view priors and head-hair structural coherence from multi-view datasets, enabling effective and efficient feature extraction and fusion from few casual captures. Extensive quantitative and qualitative experiments demonstrate that FHAvatar achieves state-of-the-art reconstruction quality from only a few observations of new identities within minutes, while supporting real-time animation, convenient hairstyle transfer, and stylized editing, broadening the accessibility and applicability of digital avatar creation.

  </details>



- **An Explainable AI-Driven Framework for Automated Brain Tumor Segmentation Using an Attention-Enhanced U-Net**  
  MD Rashidul Islam, Bakary Gibba  
  _2026-03-24_ · https://arxiv.org/abs/2603.23344v1  
  <details><summary>Abstract</summary>

  Computer-aided segmentation of brain tumors from MRI data is of crucial significance to clinical decision-making in diagnosis, treatment planning, and follow-up disease monitoring. Gliomas, owing to their high malignancy and heterogeneity, represent a very challenging task for accurate and reliable segmentation into intra-tumoral sub-regions. Manual segmentation is typically time-consuming and not reliable, which justifies the need for robust automated techniques.This research resolves this problem by leveraging the BraTS 2020 dataset, where we have labeled MRI scans of glioma patients with four significant classes: background/healthy tissue, necrotic/non-enhancing core, edema, and enhancing tumor. In this work, we present a new segmentation technique based on a U-Net model augmented with executed attention gates to focus on the most significant regions of images. To counter class imbalance, we employ manually designed loss functions like Dice Loss and Categorical Dice Loss, in conjunction with standard categorical cross-entropy. Other evaluation metrics, like sensitivity and specificity, were used to measure discriminability of the model between tumor classes. Besides, we introduce Grad-CAM-based explainable AI to enable visualizing attention regions and improve model interpretability, together with a smooth heatmap generation technique through Gaussian filtering. Our approach achieved superior performance with accuracy of 0.9919, Dice coefficient of 0.9901, mean IoU of 0.9873, sensitivity of 0.9908, and specificity of 0.9974. This study demonstrates that the use of attention mechanisms, personalized loss functions, and explainable AI significantly improves highly complex tumor structure segmentation precision in MRI scans, providing a reliable and explainable method for clinical applications.

  </details>



- **ViBe: Ultra-High-Resolution Video Synthesis Born from Pure Images**  
  Yunfeng Wu, Hongying Cheng, Zihao He, Songhua Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.23326v1  
  <details><summary>Abstract</summary>

  Transformer-based video diffusion models rely on 3D attention over spatial and temporal tokens, which incurs quadratic time and memory complexity and makes end-to-end training for ultra-high-resolution videos prohibitively expensive. To overcome this bottleneck, we propose a pure image adaptation framework that upgrades a video Diffusion Transformer pre-trained at its native scale to synthesize higher-resolution videos. Unfortunately, naively fine-tuning with high-resolution images alone often introduces noticeable noise due to the image-video modality gap. To address this, we decouple the learning objective to separately handle modality alignment and spatial extrapolation. At the core of our approach is Relay LoRA, a two-stage adaptation strategy. In the first stage, the video diffusion model is adapted to the image domain using low-resolution images to bridge the modality gap. In the second stage, the model is further adapted with high-resolution images to acquire spatial extrapolation capability. During inference, only the high-resolution adaptation is retained to preserve the video generation modality while enabling high-resolution video synthesis. To enhance fine-grained detail synthesis, we further propose a High-Frequency-Awareness-Training-Objective, which explicitly encourages the model to recover high-frequency components from degraded latent representations via a dedicated reconstruction loss. Extensive experiments demonstrate that our method produces ultra-high-resolution videos with rich visual details without requiring any video training data, even outperforming previous state-of-the-art models trained on high-resolution videos by 0.8 on the VBench benchmark. Code will be available at https://github.com/WillWu111/ViBe.

  </details>



- **Curriculum-Driven 3D CT Report Generation via Language-Free Visual Grafting and Zone-Constrained Compression**  
  V. K. Cody Bumgardner, Mitchell A. Klusty, Mahmut S. Gokmen, Evan W. Damron  
  _2026-03-24_ · https://arxiv.org/abs/2603.23308v1  
  <details><summary>Abstract</summary>

  Automated radiology report generation from 3D computed tomography (CT) volumes is challenging due to extreme sequence lengths, severe class imbalance, and the tendency of large language models (LLMs) to ignore visual tokens in favor of linguistic priors. We present Ker-VLJEPA-3B, a four-phase curriculum learning framework for free-text report generation from thoracic CT volumes. A phased training curriculum progressively adapts a Llama 3.2 3B decoder to ground its output in visual features from a frozen, self-supervised encoder. Our visual backbone (LeJEPA ViT-Large) is trained via self-supervised joint-embedding prediction on unlabeled CTs, without text supervision. Unlike contrastive models (CLIP, BiomedCLIP), this language-free backbone yields modality-pure representations. Vision-language alignment is deferred to the curriculum's bridge and generation phases. This modality-agnostic design can integrate any self-supervised encoder into an LLM without paired text during foundation training. Methodological innovations include: (1) zone-constrained cross-attention compressing slice embeddings into 32 spatially-grounded visual tokens; (2) PCA whitening of anisotropic LLM embeddings; (3) a positive-findings-only strategy eliminating posterior collapse; (4) warm bridge initialization transferring projection weights; and (5) selective cross-attention freezing with elastic weight consolidation to prevent catastrophic forgetting. Evaluated on the CT-RATE benchmark (2,984 validation volumes, 18 classes), Ker-VLJEPA-3B achieves a macro F1 of 0.429, surpassing the state-of-the-art (U-VLM, macro F1 = 0.414) by 3.6%, and reaching 0.448 (+8.2%) with threshold optimization. Ablation studies confirm 56.6% of generation quality derives from patient-specific visual content. Code and weights are available.

  </details>



- **Mamba-driven MRI-to-CT Synthesis for MRI-only Radiotherapy Planning**  
  Konstantinos Barmpounakis, Theodoros P. Vagenas, Maria Vakalopoulou, George K. Matsopoulos  
  _2026-03-24_ · https://arxiv.org/abs/2603.23295v1  
  <details><summary>Abstract</summary>

  Radiotherapy workflows for oncological patients increasingly rely on multi-modal medical imaging, commonly involving both Magnetic Resonance Imaging (MRI) and Computed Tomography (CT). MRI-only treatment planning has emerged as an attractive alternative, as it reduces patient exposure to ionizing radiation and avoids errors introduced by inter-modality registration. While nnU-Net-based frameworks are predominantly used for MRI-to-CT synthesis, we explore Mamba-based architectures for this task, aiming to showcase the advantages of state-space modeling for cross-modality translation compared to standard convolutional neural networks. Specifically, we adapt both the U-Mamba and the SegMamba architecture, originally proposed for segmentation, to perform cross-modality image generation. Our 3D Mamba architecture effectively captures complex volumetric features and long-range dependencies, thus allowing accurate CT synthesis while maintaining fast inference times. Experiments were conducted on a subset of SynthRAD2025 dataset, comprising registered single-channel MRI-CT volume pairs across three anatomical regions. Quantitative evaluation is performed via a combination of image similarity metrics computed in Hounsefield Units (HU) and segmentation-based metrics obtained from TotalSegmentator to ensure geometric consistency is preserved. The findings pave the way for the integration of state-space models into radiotherapy workflows.

  </details>



- **Knot-10:A Tightness-Stratified Benchmark for Real-World Knot Classification with Topological Difficulty Analysis**  
  Shiheng Nie, Yunguang Yue  
  _2026-03-24_ · https://arxiv.org/abs/2603.23286v1  
  <details><summary>Abstract</summary>

  Physical knot classification is a fine-grained visual classification (FGVC) scenario in which appearance cues are deliberately suppressed: different classes share the same rope material, color, and background, and class identity resides primarily in crossing structure. We introduce the Knots-10 benchmark, comprising 1,440 images with a deployment-oriented split that trains on loosely tied knots and tests on tightly dressed ones. Swin-T and TransFG both average 97.2% accuracy; PMG scores 94.5%, consistent with the hypothesis that jigsaw shuffling disrupts crossing continuity. McNemar tests cannot separate four of the five general-purpose backbones, so small ranking margins should be interpreted with caution. A Mantel permutation test shows that topological distance significantly correlates with confusion patterns in three of the five models (p < 0.01). We propose TACA regularization, which improves embedding-topology alignment from rho=0.46 to rho=0.65 without improving classification accuracy; a random-distance ablation yields comparable alignment, indicating the benefit is likely driven by generic regularization. A pilot cross-domain test with 100 phone photographs reveals a 58-69 percentage-point accuracy drop, exposing rope appearance bias as the dominant failure mode.

  </details>



- **Multi-Modal Image Fusion via Intervention-Stable Feature Learning**  
  Xue Wang, Zheng Guan, Wenhua Qian, Chengchao Wang, Runzhuo Ma  
  _2026-03-24_ · https://arxiv.org/abs/2603.23272v1  
  <details><summary>Abstract</summary>

  Multi-modal image fusion integrates complementary information from different modalities into a unified representation. Current methods predominantly optimize statistical correlations between modalities, often capturing dataset-induced spurious associations that degrade under distribution shifts. In this paper, we propose an intervention-based framework inspired by causal principles to identify robust cross-modal dependencies. Drawing insights from Pearl's causal hierarchy, we design three principled intervention strategies to probe different aspects of modal relationships: i) complementary masking with spatially disjoint perturbations tests whether modalities can genuinely compensate for each other's missing information, ii) random masking of identical regions identifies feature subsets that remain informative under partial observability, and iii) modality dropout evaluates the irreplaceable contribution of each modality. Based on these interventions, we introduce a Causal Feature Integrator (CFI) that learns to identify and prioritize intervention-stable features maintaining importance across different perturbation patterns through adaptive invariance gating, thereby capturing robust modal dependencies rather than spurious correlations. Extensive experiments demonstrate that our method achieves SOTA performance on both public benchmarks and downstream high-level vision tasks.

  </details>



- **AeroScene: Progressive Scene Synthesis for Aerial Robotics**  
  Nghia Vu, Tuong Do, Dzung Tran, Binh X. Nguyen, Hoan Nguyen, Erman Tjiputra, Quang D. Tran, Hai-Nguyen Nguyen, Anh Nguyen  
  _2026-03-24_ · https://arxiv.org/abs/2603.23224v1  
  <details><summary>Abstract</summary>

  Generative models have shown substantial impact across multiple domains, their potential for scene synthesis remains underexplored in robotics. This gap is more evident in drone simulators, where simulation environments still rely heavily on manual efforts, which are time-consuming to create and difficult to scale. In this work, we introduce AeroScene, a hierarchical diffusion model for progressive 3D scene synthesis. Our approach leverages hierarchy-aware tokenization and multi-branch feature extraction to reason across both global layouts and local details, ensuring physical plausibility and semantic consistency. This makes AeroScene particularly suited for generating realistic scenes for aerial robotics tasks such as navigation, landing, and perching. We demonstrate its effectiveness through extensive experiments on our newly collected dataset and a public benchmark, showing that AeroScene significantly outperforms prior methods. Furthermore, we use AeroScene to generate a large-scale dataset of over 1,000 physics-ready, high fidelity 3D scenes that can be directly integrated into NVIDIA Isaac Sim. Finally, we illustrate the utility of these generated environments on downstream drone navigation tasks. Our code and dataset are publicly available at aioz-ai.github.io/AeroScene/

  </details>



- **PoseDriver: A Unified Approach to Multi-Category Skeleton Detection for Autonomous Driving**  
  Yasamin Borhani, Taylor Mordan, Yihan Wang, Reyhaneh Hosseininejad, Javad Khoramdel, Alexandre Alahi  
  _2026-03-24_ · https://arxiv.org/abs/2603.23215v1  
  <details><summary>Abstract</summary>

  Object skeletons offer a concise representation of structural information, capturing essential aspects of posture and orientation that are crucial for autonomous driving applications. However, a unified architecture that simultaneously handles multiple instances and categories using only the input image remains elusive. In this paper, we introduce PoseDriver, a unified framework for bottom-up multi-category skeleton detection tailored to common objects in driving scenarios. We model each category as a distinct task to systematically address the challenges of multi-task learning. Specifically, we propose a novel approach for lane detection based on skeleton representations, achieving state-of-the-art performance on the OpenLane dataset. Moreover, we present a new dataset for bicycle skeleton detection and assess the transferability of our framework to novel categories. Experimental results validate the effectiveness of the proposed approach.

  </details>



- **FDIF: Formula-Driven supervised Learning with Implicit Functions for 3D Medical Image Segmentation**  
  Yukinori Yamamoto, Kazuya Nishimura, Tsukasa Fukusato, Hirokazu Nosato, Tetsuya Ogata, Hirokatsu Kataoka  
  _2026-03-24_ · https://arxiv.org/abs/2603.23199v1  
  <details><summary>Abstract</summary>

  Deep learning-based 3D medical image segmentation methods relies on large-scale labeled datasets, yet acquiring such data is difficult due to privacy constraints and the high cost of expert annotation. Formula-Driven Supervised Learning (FDSL) offers an appealing alternative by generating training data and labels directly from mathematical formulas. However, existing voxel-based approaches are limited in geometric expressiveness and cannot synthesize realistic textures. We introduce Formula-Driven supervised learning with Implicit Functions (FDIF), a framework that enables scalable pre-training without using any real data and medical expert annotations. FDIF introduces an implicit-function representation based on signed distance functions (SDFs), enabling compact modeling of complex geometries while exploiting the surface representation of SDFs to support controllable synthesis of both geometric and intensity textures. Across three medical image segmentation benchmarks (AMOS, ACDC, and KiTS) and three architectures (SwinUNETR, nnUNet ResEnc-L, and nnUNet Primus-M), FDIF consistently improves over a formula-driven method, and achieves performance comparable to self-supervised approaches pre-trained on large-scale real datasets. We further show that FDIF pre-training also benefits 3D classification tasks, highlighting implicit-function-based formula supervision as a promising paradigm for data-free representation learning. Code is available at https://github.com/yamanoko/FDIF.

  </details>



- **Gimbal360: Differentiable Auto-Leveling for Canonicalized $360^\circ$ Panoramic Image Completion**  
  Yuqin Lu, Haofeng Liu, Yang Zhou, Jun Liang, Shengfeng He, Jing Li  
  _2026-03-24_ · https://arxiv.org/abs/2603.23179v1  
  <details><summary>Abstract</summary>

  Diffusion models excel at 2D outpainting, but extending them to $360^\circ$ panoramic completion from unposed perspective images is challenging due to the geometric and topological mismatch between perspective projections and spherical panoramas. We present Gimbal360, a principled framework that explicitly bridges perspective observations and spherical panoramas. We introduce a Canonical Viewing Space that regularizes projective geometry and provides a consistent intermediate representation between the two domains. To anchor in-the-wild inputs to this space, we propose a Differentiable Auto-Leveling module that stabilizes feature orientation without requiring camera parameters at inference. Panoramic generation also introduces a topological challenge. Standard generative architectures assume a bounded Euclidean image plane, while Equirectangular Projection (ERP) panoramas exhibit intrinsic $S^1$ periodicity. Euclidean operations therefore break boundary continuity. We address this mismatch by enforcing topological equivariance in the latent space to preserve seamless periodic structure. To support this formulation, we introduce Horizon360, a curated large-scale dataset of gravity-aligned panoramic environments. Extensive experiments show that explicitly standardizing geometric and topological priors enables Gimbal360 to achieve state-of-the-art performance in structurally consistent $360^\circ$ scene completion.

  </details>



- **LiZIP: An Auto-Regressive Compression Framework for LiDAR Point Clouds**  
  Aditya Shibu, Kayvan Karim, Claudio Zito  
  _2026-03-24_ · https://arxiv.org/abs/2603.23162v1  
  <details><summary>Abstract</summary>

  The massive volume of data generated by LiDAR sensors in autonomous vehicles creates a bottleneck for real-time processing and vehicle-to-everything (V2X) transmission. Existing lossless compression methods often force a trade-off: industry standard algorithms (e.g., LASzip) lack adaptability, while deep learning approaches suffer from prohibitive computational costs. This paper proposes LiZIP, a lightweight, near-lossless zero-drift compression framework based on neural predictive coding. By utilizing a compact Multi-Layer Perceptron (MLP) to predict point coordinates from local context, LiZIP efficiently encodes only the sparse residuals. We evaluate LiZIP on the NuScenes and Argoverse datasets, benchmarking against GZip, LASzip, and Google Draco (configured with 24-bit quantization to serve as a high-precision geometric baseline). Results demonstrate that LiZIP consistently achieves superior compression ratios across varying environments. The proposed system achieves a 7.5%-14.8% reduction in file size compared to the industry-standard LASzip and outperforms Google Draco by 8.8%-11.3% across diverse datasets. Furthermore, the system demonstrates generalization capabilities on the unseen Argoverse dataset without retraining. Against the general purpose GZip algorithm, LiZIP achieves a reduction of 38%-48%. This efficiency offers a distinct advantage for bandwidth constrained V2X applications and large scale cloud archival.

  </details>



- **Dual Contrastive Network for Few-Shot Remote Sensing Image Scene Classification**  
  Zhong Ji, Liyuan Hou, Xuan Wang, Gang Wang, Yanwei Pang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23161v1  
  <details><summary>Abstract</summary>

  Few-shot remote sensing image scene classification (FS-RSISC) aims at classifying remote sensing images with only a few labeled samples. The main challenges lie in small inter-class variances and large intra-class variances, which are the inherent property of remote sensing images. To address these challenges, we propose a transfer-based Dual Contrastive Network (DCN), which incorporates two auxiliary supervised contrastive learning branches during the training process. Specifically, one is a Context-guided Contrastive Learning (CCL) branch and the other is a Detail-guided Contrastive Learning (DCL) branch, which focus on inter-class discriminability and intra-class invariance, respectively. In the CCL branch, we first devise a Condenser Network to capture context features, and then leverage a supervised contrastive learning on top of the obtained context features to facilitate the model to learn more discriminative features. In the DCL branch, a Smelter Network is designed to highlight the significant local detail information. And then we construct a supervised contrastive learning based on the detail feature maps to fully exploit the spatial information in each map, enabling the model to concentrate on invariant detail features. Extensive experiments on four public benchmark remote sensing datasets demonstrate the competitive performance of our proposed DCN.

  </details>



- **Conformal Cross-Modal Active Learning**  
  Huy Hoang Nguyen, Cédric Jung, Shirin Salehi, Tobias Glück, Anke Schmeink, Andreas Kugi  
  _2026-03-24_ · https://arxiv.org/abs/2603.23159v1  
  <details><summary>Abstract</summary>

  Foundation models for vision have transformed visual recognition with powerful pretrained representations and strong zero-shot capabilities, yet their potential for data-efficient learning remains largely untapped. Active Learning (AL) aims to minimize annotation costs by strategically selecting the most informative samples for labeling, but existing methods largely overlook the rich multimodal knowledge embedded in modern vision-language models (VLMs). We introduce Conformal Cross-Modal Acquisition (CCMA), a novel AL framework that bridges vision and language modalities through a teacher-student architecture. CCMA employs a pretrained VLM as a teacher to provide semantically grounded uncertainty estimates, conformally calibrated to guide sample selection for a vision-only student model. By integrating multimodal conformal scoring with diversity-aware selection strategies, CCMA achieves superior data efficiency across multiple benchmarks. Our approach consistently outperforms state-of-the-art AL baselines, demonstrating clear advantages over methods relying solely on uncertainty or diversity metrics.

  </details>



- **VoDaSuRe: A Large-Scale Dataset Revealing Domain Shift in Volumetric Super-Resolution**  
  August Leander Høeg, Sophia Wiinberg Bardenfleth, Hans Martin Kjer, Tim Bjørn Dyrby, Vedrana Andersen Dahl, Anders Bjorholm Dahl  
  _2026-03-24_ · https://arxiv.org/abs/2603.23153v1  
  <details><summary>Abstract</summary>

  Recent advances in volumetric super-resolution (SR) have demonstrated strong performance in medical and scientific imaging, with transformer- and CNN-based approaches achieving impressive results even at extreme scaling factors. In this work, we show that much of this performance stems from training on downsampled data rather than real low-resolution scans. This reliance on downsampling is partly driven by the scarcity of paired high- and low-resolution 3D datasets. To address this, we introduce VoDaSuRe, a large-scale volumetric dataset containing paired high- and low-resolution scans. When training models on VoDaSuRe, we reveal a significant discrepancy: SR models trained on downsampled data produce substantially sharper predictions than those trained on real low-resolution scans, which smooth fine structures. Conversely, applying models trained on downsampled data to real scans preserves more structure but is inaccurate. Our findings suggest that current SR methods are overstated - when applied to real data, they do not recover structures lost in low-resolution scans and instead predict a smoothed average. We argue that progress in deep learning-based volumetric SR requires datasets with paired real scans of high complexity, such as VoDaSuRe. Our dataset and code are publicly available through: https://augusthoeg.github.io/VoDaSuRe/

  </details>



- **PiCo: Active Manifold Canonicalization for Robust Robotic Visual Anomaly Detection**  
  Teng Yan, Binkai Liu, Shuai Liu, Yue Yu, Bingzhuo Zhong  
  _2026-03-24_ · https://arxiv.org/abs/2603.23122v1  
  <details><summary>Abstract</summary>

  Industrial deployment of robotic visual anomaly detection (VAD) is fundamentally constrained by passive perception under diverse 6-DoF pose configurations and unstable operating conditions such as illumination changes and shadows, where intrinsic semantic anomalies and physical disturbances coexist and interact. To overcome these limitations, a paradigm shift from passive feature learning to Active Canonicalization is proposed. PiCo (Pose-in-Condition Canonicalization) is introduced as a unified framework that actively projects observations onto a condition-invariant canonical manifold. PiCo operates through a cascaded mechanism. The first stage, Active Physical Canonicalization, enables a robotic agent to reorient objects in order to reduce geometric uncertainty at its source. The second stage, Neural Latent Canonicalization, adopts a three-stage denoising hierarchy consisting of photometric processing at the input level, latent refinement at the feature level, and contextual reasoning at the semantic level, progressively eliminating nuisance factors across representational scales. Extensive evaluations on the large-scale M2AD benchmark demonstrate the superiority of this paradigm. PiCo achieves a state-of-the-art 93.7% O-AUROC, representing a 3.7% improvement over prior methods in static settings, and attains 98.5% accuracy in active closed-loop scenarios. These results demonstrate that active manifold canonicalization is critical for robust embodied perception.

  </details>



- **SMSP: A Plug-and-Play Strategy of Multi-Scale Perception for MLLMs to Perceive Visual Illusions**  
  Jinzhe Tu, Ruilei Guo, Zihan Guo, Junxiao Yang, Shiyao Cui, Minlie Huang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23118v1  
  <details><summary>Abstract</summary>

  Recent works have shown that Multimodal Large Language Models (MLLMs) are highly vulnerable to hidden-pattern visual illusions, where the hidden content is imperceptible to models but obvious to humans. This deficiency highlights a perceptual misalignment between current MLLMs and humans, and also introduces potential safety concerns. To systematically investigate this failure, we introduce IlluChar, a comprehensive and challenging illusion dataset, and uncover a key underlying mechanism for the models' failure: high-frequency attention bias, where the models are easily distracted by high-frequency background textures in illusion images, causing them to overlook hidden patterns. To address the issue, we propose the Strategy of Multi-Scale Perception (SMSP), a plug-and-play framework that aligns with human visual perceptual strategies. By suppressing distracting high-frequency backgrounds, SMSP generates images closer to human perception. Our experiments demonstrate that SMSP significantly improves the performance of all evaluated MLLMs on illusion images, for instance, increasing the accuracy of Qwen3-VL-8B-Instruct from 13.0% to 84.0%. Our work provides novel insights into MLLMs' visual perception, and offers a practical and robust solution to enhance it. Our code is publicly available at https://github.com/Tujz2023/SMSP.

  </details>



- **Automatic Segmentation of 3D CT scans with SAM2 using a zero-shot approach**  
  Miquel Lopez Escoriza, Pau Amargant Alvarez  
  _2026-03-24_ · https://arxiv.org/abs/2603.23116v1  
  <details><summary>Abstract</summary>

  Foundation models for image segmentation have shown strong generalization in natural images, yet their applicability to 3D medical imaging remains limited. In this work, we study the zero-shot use of Segment Anything Model 2 (SAM2) for automatic segmentation of volumetric CT data, without any fine-tuning or domain-specific training. We analyze how SAM2 should be applied to CT volumes and identify its main limitation: the lack of inherent volumetric awareness. To address this, we propose a set of inference-alone architectural and procedural modifications that adapt SAM2's video-based memory mechanism to 3D data by treating CT slices as ordered sequences. We conduct a systematic ablation study on a subset of 500 CT scans from the TotalSegmentator dataset to evaluate prompt strategies, memory propagation schemes and multi-pass refinement. Based on these findings, we select the best-performing configuration and report final results on a bigger sample of the TotalSegmentator dataset comprising 2,500 CT scans. Our results show that, even with frozen weights, SAM2 can produce coherent 3D segmentations when its inference pipeline is carefully structured, demonstrating the feasibility of a fully zero-shot approach for volumetric medical image segmentation.

  </details>



- **Traffic Sign Recognition in Autonomous Driving: Dataset, Benchmark, and Field Experiment**  
  Guoyang Zhao, Weiqing Qi, Kai Zhang, Chenguang Zhang, Zeying Gong, Zhihai Bi, Kai Chen, Benshan Ma, Ming Liu, Jun Ma  
  _2026-03-24_ · https://arxiv.org/abs/2603.23034v1  
  <details><summary>Abstract</summary>

  Traffic Sign Recognition (TSR) is a core perception capability for autonomous driving, where robustness to cross-region variation, long-tailed categories, and semantic ambiguity is essential for reliable real-world deployment. Despite steady progress in recognition accuracy, existing traffic sign datasets and benchmarks offer limited diagnostic insight into how different modeling paradigms behave under these practical challenges. We present TS-1M, a large-scale and globally diverse traffic sign dataset comprising over one million real-world images across 454 standardized categories, together with a diagnostic benchmark designed to analyze model capability boundaries. Beyond standard train-test evaluation, we provide a suite of challenge-oriented settings, including cross-region recognition, rare-class identification, low-clarity robustness, and semantic text understanding, enabling systematic and fine-grained assessment of modern TSR models. Using TS-1M, we conduct a unified benchmark across three representative learning paradigms: classical supervised models, self-supervised pretrained models, and multimodal vision-language models (VLMs). Our analysis reveals consistent paradigm-dependent behaviors, showing that semantic alignment is a key factor for cross-region generalization and rare-category recognition, while purely visual models remain sensitive to appearance shift and data imbalance. Finally, we validate the practical relevance of TS-1M through real-scene autonomous driving experiments, where traffic sign recognition is integrated with semantic reasoning and spatial localization to support map-level decision constraints. Overall, TS-1M establishes a reference-level diagnostic benchmark for TSR and provides principled insights into robust and semantic-aware traffic sign perception. Project page: https://guoyangzhao.github.io/projects/ts1m.

  </details>



- **Concept-based explanations of Segmentation and Detection models in Natural Disaster Management**  
  Samar Heydari, Jawher Said, Galip Ümit Yolcu, Evgenii Kortukov, Elena Golimblevskaia, Evgenios Vlachos, Vasileios Mygdalis, Ioannis Pitas, Sebastian Lapuschkin, Leila Arras  
  _2026-03-24_ · https://arxiv.org/abs/2603.23020v1  
  <details><summary>Abstract</summary>

  Deep learning models for flood and wildfire segmentation and object detection enable precise, real-time disaster localization when deployed on embedded drone platforms. However, in natural disaster management, the lack of transparency in their decision-making process hinders human trust required for emergency response. To address this, we present an explainability framework for understanding flood segmentation and car detection predictions on the widely used PIDNet and YOLO architectures. More specifically, we introduce a novel redistribution strategy that extends Layer-wise Relevance Propagation (LRP) explanations for sigmoid-gated element-wise fusion layers. This extension allows LRP relevances to flow through the fusion modules of PIDNet, covering the entire computation graph back to the input image. Furthermore, we apply Prototypical Concept-based Explanations (PCX) to provide both local and global explanations at the concept level, revealing which learned features drive the segmentation and detection of specific disaster semantic classes. Experiments on a publicly available flood dataset show that our framework provides reliable and interpretable explanations while maintaining near real-time inference capabilities, rendering it suitable for deployment on resource-constrained platforms, such as Unmanned Aerial Vehicles (UAVs).

  </details>



- **VQ-Jarvis: Retrieval-Augmented Video Restoration Agent with Sharp Vision and Fast Thought**  
  Xuanyu Zhang, Weiqi Li, Qunliang Xing, Jingfen Xie, Bin Chen, Junlin Li, Li Zhang, Jian Zhang, Shijie Zhao  
  _2026-03-24_ · https://arxiv.org/abs/2603.22998v1  
  <details><summary>Abstract</summary>

  Video restoration in real-world scenarios is challenged by heterogeneous degradations, where static architectures and fixed inference pipelines often fail to generalize. Recent agent-based approaches offer dynamic decision making, yet existing video restoration agents remain limited by insufficient quality perception and inefficient search strategies. We propose VQ-Jarvis, a retrieval-augmented, all-in-one intelligent video restoration agent with sharper vision and faster thought. VQ-Jarvis is designed to accurately perceive degradations and subtle differences among paired restoration results, while efficiently discovering optimal restoration trajectories. To enable sharp vision, we construct VSR-Compare, the first large-scale video paired enhancement dataset with 20K comparison pairs covering 7 degradation types, 11 enhancement operators, and diverse content domains. Based on this dataset, we train a multiple operator judge model and a degradation perception model to guide agent decisions. To achieve fast thought, we introduce a hierarchical operator scheduling strategy that adapts to video difficulty: for easy cases, optimal restoration trajectories are retrieved in a one-step manner from a retrieval-augmented generation (RAG) library; for harder cases, a step-by-step greedy search is performed to balance efficiency and accuracy. Extensive experiments demonstrate that VQ-Jarvis consistently outperforms existing methods on complex degraded videos.

  </details>



- **VLA-IAP: Training-Free Visual Token Pruning via Interaction Alignment for Vision-Language-Action Models**  
  Jintao Cheng, Haozhe Wang, Weibin Li, Gang Wang, Yipu Zhang, Xiaoyu Tang, Jin Wu, Xieyuanli Chen, Yunhui Liu, Wei Zhang  
  _2026-03-24_ · https://arxiv.org/abs/2603.22991v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have rapidly advanced embodied intelligence, enabling robots to execute complex, instruction-driven tasks. However, as model capacity and visual context length grow, the inference cost of VLA systems becomes a major bottleneck for real-world deployment on resource-constrained platforms. Existing visual token pruning methods mainly rely on semantic saliency or simple temporal cues, overlooking the continuous physical interaction, a fundamental property of VLA tasks. Consequently, current approaches often prune visually sparse yet structurally critical regions that support manipulation, leading to unstable behavior during early task phases. To overcome this, we propose a shift toward an explicit Interaction-First paradigm. Our proposed \textbf{training-free} method, VLA-IAP (Interaction-Aligned Pruning), introduces a geometric prior mechanism to preserve structural anchors and a dynamic scheduling strategy that adapts pruning intensity based on semantic-motion alignment. This enables a conservative-to-aggressive transition, ensuring robustness during early uncertainty and efficiency once interaction is locked. Extensive experiments show that VLA-IAP achieves a \textbf{97.8\% success rate} with a \textbf{$1.25\times$ speedup} on the LIBERO benchmark, and up to \textbf{$1.54\times$ speedup} while maintaining performance \textbf{comparable to the unpruned backbone}. Moreover, the method demonstrates superior and consistent performance across multiple model architectures and three different simulation environments, as well as a real robot platform, validating its strong generalization capability and practical applicability. Our project website is: \href{https://chengjt1999.github.io/VLA-IAP.github.io/}{VLA-IAP.com}.

  </details>



- **Caption Generation for Dongba Paintings via Prompt Learning and Semantic Fusion**  
  Shuangwu Qian, Xiaochan Yuan, Pengfei Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22946v1  
  <details><summary>Abstract</summary>

  Dongba paintings, the treasured pictorial legacy of the Naxi people in southwestern China, feature richly layered visual elements, vivid color palettes, and pronounced ethnic and regional cultural symbolism, yet their automatic textual description remains largely unexplored owing to severe domain shift when mainstream captioning models are applied directly. This paper proposes \textbf{PVGF-DPC} (\textit{Prompt and Visual Semantic-Generation Fusion-based Dongba Painting Captioning}), an encoder-decoder framework that integrates a content prompt module with a novel visual semantic-generation fusion loss to bridge the gap between generic natural-image captioning and the culturally specific imagery found in Dongba art. A MobileNetV2 encoder extracts discriminative visual features, which are injected into the layer normalization of a 10-layer Transformer decoder initialized with pretrained BERT weights; meanwhile, the content prompt module maps the image feature vector to culture-aware labels -- such as \emph{deity}, \emph{ritual pattern}, or \emph{hell ghost} -- and constructs a post-prompt that steers the decoder toward thematically accurate descriptions. The visual semantic-generation fusion loss jointly optimizes the cross-entropy objectives of both the prompt predictor and the caption generator, encouraging the model to extract key cultural and visual cues and to produce captions that are semantically aligned with the input image. We construct a dedicated Dongba painting captioning dataset comprising 9{}408 augmented images with culturally grounded annotations spanning seven thematic categories.

  </details>



- **FixationFormer: Direct Utilization of Expert Gaze Trajectories for Chest X-Ray Classification**  
  Daniel Beckmann, Benjamin Risse  
  _2026-03-24_ · https://arxiv.org/abs/2603.22939v1  
  <details><summary>Abstract</summary>

  Expert eye movements provide a rich, passive source of domain knowledge in radiology, offering a powerful cue for integrating diagnostic reasoning into computer-aided analysis. However, direct integration into CNN-based systems, which historically have dominated the medical image analysis domain, is challenging: gaze recordings are sequential, temporally dense yet spatially sparse, noisy, and variable across experts. As a consequence, most existing image-based models utilize reduced representations such as heatmaps. In contrast, gaze naturally aligns with transformer architectures, as both are sequential in nature and rely on attention to highlight relevant input regions. In this work, we introduce FixationFormer, a transformer-based architecture that represents expert gaze trajectories as sequences of tokens, thereby preserving their temporal and spatial structure. By modeling gaze sequences jointly with image features, our approach addresses sparsity and variability in gaze data while enabling a more direct and fine-grained integration of expert diagnostic cues through explicit cross-attention between the image and gaze token sequences. We evaluate our method on three publicly available benchmark chest X-ray datasets and demonstrate that it achieves state-of-the-art classification performance, highlighting the value of representing gaze as a sequence in transformer-based medical image analysis.

  </details>



- **When AVSR Meets Video Conferencing: Dataset, Degradation, and the Hidden Mechanism Behind Performance Collapse**  
  Yihuan Huang, Jun Xue, Liu Jiajun, Daixian Li, Tong Zhang, Zhuolin Yi, Yanzhen Ren, Kai Li  
  _2026-03-24_ · https://arxiv.org/abs/2603.22915v1  
  <details><summary>Abstract</summary>

  Audio-Visual Speech Recognition (AVSR) has achieved remarkable progress in offline conditions, yet its robustness in real-world video conferencing (VC) remains largely unexplored. This paper presents the first systematic evaluation of state-of-the-art AVSR models across mainstream VC platforms, revealing severe performance degradation caused by transmission distortions and spontaneous human hyper-expression. To address this gap, we construct \textbf{MLD-VC}, the first multimodal dataset tailored for VC, comprising 31 speakers, 22.79 hours of audio-visual data, and explicit use of the Lombard effect to enhance human hyper-expression. Through comprehensive analysis, we find that speech enhancement algorithms are the primary source of distribution shift, which alters the first and second formants of audio. Interestingly, we find that the distribution shift induced by the Lombard effect closely resembles that introduced by speech enhancement, which explains why models trained on Lombard data exhibit greater robustness in VC. Fine-tuning AVSR models on MLD-VC mitigates this issue, achieving an average 17.5% reduction in CER across several VC platforms. Our findings and dataset provide a foundation for developing more robust and generalizable AVSR systems in real-world video conferencing. MLD-VC is available at https://huggingface.co/datasets/nccm2p2/MLD-VC.

  </details>



- **Dual-Teacher Distillation with Subnetwork Rectification for Black-Box Domain Adaptation**  
  Zhe Zhang, Jing Li, Wanli Xue, Xu Cheng, Jianhua Zhang, Qinghua Hu, Shengyong Chen  
  _2026-03-24_ · https://arxiv.org/abs/2603.22908v1  
  <details><summary>Abstract</summary>

  Assuming that neither source data nor the source model is accessible, black box domain adaptation represents a highly practical yet extremely challenging setting, as transferable information is restricted to the predictions of the black box source model, which can only be queried using target samples. Existing approaches attempt to extract transferable knowledge through pseudo label refinement or by leveraging external vision language models (ViLs), but they often suffer from noisy supervision or insufficient utilization of the semantic priors provided by ViLs, which ultimately hinder adaptation performance. To overcome these limitations, we propose a dual teacher distillation with subnetwork rectification (DDSR) model that jointly exploits the specific knowledge embedded in black box source models and the general semantic information of a ViL. DDSR adaptively integrates their complementary predictions to generate reliable pseudo labels for the target domain and introduces a subnetwork driven regularization strategy to mitigate overfitting caused by noisy supervision. Furthermore, the refined target predictions iteratively enhance both the pseudo labels and ViL prompts, enabling more accurate and semantically consistent adaptation. Finally, the target model is further optimized through self training with classwise prototypes. Extensive experiments on multiple benchmark datasets validate the effectiveness of our approach, demonstrating consistent improvements over state of the art methods, including those using source data or models.

  </details>



- **Group Editing : Edit Multiple Images in One Go**  
  Yue Ma, Xinyu Wang, Qianli Ma, Qinghe Wang, Mingzhe Zheng, Xiangpeng Yang, Hao Li, Chongbo Zhao, Jixuan Ying, Harry Yang, et al.  
  _2026-03-24_ · https://arxiv.org/abs/2603.22883v1  
  <details><summary>Abstract</summary>

  In this paper, we tackle the problem of performing consistent and unified modifications across a set of related images. This task is particularly challenging because these images may vary significantly in pose, viewpoint, and spatial layout. Achieving coherent edits requires establishing reliable correspondences across the images, so that modifications can be applied accurately to semantically aligned regions. To address this, we propose GroupEditing, a novel framework that builds both explicit and implicit relationships among images within a group. On the explicit side, we extract geometric correspondences using VGGT, which provides spatial alignment based on visual features. On the implicit side, we reformulate the image group as a pseudo-video and leverage the temporal coherence priors learned by pre-trained video models to capture latent relationships. To effectively fuse these two types of correspondences, we inject the explicit geometric cues from VGGT into the video model through a novel fusion mechanism. To support large-scale training, we construct GroupEditData, a new dataset containing high-quality masks and detailed captions for numerous image groups. Furthermore, to ensure identity preservation during editing, we introduce an alignment-enhanced RoPE module, which improves the model's ability to maintain consistent appearance across multiple images. Finally, we present GroupEditBench, a dedicated benchmark designed to evaluate the effectiveness of group-level image editing. Extensive experiments demonstrate that GroupEditing significantly outperforms existing methods in terms of visual quality, cross-view consistency, and semantic alignment.

  </details>



- **Grounding Sim-to-Real Generalization in Dexterous Manipulation: An Empirical Study with Vision-Language-Action Models**  
  Ruixing Jin, Zicheng Zhu, Ruixiang Ouyang, Sheng Xu, Bo Yue, Zhizheng Wu, Guiliang Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22876v1  
  <details><summary>Abstract</summary>

  Learning a generalist control policy for dexterous manipulation typically relies on large-scale datasets. Given the high cost of real-world data collection, a practical alternative is to generate synthetic data through simulation. However, the resulting synthetic data often exhibits a significant gap from real-world distributions. While many prior studies have proposed algorithms to bridge the Sim-to-Real discrepancy, there remains a lack of principled research that grounds these methods in real-world manipulation tasks, particularly their performance on generalist policies such as Vision-Language-Action (VLA) models. In this study, we empirically examine the primary determinants of Sim-to-Real generalization across four dimensions: multi-level domain randomization, photorealistic rendering, physics-realistic modeling, and reinforcement learning updates. To support this study, we design a comprehensive evaluation protocol to quantify the real-world performance of manipulation tasks. The protocol accounts for key variations in background, lighting, distractors, object types, and spatial features. Through experiments involving over 10k real-world trials, we derive critical insights into Sim-to-Real transfer. To inform and advance future studies, we release both the robotic platforms and the evaluation protocol for public access to facilitate independent verification, thereby establishing a realistic and standardized benchmark for dexterous manipulation policies.

  </details>



- **ForeSea: AI Forensic Search with Multi-modal Queries for Video Surveillance**  
  Hyojin Park, Yi Li, Janghoon Cho, Sungha Choi, Jungsoo Lee, Taotao Jing, Shuai Zhang, Munawar Hayat, Dashan Gao, Ning Bi, et al.  
  _2026-03-24_ · https://arxiv.org/abs/2603.22872v1  
  <details><summary>Abstract</summary>

  Despite decades of work, surveillance still struggles to find specific targets across long, multi-camera video. Prior methods -- tracking pipelines, CLIP based models, and VideoRAG -- require heavy manual filtering, capture only shallow attributes, and fail at temporal reasoning. Real-world searches are inherently multimodal (e.g., "When does this person join the fight?" with the person's image), yet this setting remains underexplored. Also, there are no proper benchmarks to evaluate those setting - asking video with multimodal queries. To address this gap, we introduce ForeSeaQA, a new benchmark specifically designed for video QA with image-and-text queries and timestamped annotations of key events. The dataset consists of long-horizon surveillance footage paired with diverse multimodal questions, enabling systematic evaluation of retrieval, temporal grounding, and multimodal reasoning in realistic forensic conditions. Not limited to this benchmark, we propose ForeSea, an AI forensic search system with a 3-stage, plug-and-play pipeline. (1) A tracking module filters irrelevant footage; (2) a multimodal embedding module indexes the remaining clips; and (3) during inference, the system retrieves top-K candidate clips for a Video Large Language Model (VideoLLM) to answer queries and localize events. On ForeSeaQA, ForeSea improves accuracy by 3.5% and temporal IoU by 11.0 over prior VideoRAG models. To our knowledge, ForeSeaQA is the first benchmark to support complex multimodal queries with precise temporal grounding, and ForeSea is the first VideoRAG system built to excel in this setting.

  </details>



- **UAV-DETR: DETR for Anti-Drone Target Detection**  
  Jun Yang, Dong Wang, Hongxu Yin, Hongpeng Li, Jianxiong Yu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22841v1  
  <details><summary>Abstract</summary>

  Drone detection is pivotal in numerous security and counter-UAV applications. However, existing deep learning-based methods typically struggle to balance robust feature representation with computational efficiency. This challenge is particularly acute when detecting miniature drones against complex backgrounds under severe environmental interference. To address these issues, we introduce UAV-DETR, a novel framework that integrates a small-target-friendly architecture with real-time detection capabilities. Specifically, UAV-DETR features a WTConv-enhanced backbone and a Sliding Window Self-Attention (SWSA-IFI) encoder, capturing the high-frequency structural details of tiny targets while drastically reducing parameter overhead. Furthermore, we propose an Efficient Cross-Scale Feature Recalibration and Fusion Network (ECFRFN) to suppress background noise and aggregate multi-scale semantics. To further enhance accuracy, UAV-DETR incorporates a hybrid Inner-CIoU and NWD loss strategy, mitigating the extreme sensitivity of standard IoU metrics to minor positional deviations in small objects. Extensive experiments demonstrate that UAV-DETR significantly outperforms the baseline RT-DETR on our custom UAV dataset (+6.61% in mAP50:95, with a 39.8% reduction in parameters) and the public DUT-ANTI-UAV benchmark (+1.4% in Precision, +1.0% in F1-Score). These results establish UAV-DETR as a superior trade-off between efficiency and precision in counter-UAV object detection. The code is available at https://github.com/wd-sir/UAVDETR.

  </details>



- **URA-Net: Uncertainty-Integrated Anomaly Perception and Restoration Attention Network for Unsupervised Anomaly Detection**  
  Wei Luo, Peng Xing, Yunkang Cao, Haiming Yao, Weiming Shen, Zechao Li  
  _2026-03-24_ · https://arxiv.org/abs/2603.22840v1  
  <details><summary>Abstract</summary>

  Unsupervised anomaly detection plays a pivotal role in industrial defect inspection and medical image analysis, with most methods relying on the reconstruction framework. However, these methods may suffer from over-generalization, enabling them to reconstruct anomalies well, which leads to poor detection performance. To address this issue, instead of focusing solely on normality reconstruction, we propose an innovative Uncertainty-Integrated Anomaly Perception and Restoration Attention Network (URA-Net), which explicitly restores abnormal patterns to their corresponding normality. First, unlike traditional image reconstruction methods, we utilize a pre-trained convolutional neural network to extract multi-level semantic features as the reconstruction target. To assist the URA-Net learning to restore anomalies, we introduce a novel feature-level artificial anomaly synthesis module to generate anomalous samples for training. Subsequently, a novel uncertainty-integrated anomaly perception module based on Bayesian neural networks is introduced to learn the distributions of anomalous and normal features. This facilitates the estimation of anomalous regions and ambiguous boundaries, laying the foundation for subsequent anomaly restoration. Then, we propose a novel restoration attention mechanism that leverages global normal semantic information to restore detected anomalous regions, thereby obtaining defect-free restored features. Finally, we employ residual maps between input features and restored features for anomaly detection and localization. The comprehensive experimental results on two industrial datasets, MVTec AD and BTAD, along with a medical image dataset, OCT-2017, unequivocally demonstrate the effectiveness and superiority of the proposed method.

  </details>



- **MultiCam: On-the-fly Multi-Camera Pose Estimation Using Spatiotemporal Overlaps of Known Objects**  
  Shiyu Li, Hannah Schieber, Kristoffer Waldow, Benjamin Busam, Julian Kreimeier, Daniel Roth  
  _2026-03-24_ · https://arxiv.org/abs/2603.22839v1  
  <details><summary>Abstract</summary>

  Multi-camera dynamic Augmented Reality (AR) applications require a camera pose estimation to leverage individual information from each camera in one common system. This can be achieved by combining contextual information, such as markers or objects, across multiple views. While commonly cameras are calibrated in an initial step or updated through the constant use of markers, another option is to leverage information already present in the scene, like known objects. Another downside of marker-based tracking is that markers have to be tracked inside the field-of-view (FoV) of the cameras. To overcome these limitations, we propose a constant dynamic camera pose estimation leveraging spatiotemporal FoV overlaps of known objects on the fly. To achieve that, we enhance the state-of-the-art object pose estimator to update our spatiotemporal scene graph, enabling a relation even among non-overlapping FoV cameras. To evaluate our approach, we introduce a multi-camera, multi-object pose estimation dataset with temporal FoV overlap, including static and dynamic cameras. Furthermore, in FoV overlapping scenarios, we outperform the state-of-the-art on the widely used YCB-V and T-LESS dataset in camera pose accuracy. Our performance on both previous and our proposed datasets validates the effectiveness of our marker-less approach for AR applications. The code and dataset are available on https://github.com/roth-hex-lab/IEEE-VR-2026-MultiCam.

  </details>



- **MVRD-Bench: Multi-View Learning and Benchmarking for Dynamic Remote Photoplethysmography under Occlusion**  
  Zuxian He, Xu Cheng, Zhaodong Sun, Haoyu Chen, Jingang Shi, Xiaobai Li, Guoying Zhao  
  _2026-03-24_ · https://arxiv.org/abs/2603.22826v1  
  <details><summary>Abstract</summary>

  Remote photoplethysmography (rPPG) is a non-contact technique that estimates physiological signals by analyzing subtle skin color changes in facial videos. Existing rPPG methods often encounter performance degradation under facial motion and occlusion scenarios due to their reliance on static and single-view facial videos. Thus, this work focuses on tackling the motion-induced occlusion problem for rPPG measurement in unconstrained multi-view facial videos. Specifically, we introduce a Multi-View rPPG Dataset (MVRD), a high-quality benchmark dataset featuring synchronized facial videos from three viewpoints under stationary, speaking, and head movement scenarios to better match real-world conditions. We also propose MVRD-rPPG, a unified multi-view rPPG learning framework that fuses complementary visual cues to maintain robust facial skin coverage, especially under motion conditions. Our method integrates an Adaptive Temporal Optical Compensation (ATOC) module for motion artifact suppression, a Rhythm-Visual Dual-Stream Network to disentangle rhythmic and appearance-related features, and a Multi-View Correlation-Aware Attention (MVCA) for adaptive view-wise signal aggregation. Furthermore, we introduce a Correlation Frequency Adversarial (CFA) learning strategy, which jointly enforces temporal accuracy, spectral consistency, and perceptual realism in the predicted signals. Extensive experiments and ablation studies on the MVRD dataset demonstrate the superiority of our approach. In the MVRD movement scenario, MVRD-rPPG achieves an MAE of 0.90 and a Pearson correlation coefficient (R) of 0.99. The source code and dataset will be made available.

  </details>



- **TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment**  
  Chunxia Qin, Chenyu Liu, Pengcheng Xia, Jun Du, Baocai Yin, Bing Yin, Cong Liu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22819v1  
  <details><summary>Abstract</summary>

  Tables are pervasive in diverse documents, making table recognition (TR) a fundamental task in document analysis. Existing modular TR pipelines separately model table structure and content, leading to suboptimal integration and complex workflows. End-to-end approaches rely heavily on large-scale TR data and struggle in data-constrained scenarios. To address these issues, we propose TDATR (Table Detail-Aware Table Recognition) improves end-to-end TR through table detail-aware learning and cell-level visual alignment. TDATR adopts a ``perceive-then-fuse'' strategy. The model first performs table detail-aware learning to jointly perceive table structure and content through multiple structure understanding and content recognition tasks designed under a language modeling paradigm. These tasks can naturally leverage document data from diverse scenarios to enhance model robustness. The model then integrates implicit table details to generate structured HTML outputs, enabling more efficient TR modeling when trained with limited data. Furthermore, we design a structure-guided cell localization module integrated into the end-to-end TR framework, which efficiently locates cell and strengthens vision-language alignment. It enhances the interpretability and accuracy of TR. We achieve state-of-the-art or highly competitive performance on seven benchmarks without dataset-specific fine-tuning.

  </details>



- **Instrument-Splatting++: Towards Controllable Surgical Instrument Digital Twin Using Gaussian Splatting**  
  Shuojue Yang, Zijian Wu, Chengjiaao Liao, Qian Li, Daiyun Shen, Chang Han Low, Septimiu E. Salcudean, Yueming Jin  
  _2026-03-24_ · https://arxiv.org/abs/2603.22792v1  
  <details><summary>Abstract</summary>

  High-quality and controllable digital twins of surgical instruments are critical for Real2Sim in robot-assisted surgery, as they enable realistic simulation, synthetic data generation, and perception learning under novel poses. We present Instrument-Splatting++, a monocular 3D Gaussian Splatting (3DGS) framework that reconstructs surgical instruments as a fully controllable Gaussian asset with high fidelity. Our pipeline starts with part-wise geometry pretraining that injects CAD priors into Gaussian primitives and equips the representation with part-aware semantic rendering. Built on the pretrained model, we propose a semantics-aware pose estimation and tracking (SAPET) method to recover per-frame 6-DoF pose and joint angles from unposed endoscopic videos, where a gripper-tip network trained purely from synthetic semantics provides robust supervision and a loose regularization suppresses singular articulations. Finally, we introduce Robust Texture Learning (RTL), which alternates pose refinement and robust appearance optimization, mitigating pose noise during texture learning. The proposed framework can perform pose estimation and learn realistic texture from unposed videos. We validate our method on sequences extracted from EndoVis17/18, SAR-RARP, and an in-house dataset, showing superior photometric quality and improved geometric accuracy over state-of-the-art baselines. We further demonstrate a downstream keypoint detection task where unseen-pose data augmentation from our controllable instrument Gaussian improves performance.

  </details>



- **From Pixels to Semantics: A Multi-Stage AI Framework for Structural Damage Detection in Satellite Imagery**  
  Bijay Shakya, Catherine Hoier, Khandaker Mamun Ahmed  
  _2026-03-24_ · https://arxiv.org/abs/2603.22768v1  
  <details><summary>Abstract</summary>

  Rapid and accurate structural damage assessment following natural disasters is critical for effective emergency response and recovery. However, remote sensing imagery often suffers from low spatial resolution, contextual ambiguity, and limited semantic interpretability, reducing the reliability of traditional detection pipelines. In this work, we propose a novel hybrid framework that integrates AI-based super-resolution, deep learning object detection, and Vision-Language Models (VLMs) for comprehensive post-disaster building damage assessment. First, we enhance pre- and post-disaster satellite imagery using a Video Restoration Transformer (VRT) to upscale images from 1024x1024 to 4096x4096 resolution, improving structural detail visibility. Next, a YOLOv11-based detector localizes buildings in pre-disaster imagery, and cropped building regions are analyzed using VLMs to semantically assess structural damage across four severity levels. To ensure robust evaluation in the absence of ground-truth captions, we employ CLIPScore for reference-free semantic alignment and introduce a multi-model VLM-as-a-Jury strategy to reduce individual model bias in safety-critical decision making. Experiments on subsets of the xBD dataset, including the Moore Tornado and Hurricane Matthew events, demonstrate that the proposed framework enhances the semantic interpretation of damaged buildings. In addition, our framework provides helpful recommendations to first responders for recovery based on damage analysis.

  </details>



- **ENC-Bench: A Benchmark for Evaluating Multimodal Large Language Models in Electronic Navigational Chart Understanding**  
  Ao Cheng, Xingming Li, Xuanyu Ji, Xixiang He, Qiyao Sun, Chunping Qiu, Runke Huang, Qingyong Hu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22763v1  
  <details><summary>Abstract</summary>

  Electronic Navigational Charts (ENCs) are the safety-critical backbone of modern maritime navigation, yet it remains unclear whether multimodal large language models (MLLMs) can reliably interpret them. Unlike natural images or conventional charts, ENCs encode regulations, bathymetry, and route constraints via standardized vector symbols, scale-dependent rendering, and precise geometric structure -- requiring specialized maritime expertise for interpretation. We introduce ENC-Bench, the first benchmark dedicated to professional ENC understanding. ENC-Bench contains 20,490 expert-validated samples from 840 authentic National Oceanic and Atmospheric Administration (NOAA) ENCs, organized into a three-level hierarchy: Perception (symbol and feature recognition), Spatial Reasoning (coordinate localization, bearing, distance), and Maritime Decision-Making (route legality, safety assessment, emergency planning under multiple constraints). All samples are generated from raw S-57 data through a calibrated vector-to-image pipeline with automated consistency checks and expert review. We evaluate 10 state-of-the-art MLLMs such as GPT-4o, Gemini 2.5, Qwen3-VL, InternVL-3, and GLM-4.5V, under a unified zero-shot protocol. The best model achieves only 47.88% accuracy, with systematic challenges in symbolic grounding, spatial computation, multi-constraint reasoning, and robustness to lighting and scale variations. By establishing the first rigorous ENC benchmark, we open a new research frontier at the intersection of specialized symbolic reasoning and safety-critical AI, providing essential infrastructure for advancing MLLMs toward professional maritime applications.

  </details>



- **MVPBench: A Multi-Video Perception Evaluation Benchmark for Multi-Modal Video Understanding**  
  Purui Bai, Tao Wu, Jiayang Sun, Xinyue Liu, Huaibo Huang, Ran He  
  _2026-03-24_ · https://arxiv.org/abs/2603.22756v1  
  <details><summary>Abstract</summary>

  The rapid progress of Large Language Models (LLMs) has spurred growing interest in Multi-modal LLMs (MLLMs) and motivated the development of benchmarks to evaluate their perceptual and comprehension abilities. Existing benchmarks, however, are limited to static images or single videos, overlooking the complex interactions across multiple videos. To address this gap, we introduce the Multi-Video Perception Evaluation Benchmark (MVPBench), a new benchmark featuring 14 subtasks across diverse visual domains designed to evaluate models on extracting relevant information from video sequences to make informed decisions. MVPBench includes 5K question-answering tests involving 2.7K video clips sourced from existing datasets and manually annotated clips. Extensive evaluations reveal that current models struggle to process multi-video inputs effectively, underscoring substantial limitations in their multi-video comprehension. We anticipate MVPBench will drive advancements in multi-video perception.

  </details>



- **WiFi2Cap: Semantic Action Captioning from Wi-Fi CSI via Limb-Level Semantic Alignment**  
  Tzu-Ti Wei, Chu-Yu Huang, Yu-Chee Tseng, Jen-Jee Chen  
  _2026-03-24_ · https://arxiv.org/abs/2603.22690v1  
  <details><summary>Abstract</summary>

  Privacy-preserving semantic understanding of human activities is important for indoor sensing, yet existing Wi-Fi CSI-based systems mainly focus on pose estimation or predefined action classification rather than fine-grained language generation. Mapping CSI to natural-language descriptions remains challenging because of the semantic gap between wireless signals and language and direction-sensitive ambiguities such as left/right limb confusion. We propose WiFi2Cap, a three-stage framework for generating action captions directly from Wi-Fi CSI. A vision-language teacher learns transferable supervision from synchronized video-text pairs, and a CSI student is aligned to the teacher's visual space and text embeddings. To improve direction-sensitive captioning, we introduce a Mirror-Consistency Loss that reduces mirrored-action and left-right ambiguities during cross-modal alignment. A prefix-tuned language model then generates action descriptions from CSI embeddings. We also introduce the WiFi2Cap Dataset, a synchronized CSI-RGB-sentence benchmark for semantic captioning from Wi-Fi signals. Experimental results show that WiFi2Cap consistently outperforms baseline methods on BLEU-4, METEOR, ROUGE-L, CIDEr, and SPICE, demonstrating effective privacy-friendly semantic sensing.

  </details>



- **Think 360°: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth**  
  Mingrui Chen, Hexiong Yang, Haogeng Liu, Huaibo Huang, Ran He  
  _2026-03-24_ · https://arxiv.org/abs/2603.22689v1  
  <details><summary>Abstract</summary>

  In this paper, we present a holistic multimodal benchmark that evaluates the reasoning capabilities of MLLMs with an explicit focus on reasoning width, a complementary dimension to the more commonly studied reasoning depth. Specifically, reasoning depth measures the model's ability to carry out long-chain, sequential reasoning in which each step is tightly and rigorously linked to the next. Reasoning width tends to focus more on the model's capacity for broad trial-and-error search or multi-constrained optimization: it must systematically traverse many possible and parallelized reasoning paths, apply diverse constraints to prune unpromising branches, and identify valid solution routes for efficient iteration or backtracking. To achieve it, we carefully curate 1200+ high-quality multimodal cases spanning heterogeneous domains, and propose a fine-grained tree-of-thought evaluation protocol that jointly quantifies reasoning width and depth. We evaluate 12 major model families (over 30 advanced MLLMs) across difficulty tiers, question types, and required skills. Results show that while current models exhibit strong performance on general or common-sense VQA tasks, they still struggle to combine deep sequential thought chains with wide exploratory search to perform genuine insight-based reasoning. Finally, we analyze characteristic failure modes to provide possible directions for building MLLMs that reason not only deeper but also wider.

  </details>



- **GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning**  
  Jiayin Sun, Caixia Sun, Boyu Yang, Hailin Li, Xiao Chen, Yi Zhang, Errui Ding, Liang Li, Chao Deng, Junlan Feng  
  _2026-03-24_ · https://arxiv.org/abs/2603.22687v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently demonstrated remarkable perceptual and reasoning abilities. However, they struggle to perceive fine-grained geometric structures, constraining their ability of geometric understanding and visual reasoning. To address this, we propose GeoTikzBridge, a framework that enhances local geometric perception and visual reasoning through tikz-based code generation. Within this framework, we build two models supported by two complementary datasets. The GeoTikzBridge-Base model is trained on GeoTikz-Base dataset, the largest image-to-tikz dataset to date with 2.5M pairs (16 $\times$ larger than existing open-sourced datasets). This process is achieved via iterative data expansion and a localized geometric transformation strategy. Subsequently, GeoTikzBridge-Instruct is fine-tuned on GeoTikz-Instruct dataset which is the first instruction-augmented tikz dataset supporting visual reasoning. Extensive experimental results demonstrate that our models achieve state-of-the-art performance among open-sourced MLLMs. Furthermore, GeoTikzBridge models can serve as plug-and-play reasoning modules for any MLLM(LLM), enhancing reasoning performance in geometric problem-solving. Datasets and codes are publicly available at: https://github.com/sjy-1995/GeoTikzBridge-Advancing-Multimodal-Code-Generation-for-Geometric-Perception-and-Reasoning.

  </details>



- **Large-Scale Avalanche Mapping from SAR Images with Deep Learning-based Change Detection**  
  Mattia Gatti, Alberto Mariani, Ignazio Gallo, Fabiano Monti  
  _2026-03-24_ · https://arxiv.org/abs/2603.22658v1  
  <details><summary>Abstract</summary>

  Accurate change detection from satellite imagery is essential for monitoring rapid mass-movement hazards such as snow avalanches, which increasingly threaten human life, infrastructure, and ecosystems due to their rising frequency and intensity. This study presents a systematic investigation of large-scale avalanche mapping through bi-temporal change detection using Sentinel-1 synthetic aperture radar (SAR) imagery. Extensive experiments across multiple alpine ecoregions with manually validated avalanche inventories show that treating the task as a unimodal change detection problem, relying solely on pre- and post-event SAR images, achieves the most consistent performance. The proposed end-to-end pipeline achieves an F1-score of 0.8061 in a conservative (F1-optimized) configuration and attains an F2-score of 0.8414 with 80.36% avalanche-polygon hit rate under a less conservative, recall-oriented (F2-optimized) tuning. These results highlight the trade-off between precision and completeness and demonstrate how threshold adjustment can improve the detection of smaller or marginal avalanches. The release of the annotated multi-region dataset establishes a reproducible benchmark for SAR-based avalanche mapping.

  </details>



- **PIVM: Diffusion-Based Prior-Integrated Variation Modeling for Anatomically Precise Abdominal CT Synthesis**  
  Dinglun He, Baoming Zhang, Xu Wang, Yao Hao, Deshan Yang, Ye Duan  
  _2026-03-23_ · https://arxiv.org/abs/2603.22626v1  
  <details><summary>Abstract</summary>

  Abdominal CT data are limited by high annotation costs and privacy constraints, which hinder the development of robust segmentation and diagnostic models. We present a Prior-Integrated Variation Modeling (PIVM) framework, a diffusion-based method for anatomically accurate CT image synthesis. Instead of generating full images from noise, PIVM predicts voxel-wise intensity variations relative to organ-specific intensity priors derived from segmentation labels. These priors and labels jointly guide the diffusion process, ensuring spatial alignment and realistic organ boundaries. Unlike latent-space diffusion models, our approach operates directly in image space while preserving the full Hounsfield Unit (HU) range, capturing fine anatomical textures without smoothing. Source code is available at https://github.com/BZNR3/PIVM.

  </details>


