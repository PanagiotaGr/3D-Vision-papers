# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-16 07:46 UTC_

Total papers shown: **38**


---

- **Visual-ERM: Reward Modeling for Visual Equivalence**  
  Ziyu Liu, Shengyuan Ding, Xinyu Fang, Xuanlang Dai, Penghui Yang, Jianze Liang, Jiaqi Wang, Kai Chen, Dahua Lin, Yuhang Zang  
  _2026-03-13_ · https://arxiv.org/abs/2603.13224v1  
  <details><summary>Abstract</summary>

  Vision-to-code tasks require models to reconstruct structured visual inputs, such as charts, tables, and SVGs, into executable or structured representations with high visual fidelity. While recent Large Vision Language Models (LVLMs) achieve strong results via supervised fine-tuning, reinforcement learning remains challenging due to misaligned reward signals. Existing rewards either rely on textual rules or coarse visual embedding similarity, both of which fail to capture fine-grained visual discrepancies and are vulnerable to reward hacking. We propose Visual Equivalence Reward Model (Visual-ERM), a multimodal generative reward model that provides fine-grained, interpretable, and task-agnostic feedback to evaluate vision-to-code quality directly in the rendered visual space. Integrated into RL, Visual-ERM improves Qwen3-VL-8B-Instruct by +8.4 on chart-to-code and yields consistent gains on table and SVG parsing (+2.7, +4.1 on average), and further strengthens test-time scaling via reflection and revision. We also introduce VisualCritic-RewardBench (VC-RewardBench), a benchmark for judging fine-grained image-to-image discrepancies on structured visual data, where Visual-ERM at 8B decisively outperforms Qwen3-VL-235B-Instruct and approaches leading closed-source models. Our results suggest that fine-grained visual reward supervision is both necessary and sufficient for vision-to-code RL, regardless of task specificity.

  </details>



- **Out of Sight, Out of Mind? Evaluating State Evolution in Video World Models**  
  Ziqi Ma, Mengzhan Liufu, Georgia Gkioxari  
  _2026-03-13_ · https://arxiv.org/abs/2603.13215v1  
  <details><summary>Abstract</summary>

  Evolutions in the world, such as water pouring or ice melting, happen regardless of being observed. Video world models generate "worlds" via 2D frame observations. Can these generated "worlds" evolve regardless of observation? To probe this question, we design a benchmark to evaluate whether video world models can decouple state evolution from observation. Our benchmark, STEVO-Bench, applies observation control to evolving processes via instructions of occluder insertion, turning off the light, or specifying camera "lookaway" trajectories. By evaluating video models with and without camera control for a diverse set of naturally-occurring evolutions, we expose their limitations in decoupling state evolution from observation. STEVO-Bench proposes an evaluation protocol to automatically detect and disentangle failure modes of video world models across key aspects of natural state evolution. Analysis of STEVO-Bench results provide new insight into potential data and architecture bias of present-day video world models. Project website: https://glab-caltech.github.io/STEVOBench/. Blog: https://ziqi-ma.github.io/blog/2026/outofsight/

  </details>



- **Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos**  
  Rohith Peddi, Saurabh, Shravan Shanmugam, Likhitha Pallapothula, Yu Xiang, Parag Singla, Vibhav Gogate  
  _2026-03-13_ · https://arxiv.org/abs/2603.13185v1  
  <details><summary>Abstract</summary>

  Spatio-temporal scene graphs provide a principled representation for modeling evolving object interactions, yet existing methods remain fundamentally frame-centric: they reason only about currently visible objects, discard entities upon occlusion, and operate in 2D. To address this, we first introduce ActionGenome4D, a dataset that upgrades Action Genome videos into 4D scenes via feed-forward 3D reconstruction, world-frame oriented bounding boxes for every object involved in actions, and dense relationship annotations including for objects that are temporarily unobserved due to occlusion or camera motion. Building on this data, we formalize World Scene Graph Generation (WSGG), the task of constructing a world scene graph at each timestamp that encompasses all interacting objects in the scene, both observed and unobserved. We then propose three complementary methods, each exploring a different inductive bias for reasoning about unobserved objects: PWG (Persistent World Graph), which implements object permanence via a zero-order feature buffer; MWAE (Masked World Auto-Encoder), which reframes unobserved-object reasoning as masked completion with cross-view associative retrieval; and 4DST (4D Scene Transformer), which replaces the static buffer with differentiable per-object temporal attention enriched by 3D motion and camera-pose features. We further design and evaluate the performance of strong open-source Vision-Language Models on the WSGG task via a suite of Graph RAG-based approaches, establishing baselines for unlocalized relationship prediction. WSGG thus advances video scene understanding toward world-centric, temporally persistent, and interpretable scene reasoning.

  </details>



- **FDeID-Toolbox: Face De-Identification Toolbox**  
  Hui Wei, Hao Yu, Guoying Zhao  
  _2026-03-13_ · https://arxiv.org/abs/2603.13121v1  
  <details><summary>Abstract</summary>

  Face de-identification (FDeID) aims to remove personally identifiable information from facial images while preserving task-relevant utility attributes such as age, gender, and expression. It is critical for privacy-preserving computer vision, yet the field suffers from fragmented implementations, inconsistent evaluation protocols, and incomparable results across studies. These challenges stem from the inherent complexity of the task: FDeID spans multiple downstream applications (e.g., age estimation, gender recognition, expression analysis) and requires evaluation across three dimensions (e.g., privacy protection, utility preservation, and visual quality), making existing codebases difficult to use and extend. To address these issues, we present FDeID-Toolbox, a comprehensive toolbox designed for reproducible FDeID research. Our toolbox features a modular architecture comprising four core components: (1) standardized data loaders for mainstream benchmark datasets, (2) unified method implementations spanning classical approaches to SOTA generative models, (3) flexible inference pipelines, and (4) systematic evaluation protocols covering privacy, utility, and quality metrics. Through experiments, we demonstrate that FDeID-Toolbox enables fair and reproducible comparison of diverse FDeID methods under consistent conditions.

  </details>



- **Geometry-Guided Camera Motion Understanding in VideoLLMs**  
  Haoan Feng, Sri Harsha Musunuri, Guan-Ming Su  
  _2026-03-13_ · https://arxiv.org/abs/2603.13119v1  
  <details><summary>Abstract</summary>

  Camera motion is a fundamental geometric signal that shapes visual perception and cinematic style, yet current video-capable vision-language models (VideoLLMs) rarely represent it explicitly and often fail on fine-grained motion primitives. We address this gap with a framework of $\textbf{benchmarking}$, $\textbf{diagnosis}$, and $\textbf{injection}$. We curate $\textbf{CameraMotionDataset}$, a large-scale synthetic dataset with explicit camera control, formulate camera motion as constraint-aware multi-label recognition, and construct a VQA benchmark--$\textbf{CameraMotionVQA}$. Across diverse off-the-shelf VideoLLMs, we observe substantial errors in recognizing camera motion primitives. Probing experiments on a Qwen2.5-VL vision encoder suggest that camera motion cues are weakly represented, especially in deeper ViT blocks, helping explain the observed failure modes. To bridge this gap without costly training or fine-tuning, we propose a lightweight, model-agnostic pipeline that extracts geometric camera cues from 3D foundation models (3DFMs), predicts constrained motion primitives with a temporal classifier, and injects them into downstream VideoLLM inference via structured prompting. Experiments demonstrate improved motion recognition and more camera-aware model responses, highlighting geometry-driven cue extraction and structured prompting as practical steps toward a camera-aware VideoLLM and VLA system. The dataset and benchmark is publicly available at https://hf.co/datasets/fengyee/camera-motion-dataset-and-benchmark.

  </details>



- **NOIR: Neural Operator mapping for Implicit Representations**  
  Sidaty El Hadramy, Nazim Haouchine, Michael Wehrli, Philippe C. Cattin  
  _2026-03-13_ · https://arxiv.org/abs/2603.13118v1  
  <details><summary>Abstract</summary>

  This paper presents NOIR, a framework that reframes core medical imaging tasks as operator learning between continuous function spaces, challenging the prevailing paradigm of discrete grid-based deep learning. Instead of operating on fixed pixel or voxel grids, NOIR embeds discrete medical signals into shared Implicit Neural Representations and learns a Neural Operator that maps between their latent modulations, enabling resolution-independent function-to-function transformations. We evaluate NOIR across multiple 2D and 3D downstream tasks, including segmentation, shape completion, image-to-image translation, and image synthesis, on several public datasets such as Shenzhen, OASIS-4, SkullBreak, fastMRI, as well as an in-house clinical dataset. It achieves competitive performance at native resolution while demonstrating strong robustness to unseen discretizations, and empirically satisfies key theoretical properties of neural operators. The project page is available here: https://github.com/Sidaty1/NOIR-io.

  </details>



- **Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots**  
  Guoqiang Zhao, Zhe Yang, Sheng Wu, Fei Teng, Mengfei Duan, Yuanfan Zheng, Kai Luo, Kailun Yang  
  _2026-03-13_ · https://arxiv.org/abs/2603.13108v1  
  <details><summary>Abstract</summary>

  Panoramic imagery provides holistic 360° visual coverage for perception in quadruped robots. However, existing occupancy prediction methods are mainly designed for wheeled autonomous driving and rely heavily on RGB cues, limiting their robustness in complex environments. To bridge this gap, (1) we present PanoMMOcc, the first real-world panoramic multimodal occupancy dataset for quadruped robots, featuring four sensing modalities across diverse scenes. (2) We propose a panoramic multimodal occupancy perception framework, VoxelHound, tailored for legged mobility and spherical imaging. Specifically, we design (i) a Vertical Jitter Compensation (VJC) module to mitigate severe viewpoint perturbations caused by body pitch and roll during mobility, enabling more consistent spatial reasoning, and (ii) an effective Multimodal Information Prompt Fusion (MIPF) module that jointly leverages panoramic visual cues and auxiliary modalities to enhance volumetric occupancy prediction. (3) We establish a benchmark based on PanoMMOcc and provide detailed data analysis to enable systematic evaluation of perception methods under challenging embodied scenarios. Extensive experiments demonstrate that VoxelHound achieves state-of-the-art performance on PanoMMOcc (+4.16%} in mIoU). The dataset and code will be publicly released to facilitate future research on panoramic multimodal 3D perception for embodied robotic systems at https://github.com/SXDR/PanoMMOcc, along with the calibration tools released at https://github.com/losehu/CameraLiDAR-Calib.

  </details>



- **BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending**  
  Matteo Ballegeer, Dries F. Benoit  
  _2026-03-13_ · https://arxiv.org/abs/2603.13102v1  
  <details><summary>Abstract</summary>

  Predicting the manufacturability of CAD designs early, in terms of both feasibility and required effort, is a key goal of Design for Manufacturing (DFM). Despite advances in deep learning for CAD and its widespread use in manufacturing process selection, learning-based approaches for predicting manufacturability within a specific process remain limited. Two key challenges limit progress: inconsistency across prior work in how manufacturability is defined and consequently in the associated learning targets, and a scarcity of suitable datasets. Existing labels vary significantly: they may reflect intrinsic design constraints or depend on specific manufacturing capabilities (such as available tools), and they range from discrete feasibility checks to continuous complexity measures. Furthermore, industrial datasets typically contain only manufacturable parts, offering little signal for infeasible cases, while existing synthetic datasets focus on simple geometries and subtractive processes. To address these gaps, we propose a taxonomy of manufacturability metrics along the axes of configuration dependence and measurement type, allowing clearer scoping of generalizability and learning objectives. Next, we introduce BenDFM, the first synthetic dataset for manufacturability assessment in sheet metal bending. BenDFM contains 20,000 parts, both manufacturable and unmanufacturable, generated with process-aware bending simulations, providing both folded and unfolded geometries and multiple manufacturability labels across the taxonomy, enabling systematic study of previously unexplored learning-based DFM challenges. We benchmark two state-of-the-art 3D learning architectures on BenDFM, showing that graph-based representations that capture relationships between part surfaces achieve better accuracy, and that predicting metrics that depend on specific manufacturing setups remains more challenging.

  </details>



- **Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation**  
  Wayner Barrios, SouYoung Jin  
  _2026-03-13_ · https://arxiv.org/abs/2603.13099v1  
  <details><summary>Abstract</summary>

  We introduce **CRYSTAL** (*__C__lear __R__easoning via __Y__ielded __S__teps, __T__raceability and __L__ogic*), a diagnostic benchmark with 6,372 instances that evaluates multimodal reasoning through verifiable intermediate steps. We propose two complementary metrics: *Match F1*, which scores step-level precision and recall via semantic similarity matching, and *Ordered Match F1*, which further penalizes disordered reasoning chains. References are constructed through a Delphi-inspired pipeline where four independent MLLMs generate trajectories, aggregated via semantic clustering and validated through human quality gates. Evaluation of 20 MLLMs, including commercial frontier systems not used during benchmark construction, reveals systematic failures invisible to accuracy: universal cherry-picking (precision far exceeds recall), non-monotonic scaling trade-offs, and disordered reasoning where no competitive model preserves more than 60% of matched steps in correct order. Beyond evaluation, we propose the **Causal Process Reward (CPR)**, a multiplicative reward that couples answer correctness with step-level alignment, and **CPR-Curriculum**, which progressively increases reasoning difficulty during training. CPR-Curriculum achieves +32% Match F1 via GRPO where additive reward strategies fail, improving reasoning without manual step annotation.

  </details>



- **SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design**  
  Ruogu Li, Sikai Li, Yao Mu, Mingyu Ding  
  _2026-03-13_ · https://arxiv.org/abs/2603.13098v1  
  <details><summary>Abstract</summary>

  We introduce SldprtNet, a large-scale dataset comprising over 242,000 industrial parts, designed for semantic-driven CAD modeling, geometric deep learning, and the training and fine-tuning of multimodal models for 3D design. The dataset provides 3D models in both .step and .sldprt formats to support diverse training and testing. To enable parametric modeling and facilitate dataset scalability, we developed supporting tools, an encoder and a decoder, which support 13 types of CAD commands and enable lossless transformation between 3D models and a structured text representation. Additionally, each sample is paired with a composite image created by merging seven rendered views from different viewpoints of the 3D model, effectively reducing input token length and accelerating inference. By combining this image with the parameterized text output from the encoder, we employ the lightweight multimodal language model Qwen2.5-VL-7B to generate a natural language description of each part's appearance and functionality. To ensure accuracy, we manually verified and aligned the generated descriptions, rendered images, and 3D models. These descriptions, along with the parameterized modeling scripts, rendered images, and 3D model files, are fully aligned to construct SldprtNet. To assess its effectiveness, we fine-tuned baseline models on a dataset subset, comparing image-plus-text inputs with text-only inputs. Results confirm the necessity and value of multimodal datasets for CAD generation. It features carefully selected real-world industrial parts, supporting tools for scalable dataset expansion, diverse modalities, and ensured diversity in model complexity and geometric features, making it a comprehensive multimodal dataset built for semantic-driven CAD modeling and cross-modal learning.

  </details>



- **Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence**  
  Seunghwan Bang, Hwanjun Song  
  _2026-03-13_ · https://arxiv.org/abs/2603.13091v1  
  <details><summary>Abstract</summary>

  The growing interest in embodied agents increases the demand for spatiotemporal video understanding, yet existing benchmarks largely emphasize extractive reasoning, where answers can be explicitly presented within spatiotemporal events. It remains unclear whether multimodal large language models can instead perform abstractive spatiotemporal reasoning, which requires integrating observations over time, combining dispersed cues, and inferring implicit spatial and contextual structure. To address this gap, we formalize abstractive spatiotemporal reasoning from videos by introducing a structured evaluation taxonomy that systematically targets its core dimensions and construct a controllable, scenario-driven synthetic egocentric video dataset tailored to evaluate abstractive spatiotemporal reasoning capabilities, spanning object-, room-, and floor-plan-level scenarios. Based on this framework, we present VAEX-BENCH, a benchmark comprising five abstractive reasoning tasks together with their extractive counterparts. Our extensive experiments compare the performance of state-of-the-art MLLMs under extractive and abstractive settings, exposing their limitations on abstractive tasks and providing a fine-grained analysis of the underlying bottlenecks. The dataset will be released soon.

  </details>



- **InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing**  
  Yebin Yang, Di Wen, Lei Qi, Weitong Kong, Junwei Zheng, Ruiping Liu, Yufan Chen, Chengzhi Wu, Kailun Yang, Yuqian Fu, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.13082v1  
  <details><summary>Abstract</summary>

  Text-guided 3D motion editing has seen success in single-person scenarios, but its extension to multi-person settings is less explored due to limited paired data and the complexity of inter-person interactions. We introduce the task of multi-person 3D motion editing, where a target motion is generated from a source and a text instruction. To support this, we propose InterEdit3D, a new dataset with manual two-person motion change annotations, and a Text-guided Multi-human Motion Editing (TMME) benchmark. We present InterEdit, a synchronized classifier-free conditional diffusion model for TMME. It introduces Semantic-Aware Plan Token Alignment with learnable tokens to capture high-level interaction cues and an Interaction-Aware Frequency Token Alignment strategy using DCT and energy pooling to model periodic motion dynamics. Experiments show that InterEdit improves text-to-motion consistency and edit fidelity, achieving state-of-the-art TMME performance. The dataset and code will be released at https://github.com/YNG916/InterEdit.

  </details>



- **Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback**  
  Yuki Hirakawa, Takashi Wada, Ryotaro Shimizu, Takuya Furusawa, Yuki Saito, Ryosuke Araki, Tianwei Chen, Fan Mo, Yoshimitsu Aoki  
  _2026-03-13_ · https://arxiv.org/abs/2603.13057v1  
  <details><summary>Abstract</summary>

  Given a person image and a garment image, image-based Virtual Try-ON (VTON) synthesizes a try-on image of the person wearing the target garment. As VTON systems become increasingly important in practical applications such as fashion e-commerce, reliable evaluation of their outputs has emerged as a critical challenge. In real-world scenarios, ground-truth images of the same person wearing the target garment are typically unavailable, making reference-based evaluation impractical. Moreover, widely used distribution-level metrics such as Fréchet Inception Distance and Kernel Inception Distance measure dataset-level similarity and fail to reflect the perceptual quality of individual generated images. To address these limitations, we propose Image Quality Assessment for Virtual Try-On (VTON-IQA), a reference-free framework for human-aligned, image-level quality assessment without requiring ground-truth images. To model human perceptual judgments, we construct VTON-QBench, a large-scale human-annotated benchmark comprising 62,688 try-on images generated by 14 representative VTON models and 431,800 quality annotations collected from 13,838 qualified annotators. To the best of our knowledge, this is the largest dataset to date for human subjective evaluation in virtual try-on. Evaluating virtual try-on quality requires verifying both garment fidelity and the preservation of person-specific details. To explicitly model such interactions, we introduce an Interleaved Cross-Attention module that extends standard transformer blocks by inserting a cross-attention layer between self-attention and MLP in the latter blocks. Extensive experiments show that VTON-IQA achieves reliable human-aligned image-level quality prediction. Moreover, we conduct a comprehensive benchmark evaluation of 14 representative VTON models using VTON-IQA.

  </details>



- **Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach**  
  Elena Ryumina, Maxim Markitantov, Alexandr Axyonov, Dmitry Ryumin, Mikhail Dolgushin, Denis Dresvyanskiy, Alexey Karpov  
  _2026-03-13_ · https://arxiv.org/abs/2603.13056v1  
  <details><summary>Abstract</summary>

  Continuous emotion recognition in terms of valence and arousal under in-the-wild (ITW) conditions remains a challenging problem due to large variations in appearance, head pose, illumination, occlusions, and subject-specific patterns of affective expression. We present a multimodal method for valence-arousal estimation ITW. Our method combines three complementary modalities: face, behavior, and audio. The face modality relies on GRADA-based frame-level embeddings and Transformer-based temporal regression. We use Qwen3-VL-4B-Instruct to extract behavior-relevant information from video segments, while Mamba is used to model temporal dynamics across segments. The audio modality relies on WavLM-Large with attention-statistics pooling and includes a cross-modal filtering stage to reduce the influence of unreliable or non-speech segments. To fuse modalities, we explore two fusion strategies: a Directed Cross-Modal Mixture-of-Experts Fusion Strategy that learns interactions between modalities with adaptive weighting, and a Reliability-Aware Audio-Visual Fusion Strategy that combines visual features at the frame-level while using audio as complementary context. The results are reported on the Aff-Wild2 dataset following the 10th Affective Behavior Analysis in-the-Wild (ABAW) challenge protocol. Experiments demonstrate that the proposed multimodal fusion strategy achieves a Concordance Correlation Coefficient (CCC) of 0.658 on the Aff-Wild2 development set.

  </details>



- **Topo-R1: Detecting Topological Anomalies via Vision-Language Models**  
  Meilong Xu, Qingqiao Hu, Xiaoling Hu, Shahira Abousamra, Xin Yu, Weimin Lyu, Kehan Qi, Dimitris Samaras, Chao Chen  
  _2026-03-13_ · https://arxiv.org/abs/2603.13054v1  
  <details><summary>Abstract</summary>

  Topological correctness is crucial for tubular structures such as blood vessels, nerve fibers, and road networks. Existing topology-preserving methods rely on domain-specific ground truth, which is costly and rarely transfers across domains. When deployed to a new domain without annotations, a key question arises: how can we detect topological anomalies without ground-truth supervision? We reframe this as topological anomaly detection, a structured visual reasoning task requiring a model to locate and classify topological errors in predicted segmentation masks. Vision-Language Models (VLMs) are natural candidates; however, we find that state-of-the-art VLMs perform nearly at random, lacking the fine-grained, topology-aware perception needed to identify sparse connectivity errors in dense structures. To bridge this gap, we develop an automated data-curation pipeline that synthesizes diverse topological anomalies with verifiable annotations across progressively difficult levels, thereby constructing the first large-scale, multi-domain benchmark for this task. We then introduce Topo-R1, a framework that endows VLMs with topology-aware perception via two-stage training: supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO). Central to our approach is a topology-aware composite reward that integrates type-aware Hungarian matching for structured error classification, spatial localization scoring, and a centerline Dice (clDice) reward that directly penalizes connectivity disruptions, thereby jointly incentivizing semantic precision and structural fidelity. Extensive experiments demonstrate that Topo-R1 establishes a new paradigm for annotation-free topological quality assessment, consistently outperforming general-purpose VLMs and supervised baselines across all evaluation protocols.

  </details>



- **Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study**  
  Vanessa Borst, Samuel Kounev  
  _2026-03-13_ · https://arxiv.org/abs/2603.13044v1  
  <details><summary>Abstract</summary>

  Medical image segmentation (MIS) is a fundamental component of computer-assisted diagnosis and clinical decision support systems. Over the past decade, numerous architectures specifically tailored to medical imaging have emerged to address domain-specific challenges such as low contrast, small anatomical structures, and limited annotated data. In parallel, rapid progress in computer vision has produced highly capable general-purpose vision models (GP-VMs) originally designed for natural images. Despite their strong performance on standard vision benchmarks, their effectiveness for MIS remains insufficiently understood. In this work, we conduct a controlled empirical study to examine whether specialized medical segmentation architectures (SMAs) provide systematic advantages over modern GP-VMs for 2D MIS. We compare eleven SMAs and GP-VMs using a unified training and evaluation protocol. Experiments are performed across three heterogeneous datasets covering different imaging modalities, class structures, and data characteristics. Beyond segmentation accuracy, we analyze qualitative Grad-CAM visualizations to investigate explainability (XAI) behavior. Our results demonstrate that, for the analyzed datasets, GP-VMs out-perform the majority of specialized MIS models. Moreover, XAI analyses indicate that GP-VMs can capture clinically relevant structures without explicit domain-specific architectural design. These findings suggest that GP-VMs can represent a viable alternative to domain-specific methods, highlighting the importance of informed model selection for end-to-end MIS systems. All code and resources are available at GitHub.

  </details>



- **ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models**  
  Yanpeng Zhao, Wentao Ding, Hongtao Li, Baoxiong Jia, Zilong Zheng  
  _2026-03-13_ · https://arxiv.org/abs/2603.13033v1  
  <details><summary>Abstract</summary>

  A recent trend in vision-language models (VLMs) has been to enhance their spatial cognition for embodied domains. Despite progress, existing evaluations have been limited both in paradigm and in coverage, hindering rapid, iterative model development. To address these limitations, we propose ESPIRE, a diagnostic benchmark for embodied spatial reasoning. ESPIRE offers a simulated world that physically grounds VLMs and evaluates them on spatial-reasoning-centric robotic tasks, thus narrowing the gap between evaluation and real-world deployment. To adapt VLMs to robotic tasks, we decompose each task into localization and execution, and frame both as generative problems, in stark contrast to predominant discriminative evaluations (e.g., via visual-question answering) that rely on distractors and discard execution. This decomposition further enables a fine-grained analysis beyond passive spatial reasoning toward reasoning to act. We systematically design ESPIRE both at the instruction level and at the environment level, ensuring broad coverage of spatial reasoning scenarios. We use ESPIRE to diagnose a range of frontier VLMs and provide in-depth analysis of their spatial reasoning behaviors.

  </details>



- **SortScrews: A Dataset and Baseline for Real-time Screw Classification**  
  Tianhao Fu, Bingxuan Yang, Juncheng Guo, Shrena Sribalan, Yucheng Chen  
  _2026-03-13_ · https://arxiv.org/abs/2603.13027v1  
  <details><summary>Abstract</summary>

  Automatic identification of screw types is important for industrial automation, robotics, and inventory management. However, publicly available datasets for screw classification are scarce, particularly for controlled single-object scenarios commonly encountered in automated sorting systems. In this work, we introduce $\textbf{SortScrews}$, a dataset for casewise visual classification of screws. The dataset contains 560 RGB images at $512\times512$ resolution covering six screw types and a background class. Images are captured using a standardized acquisition setup and include mild variations in lighting and camera perspective across four capture settings. To facilitate reproducible research and dataset expansion, we also provide a reusable data collection script that allows users to easily construct similar datasets for custom hardware components using inexpensive camera setups. We establish baseline results using transfer learning with EfficientNet-B0 and ResNet-18 classifiers pretrained on ImageNet. In addition, we conduct a well-explored failure analysis. Despite the limited dataset size, these lightweight models achieve strong classification accuracy, demonstrating that controlled acquisition conditions enable effective learning even with relatively small datasets. The dataset, collection pipeline, and baseline training code are publicly available at https://github.com/ATATC/SortScrews.

  </details>



- **SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation**  
  Sampath Rapuri, Lalithkumar Seenivasan, Dominik Schneider, Roger Soberanis-Mukul, Yufan He, Hao Ding, Jiru Xu, Chenhao Yu, Chenyan Jing, Pengfei Guo, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.13024v1  
  <details><summary>Abstract</summary>

  A surgical world model capable of generating realistic surgical action videos with precise control over tool-tissue interactions can address fundamental challenges in surgical AI and simulation -- from data scarcity and rare event synthesis to bridging the sim-to-real gap for surgical automation. However, current video generation methods, the very core of such surgical world models, require expensive annotations or complex structured intermediates as conditioning signals at inference, limiting their scalability. Other approaches exhibit limited temporal consistency across complex laparoscopic scenes and do not possess sufficient realism. We propose Surgical Action World (SAW) -- a step toward surgical action world modeling through video diffusion conditioned on four lightweight signals: language prompts encoding tool-action context, a reference surgical scene, tissue affordance mask, and 2D tool-tip trajectories. We design a conditional video diffusion approach that reformulates video-to-video diffusion into trajectory-conditioned surgical action synthesis. The backbone diffusion model is fine-tuned on a custom-curated dataset of 12,044 laparoscopic clips with lightweight spatiotemporal conditioning signals, leveraging a depth consistency loss to enforce geometric plausibility without requiring depth at inference. SAW achieves state-of-the-art temporal consistency (CD-FVD: 199.19 vs. 546.82) and strong visual quality on held-out test data. Furthermore, we demonstrate its downstream utility for (a) surgical AI, where augmenting rare actions with SAW-generated videos improves action recognition (clipping F1-score: 20.93% to 43.14%; cutting: 0.00% to 8.33%) on real test data, and (b) surgical simulation, where rendering tool-tissue interaction videos from simulator-derived trajectory points toward a visually faithful simulation engine.

  </details>



- **Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning**  
  Yamin Arefeen, Sidharth Kumar, Steven Warach, Hamidreza Saber, Jonathan Tamir  
  _2026-03-13_ · https://arxiv.org/abs/2603.13007v1  
  <details><summary>Abstract</summary>

  Purpose: To develop a data-efficient strategy for accelerated MRI reconstruction with Diffusion Probabilistic Generative Models (DPMs) that enables faster scan times in clinical stroke MRI when only limited fully-sampled data samples are available. Methods: Our simple training strategy, inspired by the foundation model paradigm, first trains a DPM on a large, diverse collection of publicly available brain MRI data in fastMRI and then fine-tunes on a small dataset from the target application using carefully selected learning rates and fine-tuning durations. The approach is evaluated on controlled fastMRI experiments and on clinical stroke MRI data with a blinded clinical reader study. Results: DPMs pre-trained on approximately 4000 subjects with non-FLAIR contrasts and fine-tuned on FLAIR data from only 20 target subjects achieve reconstruction performance comparable to models trained with substantially more target-domain FLAIR data across multiple acceleration factors. Experiments reveal that moderate fine-tuning with a reduced learning rate yields improved performance, while insufficient or excessive fine-tuning degrades reconstruction quality. When applied to clinical stroke MRI, a blinded reader study involving two neuroradiologists indicates that images reconstructed using the proposed approach from $2 \times$ accelerated data are non-inferior to standard-of-care in terms of image quality and structural delineation. Conclusion: Large-scale pre-training combined with targeted fine-tuning enables DPM-based MRI reconstruction in data-constrained, accelerated clinical stroke MRI. The proposed approach substantially reduces the need for large application-specific datasets while maintaining clinically acceptable image quality, supporting the use of foundation-inspired diffusion models for accelerated MRI in targeted applications.

  </details>



- **Efficient Real-World Autonomous Racing via Attenuated Residual Policy Optimization**  
  Raphael Trumpp, Denis Hoornaert, Mirco Theile, Marco Caccamo  
  _2026-03-13_ · https://arxiv.org/abs/2603.12960v1  
  <details><summary>Abstract</summary>

  Residual policy learning (RPL), in which a learned policy refines a static base policy using deep reinforcement learning (DRL), has shown strong performance across various robotic applications. Its effectiveness is particularly evident in autonomous racing, a domain that serves as a challenging benchmark for real-world DRL. However, deploying RPL-based controllers introduces system complexity and increases inference latency. We address this by introducing an extension of RPL named attenuated residual policy optimization ($α$-RPO). Unlike standard RPL, $α$-RPO yields a standalone neural policy by progressively attenuating the base policy, which initially serves to bootstrap learning. Furthermore, this mechanism enables a form of privileged learning, where the base policy is permitted to use sensor modalities not required for final deployment. We design $α$-RPO to integrate seamlessly with PPO, ensuring that the attenuated influence of the base controller is dynamically compensated during policy optimization. We evaluate $α$-RPO by building a framework for 1:10-scaled autonomous racing around it. In both simulation and zero-shot real-world transfer to Roboracer cars, $α$-RPO not only reduces system complexity but also improves driving performance compared to baselines - demonstrating its practicality for robotic deployment. Our code is available at: https://github.com/raphajaner/arpo_racing.

  </details>



- **Rethinking VLMs for Image Forgery Detection and Localization**  
  Shaofeng Guo, Jiequan Cui, Richang Hong  
  _2026-03-13_ · https://arxiv.org/abs/2603.12930v1  
  <details><summary>Abstract</summary>

  With the rapid rise of Artificial Intelligence Generated Content (AIGC), image manipulation has become increasingly accessible, posing significant challenges for image forgery detection and localization (IFDL). In this paper, we study how to fully leverage vision-language models (VLMs) to assist the IFDL task. In particular, we observe that priors from VLMs hardly benefit the detection and localization performance and even have negative effects due to their inherent biases toward semantic plausibility rather than authenticity. Additionally, the location masks explicitly encode the forgery concepts, which can serve as extra priors for VLMs to ease their training optimization, thus enhancing the interpretability of detection and localization results. Building on these findings, we propose a new IFDL pipeline named IFDL-VLM. To demonstrate the effectiveness of our method, we conduct experiments on 9 popular benchmarks and assess the model performance under both in-domain and cross-dataset generalization settings. The experimental results show that we consistently achieve new state-of-the-art performance in detection, localization, and interpretability.Code is available at: https://github.com/sha0fengGuo/IFDL-VLM.

  </details>



- **A protocol for evaluating robustness to H&E staining variation in computational pathology models**  
  Lydia A. Schönpflug, Nikki van den Berg, Sonali Andani, Nanda Horeweg, Jurriaan Barkey Wolf, Tjalling Bosse, Viktor H. Koelzer, Maxime W. Lafarge  
  _2026-03-13_ · https://arxiv.org/abs/2603.12886v1  
  <details><summary>Abstract</summary>

  Sensitivity to staining variation remains a major barrier to deploying computational pathology (CPath) models as hematoxylin and eosin (H&E) staining varies across laboratories, requiring systematic assessment of how this variability affects model prediction. In this work, we developed a three-step protocol for evaluating robustness to H&E staining variation in CPath models. Step 1: Select reference staining conditions, Step 2: Characterize test set staining properties, Step 3: Apply CPath model(s) under simulated reference staining conditions. Here, we first created a new reference staining library based on the PLISM dataset. As an exemplary use case, we applied the protocol to assess the robustness properties of 306 microsatellite instability (MSI) classification models on the unseen SurGen colorectal cancer dataset (n=738), including 300 attention-based multiple instance learning models trained on the TCGA-COAD/READ datasets across three feature extractors (UNI2-h, H-Optimus-1, Virchow2), alongside six public MSI classification models. Classification performance was measured as AUC, and robustness as the min-max AUC range across four simulated staining conditions (low/high H&E intensity, low/high H&E color similarity). Across models and staining conditions, classification performance ranged from AUC 0.769-0.911 ($Δ$ = 0.142). Robustness ranged from 0.007-0.079 ($Δ$ = 0.072), and showed a weak inverse correlation with classification performance (Pearson r=-0.22, 95% CI [-0.34, -0.11]). Thus, we show that the proposed evaluation protocol enables robustness-informed CPath model selection and provides insight into performance shifts across H&E staining conditions, supporting the identification of operational ranges for reliable model deployment. Code is available at https://github.com/CTPLab/staining-robustness-evaluation .

  </details>



- **Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach**  
  Falko Kähler, Maxim Wille, Ole Schmedemann, Thorsten Schüppstuhl  
  _2026-03-13_ · https://arxiv.org/abs/2603.12852v1  
  <details><summary>Abstract</summary>

  Abrasive flap wheels are common for finishing complex free-form surfaces due to their flexibility. However, this flexibility results in complex wear patterns such as concave/convex flap profiles or flap tears, which influence the grinding result. This paper proposes a novel, vision-based hierarchical classification framework to automate the wear condition monitoring of flap wheels. Unlike monolithic classification approaches, we decompose the problem into three logical levels: (1) state detection (new vs. worn), (2) wear type identification (rectangular, concave, convex) and flap tear detection, and (3) severity assessment (partial vs. complete deformation). A custom-built dataset of real flap wheel images was generated and a transfer learning approach with EfficientNetV2 architecture was used. The results demonstrate high robustness with classification accuracies ranging from 93.8% (flap tears) to 99.3% (concave severity). Furthermore, Gradient-weighted Class Activation Mapping (Grad-CAM) is utilized to validate that the models learn physically relevant features and examine false classifications. The proposed hierarchical method provides a basis for adaptive process control and wear consideration in automated flap wheel grinding.

  </details>



- **AoI-FusionNet: Age-Aware Tightly Coupled Fusion of UWB-IMU under Sparse Ranging Conditions**  
  Tehmina Bibi, Anselm Köhler, Jan-Thomas Fischer, Falko Dressler  
  _2026-03-13_ · https://arxiv.org/abs/2603.12849v1  
  <details><summary>Abstract</summary>

  Accurate motion tracking of snow particles in avalanche events requires robust localization in global navigation satellite system (GNSS)-denied outdoor environments. This paper introduces AoI-FusionNet, a tightly coupled deep learning-based fusion framework that directly combines raw ultra-wideband (UWB) time-of-flight (ToF) measurements with inertial measurement unit (IMU) data for 3D trajectory estimation. Unlike loose-coupled pipelines based on intermediate trilateration, the proposed approach operates directly on heterogeneous sensor inputs, enabling localization even under insufficient ranging availability. The framework integrates an Age-of-Information (AoI)-aware decay module to reduce the influence of stale UWB ranging measurements and a learned attention gating mechanism that adaptively balances the contribution of UWB and IMU modalities based on measurement availability and temporal freshness. To evaluate robustness under limited data and measurement variability, we apply a diffusion-based residual augmentation strategy during training, producing an augmented variant termed AoI-FusionNet-DGAN. We assess the performance of the proposed model using offline post-processing of real-world measurement data collected in an alpine environment and benchmark it against UWB multilateration and loose-coupled fusion baselines. The results demonstrate that AoI-FusionNet substantially reduces mean and tail localization errors under intermittent and degraded sensing conditions.

  </details>



- **Hierarchical Dual-Change Collaborative Learning for UAV Scene Change Captioning**  
  Fuhai Chen, Pengpeng Huang, Junwen Wu, Hehong Zhang, Shiping Wang, Xiaoguang Ma, Xuri Ge  
  _2026-03-13_ · https://arxiv.org/abs/2603.12832v1  
  <details><summary>Abstract</summary>

  This paper proposes a novel task for UAV scene understanding - UAV Scene Change Captioning (UAV-SCC) - which aims to generate natural language descriptions of semantic changes in dynamic aerial imagery captured from a movable viewpoint. Unlike traditional change captioning that mainly describes differences between image pairs captured from a fixed camera viewpoint over time, UAV scene change captioning focuses on image-pair differences resulting from both temporal and spatial scene variations dynamically captured by a moving camera. The key challenge lies in understanding viewpoint-induced scene changes from UAV image pairs that share only partially overlapping scene content due to viewpoint shifts caused by camera rotation, while effectively exploiting the relative orientation between the two images. To this end, we propose a Hierarchical Dual-Change Collaborative Learning (HDC-CL) method for UAV scene change captioning. In particular, a novel transformer, \emph{i.e.} Dynamic Adaptive Layout Transformer (DALT) is designed to adaptively model diverse spatial layouts of the image pair, where the interrelated features derived from the overlapping and non-overlapping regions are learned within the flexible and unified encoding layer. Furthermore, we propose a Hierarchical Cross-modal Orientation Consistency Calibration (HCM-OCC) method to enhance the model's sensitivity to viewpoint shift directions, enabling more accurate change captioning. To facilitate in-depth research on this task, we construct a new benchmark dataset, named UAV-SCC dataset, for UAV scene change captioning. Extensive experiments demonstrate that the proposed method achieves state-of-the-art performance on this task. The dataset and code will be publicly released upon acceptance of this paper.

  </details>



- **NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval**  
  Zhuchenyang Liu, Yao Zhang, Yu Xiao  
  _2026-03-13_ · https://arxiv.org/abs/2603.12824v1  
  <details><summary>Abstract</summary>

  Vision-Language Model (VLM) based retrievers have advanced visual document retrieval (VDR) to impressive quality. They require the same multi-billion parameter encoder for both document indexing and query encoding, incurring high latency and GPU dependence even for plain-text queries. We observe that this design is unnecessarily symmetric: documents are visually complex and demand strong visual understanding, whereas queries are just short text strings. NanoVDR exploits this query--document asymmetry by decoupling the two encoding paths: a frozen 2B VLM teacher indexes documents offline, while a distilled text-only student as small as 69M parameters encodes queries at inference. The key design choice is the distillation objective. Through systematic comparison of six objectives across three backbones and 22 ViDoRe benchmark datasets, we find that pointwise cosine alignment on query text consistently outperforms ranking-based and contrastive alternatives, while requiring only pre-cached teacher query embeddings and no document processing during training. Furthermore, we identify cross-lingual transfer as the primary performance bottleneck, and resolve it cheaply by augmenting training data with machine-translated queries. The resulting NanoVDR-S-Multi (DistilBERT, 69M) retains 95.1\% of teacher quality and outperforms DSE-Qwen2 (2B) on v2 and v3 with 32$\times$ fewer parameters and 50$\times$ lower CPU query latency, at a total training cost under 13 GPU-hours.

  </details>



- **Adaptive Vision-Language Model Routing for Computer Use Agents**  
  Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen  
  _2026-03-13_ · https://arxiv.org/abs/2603.12823v1  
  <details><summary>Abstract</summary>

  Computer Use Agents (CUAs) translate natural-language instructions into Graphical User Interface (GUI) actions such as clicks, keystrokes, and scrolls by relying on a Vision-Language Model (VLM) to interpret screenshots and predict grounded tool calls. However, grounding accuracy varies dramatically across VLMs, while current CUA systems typically route every action to a single fixed model regardless of difficulty. We propose \textbf{Adaptive VLM Routing} (AVR), a framework that inserts a lightweight semantic routing layer between the CUA orchestrator and a pool of VLMs. For each tool call, AVR estimates action difficulty from multimodal embeddings, probes a small VLM to measure confidence, and routes the action to the cheapest model whose predicted accuracy satisfies a target reliability threshold. For \textit{warm} agents with memory of prior UI interactions, retrieved context further narrows the capability gap between small and large models, allowing many actions to be handled without escalation. We formalize routing as a cost--accuracy trade-off, derive a threshold-based policy for model selection, and evaluate AVR using ScreenSpot-Pro grounding data together with the OpenClaw agent routing benchmark. Across these settings, AVR projects inference cost reductions of up to 78\% while staying within 2 percentage points of an all-large-model baseline. When combined with the Visual Confused Deputy guardrail, AVR also escalates high-risk actions directly to the strongest available model, unifying efficiency and safety within a single routing framework. Materials are also provided Model, benchmark, and code: https://github.com/vllm-project/semantic-router.

  </details>



- **OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution**  
  Shijie Zhao, Xuanyu Zhang, Bin Chen, Weiqi Li, Qunliang Xing, Kexin Zhang, Yan Wang, Junlin Li, Li Zhang, Jian Zhang, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.12811v1  
  <details><summary>Abstract</summary>

  Aligning generative real-world image super-resolution models with human visual preference is challenging due to the perception--fidelity trade-off and diverse, unknown degradations. Prior approaches rely on offline preference optimization and static metric aggregation, which are often non-interpretable and prone to pseudo-diversity under strong conditioning. We propose OARS, a process-aware online alignment framework built on COMPASS, a MLLM-based reward that evaluates the LR to SR transition by jointly modeling fidelity preservation and perceptual gain with an input-quality-adaptive trade-off. To train COMPASS, we curate COMPASS-20K spanning synthetic and real degradations, and introduce a three-stage perceptual annotation pipeline that yields calibrated, fine-grained training labels. Guided by COMPASS, OARS performs progressive online alignment from cold-start flow matching to full-reference and finally reference-free RL via shallow LoRA optimization for on-policy exploration. Extensive experiments and user studies demonstrate consistent perceptual improvements while maintaining fidelity, achieving state-of-the-art performance on Real-ISR benchmarks.

  </details>



- **FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning**  
  Zeying Gong, Yangyi Zhong, Yiyi Ding, Tianshuai Hu, Guoyang Zhao, Lingdong Kong, Rong Li, Jiadi You, Junwei Liang  
  _2026-03-13_ · https://arxiv.org/abs/2603.12806v1  
  <details><summary>Abstract</summary>

  Autonomous navigation requires a broad spectrum of skills, from static goal-reaching to dynamic social traversal, yet evaluation remains fragmented across disparate protocols. We introduce DynBench, a dynamic navigation benchmark featuring physically valid crowd simulation. Combined with existing static protocols, it supports comprehensive evaluation across six fundamental navigation tasks. Within this framework, we propose FLUX, the first flow-based unified navigation policy. By linearizing probability flow, FLUX replaces iterative denoising with straight-line trajectories, improving per-step inference efficiency by 47% over prior flow-based methods and 29% over diffusion-based ones. Following a static-to-dynamic curriculum, FLUX initially establishes geometric priors and is subsequently refined through reinforcement learning in dynamic social environments. This regime not only strengthens socially-aware navigation but also enhances static task robustness by capturing recovery behaviors through stochastic action distributions. FLUX achieves state-of-the-art performance across all tasks and demonstrates zero-shot sim-to-real transfer on wheeled, quadrupedal, and humanoid platforms without any fine-tuning.

  </details>



- **GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification**  
  Jiao Wang, Chi Liu, Yiying Zhang, Hongchen Luo, Zhifen Guo, Ying Hu, Ke Xu, Jing Zhou, Hongyan Xu, Ruiting Zhou, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.12800v1  
  <details><summary>Abstract</summary>

  We propose glaucoma lesion evaluation and analysis with multimodal imaging (GLEAM), the first publicly available tri-modal glaucoma dataset comprising scanning laser ophthalmoscopy fundus images, circumpapillary OCT images, and visual field pattern deviation maps, annotated with four disease stages, enabling effective exploitation of multimodal complementary information and facilitating accurate diagnosis and treatment across disease stages. To effectively integrate cross-modal information, we propose hierarchical attentive masked modeling (HAMM) for multimodal glaucoma classification. Our framework employs hierarchical attentive encoders and light decoders to focus cross-modal representation learning on the encoder.

  </details>



- **Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing**  
  Shuchang Lyu, Haiquan Wen, Guangliang Cheng, Meng Li, Zheng Zhou, You Zhou, Dingding Yao, Zhenwei Shi  
  _2026-03-13_ · https://arxiv.org/abs/2603.12788v1  
  <details><summary>Abstract</summary>

  Recent advances in reasoning language models and reinforcement learning with verifiable rewards have significantly enhanced multi-step reasoning capabilities. This progress motivates the extension of reasoning paradigms to remote sensing visual grounding task. However, existing remote sensing grounding methods remain largely confined to perception-level matching and single-entity formulations, limiting the role of explicit reasoning and inter-entity modeling. To address this challenge, we introduce a new benchmark dataset for Multi-Entity Reasoning Grounding in Remote Sensing (ME-RSRG). Based on ME-RSRG, we reformulate remote sensing grounding as a multi-entity reasoning task and propose an Entity-Aware Reasoning (EAR) framework built upon visual-linguistic foundation models. EAR generates structured reasoning traces and subject-object grounding outputs. It adopts supervised fine-tuning for cold-start initialization and is further optimized via entity-aware reward-driven Group Relative Policy Optimization (GRPO). Extensive experiments on ME-RSRG demonstrate the challenges of multi-entity reasoning and verify the effectiveness of our proposed EAR framework. Our dataset, code, and models will be available at https://github.com/CV-ShuchangLyu/ME-RSRG.

  </details>



- **Generalized Recognition of Basic Surgical Actions Enables Skill Assessment and Vision-Language-Model-based Surgical Planning**  
  Mengya Xu, Daiyun Shen, Jie Zhang, Hon Chi Yip, Yujia Gao, Cheng Chen, Dillan Imans, Yonghao Long, Yiru Ye, Yixiao Liu, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.12787v1  
  <details><summary>Abstract</summary>

  Artificial intelligence, imaging, and large language models have the potential to transform surgical practice, training, and automation. Understanding and modeling of basic surgical actions (BSA), the fundamental unit of operation in any surgery, is important to drive the evolution of this field. In this paper, we present a BSA dataset comprising 10 basic actions across 6 surgical specialties with over 11,000 video clips, which is the largest to date. Based on the BSA dataset, we developed a new foundation model that conducts general-purpose recognition of basic actions. Our approach demonstrates robust cross-specialist performance in experiments validated on datasets from different procedural types and various body parts. Furthermore, we demonstrate downstream applications enabled by the BAS foundation model through surgical skill assessment in prostatectomy using domain-specific knowledge, and action planning in cholecystectomy and nephrectomy using large vision-language models. Multinational surgeons' evaluation of the language model's output of the action planning explainable texts demonstrated clinical relevance. These findings indicate that basic surgical actions can be robustly recognized across scenarios, and an accurate BSA understanding model can essentially facilitate complex applications and speed up the realization of surgical superintelligence.

  </details>



- **SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion**  
  Xiang Li, Heqian Qiu, Lanxiao Wang, Benliu Qiu, Fanman Meng, Linfeng Xu, Hongliang Li  
  _2026-03-13_ · https://arxiv.org/abs/2603.12764v1  
  <details><summary>Abstract</summary>

  Error detection is crucial in industrial training, healthcare, and assembly quality control. Most existing work assumes a single-view setting and cannot handle the practical case where a third-person (exo) demonstration is used to assess a first-person (ego) imitation. We formalize Ego$\rightarrow$Exo Imitation Error Detection: given asynchronous, length-mismatched ego and exo videos, the model must localize procedural steps on the ego timeline and decide whether each is erroneous. This setting introduces cross-view domain shift, temporal misalignment, and heavy redundancy. Under a unified protocol, we adapt strong baselines from dense video captioning and temporal action detection and show that they struggle in this cross-view regime. We then propose SAVA-X, an Align-Fuse-Detect framework with (i) view-conditioned adaptive sampling, (ii) scene-adaptive view embeddings, and (iii) bidirectional cross-attention fusion. On the EgoMe benchmark, SAVA-X consistently improves AUPRC and mean tIoU over all baselines, and ablations confirm the complementary benefits of its components. Code is available at https://github.com/jack1ee/SAVAX.

  </details>



- **TerraFlow: Multimodal, Multitemporal Representation Learning for Earth Observation**  
  Nazar Puriy, Johannes Jakubik, Benedikt Blumenstiel, Konrad Schindler  
  _2026-03-13_ · https://arxiv.org/abs/2603.12762v1  
  <details><summary>Abstract</summary>

  We propose TerraFlow, a novel approach to multimodal, multitemporal learning for Earth observation. TerraFlow builds on temporal training objectives that enable sequence-aware learning across space, time, and modality, while remaining robust to the variable-length inputs commonly encountered in real-world Earth observation data. Our experiments demonstrate superiority of TerraFlow over state-of-the-art foundation models for Earth observation across all temporal tasks of the GEO-Bench-2 benchmark. We additionally demonstrate that TerraFlow is able to make initial steps towards deep-learning based risk map prediction for natural disasters -- a task on which other state-of-the-art foundation models frequently collapse. TerraFlow outperforms state-of-the-art foundation models by up to 50% in F1 score and 24% in Brier score.

  </details>



- **SAP: Segment Any 4K Panorama**  
  Lutao Jiang, Zidong Cao, Weikai Chen, Xu Zheng, Yuanhuiyi Lyu, Zhenyang Li, Zeyu HU, Yingda Yin, Keyang Luo, Runze Zhang, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.12759v1  
  <details><summary>Abstract</summary>

  Promptable instance segmentation is widely adopted in embodied and AR systems, yet the performance of foundation models trained on perspective imagery often degrades on 360° panoramas. In this paper, we introduce Segment Any 4K Panorama (SAP), a foundation model for 4K high-resolution panoramic instance-level segmentation. We reformulate panoramic segmentation as fixed-trajectory perspective video segmentation, decomposing a panorama into overlapping perspective patches sampled along a continuous spherical traversal. This memory-aligned reformulation preserves native 4K resolution while restoring the smooth viewpoint transitions required for stable cross-view propagation. To enable large-scale supervision, we synthesize 183,440 4K-resolution panoramic images with instance segmentation labels using the InfiniGen engine. Trained under this trajectory-aligned paradigm, SAP generalizes effectively to real-world 360° images, achieving +17.2 zero-shot mIoU gain over vanilla SAM2 of different sizes on real-world 4K panorama benchmark.

  </details>



- **FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking**  
  Cheng Ju, Zejing Zhao, Akio Namiki  
  _2026-03-13_ · https://arxiv.org/abs/2603.12758v1  
  <details><summary>Abstract</summary>

  Reliable multi-object tracking (MOT) is essential for robotic systems operating in complex and dynamic environments. Despite recent advances in detection and association, online MOT methods remain vulnerable to identity switches caused by frequent occlusions and object overlap, where incorrect associations can propagate over time and degrade tracking reliability. We present a lightweight post-association correction framework (FC-Track) for online MOT that explicitly targets overlap-induced mismatches during inference. The proposed method suppresses unreliable appearance updates under high-overlap conditions using an Intersection over Area (IoA)-based filtering strategy, and locally corrects detection-to-tracklet mismatches through appearance similarity comparison within overlapped tracklet pairs. By preventing short-term mismatches from propagating, our framework effectively mitigates long-term identity switches without resorting to global optimization or re-identification. The framework operates online without global optimization or re-identification, making it suitable for real-time robotic applications. We achieve 81.73 MOTA, 82.81 IDF1, and 66.95 HOTA on the MOT17 test set with a running speed of 5.7 FPS, and 77.52 MOTA, 80.90 IDF1, and 65.67 HOTA on the MOT20 test set with a running speed of 0.6 FPS. Specifically, our framework FC-Track produces only 29.55% long-term identity switches, which is substantially lower than existing online trackers. Meanwhile, our framework maintains state-of-the-art performance on the MOT20 benchmark.

  </details>



- **Show, Don't Tell: Detecting Novel Objects by Watching Human Videos**  
  James Akl, Jose Nicolas Avendano Arbelaez, James Barabas, Jennifer L. Barry, Kalie Ching, Noam Eshed, Jiahui Fu, Michel Hidalgo, Andrew Hoelscher, Tushar Kusnur, et al.  
  _2026-03-13_ · https://arxiv.org/abs/2603.12751v1  
  <details><summary>Abstract</summary>

  How can a robot quickly identify and recognize new objects shown to it during a human demonstration? Existing closed-set object detectors frequently fail at this because the objects are out-of-distribution. While open-set detectors (e.g., VLMs) sometimes succeed, they often require expensive and tedious human-in-the-loop prompt engineering to uniquely recognize novel object instances. In this paper, we present a self-supervised system that eliminates the need for tedious language descriptions and expensive prompt engineering by training a bespoke object detector on an automatically created dataset, supervised by the human demonstration itself. In our approach, "Show, Don't Tell," we show the detector the specific objects of interest during the demonstration, rather than telling the detector about these objects via complex language descriptions. By bypassing language altogether, this paradigm enables us to quickly train bespoke detectors tailored to the relevant objects observed in human task demonstrations. We develop an integrated on-robot system to deploy our "Show, Don't Tell" paradigm of automatic dataset creation and novel object-detection on a real-world robot. Empirical results demonstrate that our pipeline significantly outperforms state-of-the-art detection and recognition methods for manipulated objects, leading to improved task completion for the robot.

  </details>


