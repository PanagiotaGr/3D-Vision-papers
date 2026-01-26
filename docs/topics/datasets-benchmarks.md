# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-26 06:54 UTC_

Total papers shown: **20**


---

- **AnyView: Synthesizing Any Novel View in Dynamic Scenes**  
  Basile Van Hoorick, Dian Chen, Shun Iwase, Pavel Tokmakov, Muhammad Zubair Irshad, Igor Vasiljevic, Swati Gupta, Fangzhou Cheng, Sergey Zakharov, Vitor Campagnolo Guizilini  
  _2026-01-23_ · https://arxiv.org/abs/2601.16982v1  
  <details><summary>Abstract</summary>

  Modern generative video models excel at producing convincing, high-quality outputs, but struggle to maintain multi-view and spatiotemporal consistency in highly dynamic real-world environments. In this work, we introduce \textbf{AnyView}, a diffusion-based video generation framework for \emph{dynamic view synthesis} with minimal inductive biases or geometric assumptions. We leverage multiple data sources with various levels of supervision, including monocular (2D), multi-view static (3D) and multi-view dynamic (4D) datasets, to train a generalist spatiotemporal implicit representation capable of producing zero-shot novel videos from arbitrary camera locations and trajectories. We evaluate AnyView on standard benchmarks, showing competitive results with the current state of the art, and propose \textbf{AnyViewBench}, a challenging new benchmark tailored towards \emph{extreme} dynamic view synthesis in diverse real-world scenarios. In this more dramatic setting, we find that most baselines drastically degrade in performance, as they require significant overlap between viewpoints, while AnyView maintains the ability to produce realistic, plausible, and spatiotemporally consistent videos when prompted from \emph{any} viewpoint. Results, data, code, and models can be viewed at: https://tri-ml.github.io/AnyView/

  </details>



- **SyncLight: Controllable and Consistent Multi-View Relighting**  
  David Serrano-Lozano, Anand Bhattad, Luis Herranz, Jean-François Lalonde, Javier Vazquez-Corral  
  _2026-01-23_ · https://arxiv.org/abs/2601.16981v1  
  <details><summary>Abstract</summary>

  We present SyncLight, the first method to enable consistent, parametric relighting across multiple uncalibrated views of a static scene. While single-view relighting has advanced significantly, existing generative approaches struggle to maintain the rigorous lighting consistency essential for multi-camera broadcasts, stereoscopic cinema, and virtual production. SyncLight addresses this by enabling precise control over light intensity and color across a multi-view capture of a scene, conditioned on a single reference edit. Our method leverages a multi-view diffusion transformer trained using a latent bridge matching formulation, achieving high-fidelity relighting of the entire image set in a single inference step. To facilitate training, we introduce a large-scale hybrid dataset comprising diverse synthetic environments -- curated from existing sources and newly designed scenes -- alongside high-fidelity, real-world multi-view captures under calibrated illumination. Surprisingly, though trained only on image pairs, SyncLight generalizes zero-shot to an arbitrary number of viewpoints, effectively propagating lighting changes across all views, without requiring camera pose information. SyncLight enables practical relighting workflows for multi-view capture systems.

  </details>



- **Evaluating Large Vision-language Models for Surgical Tool Detection**  
  Nakul Poudel, Richard Simon, Cristian A. Linte  
  _2026-01-23_ · https://arxiv.org/abs/2601.16895v1  
  <details><summary>Abstract</summary>

  Surgery is a highly complex process, and artificial intelligence has emerged as a transformative force in supporting surgical guidance and decision-making. However, the unimodal nature of most current AI systems limits their ability to achieve a holistic understanding of surgical workflows. This highlights the need for general-purpose surgical AI systems capable of comprehensively modeling the interrelated components of surgical scenes. Recent advances in large vision-language models that integrate multimodal data processing offer strong potential for modeling surgical tasks and providing human-like scene reasoning and understanding. Despite their promise, systematic investigations of VLMs in surgical applications remain limited. In this study, we evaluate the effectiveness of large VLMs for the fundamental surgical vision task of detecting surgical tools. Specifically, we investigate three state-of-the-art VLMs, Qwen2.5, LLaVA1.5, and InternVL3.5, on the GraSP robotic surgery dataset under both zero-shot and parameter-efficient LoRA fine-tuning settings. Our results demonstrate that Qwen2.5 consistently achieves superior detection performance in both configurations among the evaluated VLMs. Furthermore, compared with the open-set detection baseline Grounding DINO, Qwen2.5 exhibits stronger zero-shot generalization and comparable fine-tuned performance. Notably, Qwen2.5 shows superior instrument recognition, while Grounding DINO demonstrates stronger localization.

  </details>



- **A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study**  
  Guangping Liu, Nicholas Hawkins, Billy Madden, Tipu Sultan, Flavio Esposito, Madi Babaiasl  
  _2026-01-23_ · https://arxiv.org/abs/2601.16870v1  
  <details><summary>Abstract</summary>

  Integrated control of wheelchairs and wheelchair-mounted robotic arms (WMRAs) has strong potential to increase independence for users with severe motor limitations, yet existing interfaces often lack the flexibility needed for intuitive assistive interaction. Although data-driven AI methods show promise, progress is limited by the lack of multimodal datasets that capture natural Human-Robot Interaction (HRI), particularly conversational ambiguity in dialogue-driven control. To address this gap, we propose a multimodal data collection framework that employs a dialogue-based interaction protocol and a two-room Wizard-of-Oz (WoZ) setup to simulate robot autonomy while eliciting natural user behavior. The framework records five synchronized modalities: RGB-D video, conversational audio, inertial measurement unit (IMU) signals, end-effector Cartesian pose, and whole-body joint states across five assistive tasks. Using this framework, we collected a pilot dataset of 53 trials from five participants and validated its quality through motion smoothness analysis and user feedback. The results show that the framework effectively captures diverse ambiguity types and supports natural dialogue-driven interaction, demonstrating its suitability for scaling to a larger dataset for learning, benchmarking, and evaluation of ambiguity-aware assistive control.

  </details>



- **ColorConceptBench: A Benchmark for Probabilistic Color-Concept Understanding in Text-to-Image Models**  
  Chenxi Ruan, Yu Xiao, Yihan Hou, Guosheng Hu, Wei Zeng  
  _2026-01-23_ · https://arxiv.org/abs/2601.16836v1  
  <details><summary>Abstract</summary>

  While text-to-image (T2I) models have advanced considerably, their capability to associate colors with implicit concepts remains underexplored. To address the gap, we introduce ColorConceptBench, a new human-annotated benchmark to systematically evaluate color-concept associations through the lens of probabilistic color distributions. ColorConceptBench moves beyond explicit color names or codes by probing how models translate 1,281 implicit color concepts using a foundation of 6,369 human annotations. Our evaluation of seven leading T2I models reveals that current models lack sensitivity to abstract semantics, and crucially, this limitation appears resistant to standard interventions (e.g., scaling and guidance). This demonstrates that achieving human-like color semantics requires more than larger models, but demands a fundamental shift in how models learn and represent implicit meaning.

  </details>



- **Incorporating Eye-Tracking Signals Into Multimodal Deep Visual Models For Predicting User Aesthetic Experience In Residential Interiors**  
  Chen-Ying Chien, Po-Chih Kuo  
  _2026-01-23_ · https://arxiv.org/abs/2601.16811v1  
  <details><summary>Abstract</summary>

  Understanding how people perceive and evaluate interior spaces is essential for designing environments that promote well-being. However, predicting aesthetic experiences remains difficult due to the subjective nature of perception and the complexity of visual responses. This study introduces a dual-branch CNN-LSTM framework that fuses visual features with eye-tracking signals to predict aesthetic evaluations of residential interiors. We collected a dataset of 224 interior design videos paired with synchronized gaze data from 28 participants who rated 15 aesthetic dimensions. The proposed model attains 72.2% accuracy on objective dimensions (e.g., light) and 66.8% on subjective dimensions (e.g., relaxation), outperforming state-of-the-art video baselines and showing clear gains on subjective evaluation tasks. Notably, models trained with eye-tracking retain comparable performance when deployed with visual input alone. Ablation experiments further reveal that pupil responses contribute most to objective assessments, while the combination of gaze and visual cues enhances subjective evaluations. These findings highlight the value of incorporating eye-tracking as privileged information during training, enabling more practical tools for aesthetic assessment in interior design.

  </details>



- **An Efficient Insect-inspired Approach for Visual Point-goal Navigation**  
  Lu Yihe, Barbara Webb  
  _2026-01-23_ · https://arxiv.org/abs/2601.16806v1  
  <details><summary>Abstract</summary>

  In this work we develop a novel insect-inspired agent for visual point-goal navigation. This combines abstracted models of two insect brain structures that have been implicated, respectively, in associative learning and path integration. We draw an analogy between the formal benchmark of the Habitat point-goal navigation task and the ability of insects to learn and refine visually guided paths around obstacles between a discovered food location and their nest. We demonstrate that the simple insect-inspired agent exhibits performance comparable to recent SOTA models at many orders of magnitude less computational cost. Testing in a more realistic simulated environment shows the approach is robust to perturbations.

  </details>



- **REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion**  
  Xuewei Li, Xinghan Bao, Zhimin Chen, Xi Li  
  _2026-01-23_ · https://arxiv.org/abs/2601.16788v1  
  <details><summary>Abstract</summary>

  As an important and challenging problem in computer vision, Panoramic Semantic Segmentation (PASS) aims to give complete scene perception based on an ultra-wide angle of view. Most PASS methods often focus on spherical geometry with RGB input or using the depth information in original or HHA format, which does not make full use of panoramic image geometry. To address these shortcomings, we propose REL-SF4PASS with our REL depth representation based on cylindrical coordinate and Spherical-dynamic Multi-Modal Fusion SMMF. REL is made up of Rectified Depth, Elevation-Gained Vertical Inclination Angle, and Lateral Orientation Angle, which fully represents 3D space in cylindrical coordinate style and the surface normal direction. SMMF aims to ensure the diversity of fusion for different panoramic image regions and reduce the breakage of cylinder side surface expansion in ERP projection, which uses different fusion strategies to match the different regions in panoramic images. Experimental results show that REL-SF4PASS considerably improves performance and robustness on popular benchmark, Stanford2D3D Panoramic datasets. It gains 2.35% average mIoU improvement on all 3 folds and reduces the performance variance by approximately 70% when facing 3D disturbance.

  </details>



- **Curated endoscopic retrograde cholangiopancreatography images dataset**  
  Alda João Andrade, Mónica Martins, André Ferreira, Tarcísio Araújo, Luís Lopes, Victor Alves  
  _2026-01-23_ · https://arxiv.org/abs/2601.16759v1  
  <details><summary>Abstract</summary>

  Endoscopic Retrograde Cholangiopancreatography (ERCP) is a key procedure in the diagnosis and treatment of biliary and pancreatic diseases. Artificial intelligence has been pointed as one solution to automatize diagnosis. However, public ERCP datasets are scarce, which limits the use of such approach. Therefore, this study aims to help fill this gap by providing a large and curated dataset. The collection is composed of 19.018 raw images and 19.317 processed from 1.602 patients. 5.519 images are labeled, which provides a ready to use dataset. All images were manually inspected and annotated by two gastroenterologist with more than 5 years of experience and reviewed by another gastroenterologist with more than 20 years of experience, all with more than 400 ERCP procedures annually. The utility and validity of the dataset is proven by a classification experiment. This collection aims to provide or contribute for a benchmark in automatic ERCP analysis and diagnosis of biliary and pancreatic diseases.

  </details>



- **CER-HV: A CER-Based Human-in-the-Loop Framework for Cleaning Datasets Applied to Arabic-Script HTR**  
  Sana Al-azzawi, Elisa Barney, Marcus Liwicki  
  _2026-01-23_ · https://arxiv.org/abs/2601.16713v1  
  <details><summary>Abstract</summary>

  Handwritten text recognition (HTR) for Arabic-script languages still lags behind Latin-script HTR, despite recent advances in model architectures, datasets, and benchmarks. We show that data quality is a significant limiting factor in many published datasets and propose CER-HV (CER-based Ranking with Human Verification) as a framework to detect and clean label errors. CER-HV combines a CER-based noise detector, built on a carefully configured Convolutional Recurrent Neural Network (CRNN) with early stopping to avoid overfitting noisy samples, and a human-in-the-loop (HITL) step that verifies high-ranking samples. The framework reveals that several existing datasets contain previously underreported problems, including transcription, segmentation, orientation, and non-text content errors. These have been identified with up to 90 percent precision in the Muharaf and 80-86 percent in the PHTI datasets. We also show that our CRNN achieves state-of-the-art performance across five of the six evaluated datasets, reaching 8.45 percent Character Error Rate (CER) on KHATT (Arabic), 8.26 percent on PHTI (Pashto), 10.66 percent on Ajami, and 10.11 percent on Muharaf (Arabic), all without any data cleaning. We establish a new baseline of 11.3 percent CER on the PHTD (Persian) dataset. Applying CER-HV improves the evaluation CER by 0.3-0.6 percent on the cleaner datasets and 1.0-1.8 percent on the noisier ones. Although our experiments focus on documents written in an Arabic-script language, including Arabic, Persian, Urdu, Ajami, and Pashto, the framework is general and can be applied to other text recognition datasets.

  </details>



- **EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents**  
  Xinze Li, Ziyue Zhu, Siyuan Liu, Yubo Ma, Yuhang Zang, Yixin Cao, Aixin Sun  
  _2026-01-23_ · https://arxiv.org/abs/2601.16690v1  
  <details><summary>Abstract</summary>

  We introduce EMemBench, a programmatic benchmark for evaluating long-term memory of agents through interactive games. Rather than using a fixed set of questions, EMemBench generates questions from each agent's own trajectory, covering both text and visual game environments. Each template computes verifiable ground truth from underlying game signals, with controlled answerability and balanced coverage over memory skills: single/multi-hop recall, induction, temporal, spatial, logical, and adversarial. We evaluate memory agents with strong LMs/VLMs as backbones, using in-context prompting as baselines. Across 15 text games and multiple visual seeds, results are far from saturated: induction and spatial reasoning are persistent bottlenecks, especially in visual setting. Persistent memory yields clear gains for open backbones on text games, but improvements are less consistent for VLM agents, suggesting that visually grounded episodic memory remains an open challenge. A human study further confirms the difficulty of EMemBench.

  </details>



- **ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction**  
  Ming Li, Hui Shan, Kai Zheng, Chentao Shen, Siyu Liu, Yanwei Fu, Zhen Chen, Xiangru Huang  
  _2026-01-23_ · https://arxiv.org/abs/2601.16672v1  
  <details><summary>Abstract</summary>

  High-quality 3D garment reconstruction plays a crucial role in mitigating the sim-to-real gap in applications such as digital avatars, virtual try-on and robotic manipulation. However, existing garment reconstruction methods typically rely on unstructured representations, such as 3D Gaussian Splats, struggling to provide accurate reconstructions of garment topology and sewing structures. As a result, the reconstructed outputs are often unsuitable for high-fidelity physical simulation. We propose ReWeaver, a novel framework for topology-accurate 3D garment and sewing pattern reconstruction from sparse multi-view RGB images. Given as few as four input views, ReWeaver predicts seams and panels as well as their connectivities in both the 2D UV space and the 3D space. The predicted seams and panels align precisely with the multi-view images, yielding structured 2D--3D garment representations suitable for 3D perception, high-fidelity physical simulation, and robotic manipulation. To enable effective training, we construct a large-scale dataset GCD-TS, comprising multi-view RGB images, 3D garment geometries, textured human body meshes and annotated sewing patterns. The dataset contains over 100,000 synthetic samples covering a wide range of complex geometries and topologies. Extensive experiments show that ReWeaver consistently outperforms existing methods in terms of topology accuracy, geometry alignment and seam-panel consistency.

  </details>



- **ReViP: Reducing False Completion in Vision-Language-Action Models with Vision-Proprioception Rebalance**  
  Zhuohao Li, Yinghao Li, Jian-Jian Jiang, Lang Zhou, Tianyu Zhang, Wei-Shi Zheng  
  _2026-01-23_ · https://arxiv.org/abs/2601.16667v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have advanced robotic manipulation by combining vision, language, and proprioception to predict actions. However, previous methods fuse proprioceptive signals directly with VLM-encoded vision-language features, resulting in state-dominant bias and false completions despite visible execution failures. We attribute this to modality imbalance, where policies over-rely on internal state while underusing visual evidence. To address this, we present ReViP, a novel VLA framework with Vision-Proprioception Rebalance to enhance visual grounding and robustness under perturbations. The key insight is to introduce auxiliary task-aware environment priors to adaptively modulate the coupling between semantic perception and proprioceptive dynamics. Specifically, we use an external VLM as a task-stage observer to extract real-time task-centric visual cues from visual observations, which drive a Vision-Proprioception Feature-wise Linear Modulation to enhance environmental awareness and reduce state-driven errors. Moreover, to evaluate false completion, we propose the first False-Completion Benchmark Suite built on LIBERO with controlled settings such as Object-Drop. Extensive experiments show that ReViP effectively reduces false-completion rates and improves success rates over strong VLA baselines on our suite, with gains extending to LIBERO, RoboTwin 2.0, and real-world evaluations.

  </details>



- **PanopMamba: Vision State Space Modeling for Nuclei Panoptic Segmentation**  
  Ming Kang, Fung Fung Ting, Raphaël C. -W. Phan, Zongyuan Ge, Chee-Ming Ting  
  _2026-01-23_ · https://arxiv.org/abs/2601.16631v1  
  <details><summary>Abstract</summary>

  Nuclei panoptic segmentation supports cancer diagnostics by integrating both semantic and instance segmentation of different cell types to analyze overall tissue structure and individual nuclei in histopathology images. Major challenges include detecting small objects, handling ambiguous boundaries, and addressing class imbalance. To address these issues, we propose PanopMamba, a novel hybrid encoder-decoder architecture that integrates Mamba and Transformer with additional feature-enhanced fusion via state space modeling. We design a multiscale Mamba backbone and a State Space Model (SSM)-based fusion network to enable efficient long-range perception in pyramid features, thereby extending the pure encoder-decoder framework while facilitating information sharing across multiscale features of nuclei. The proposed SSM-based feature-enhanced fusion integrates pyramid feature networks and dynamic feature enhancement across different spatial scales, enhancing the feature representation of densely overlapping nuclei in both semantic and spatial dimensions. To the best of our knowledge, this is the first Mamba-based approach for panoptic segmentation. Additionally, we introduce alternative evaluation metrics, including image-level Panoptic Quality ($i$PQ), boundary-weighted PQ ($w$PQ), and frequency-weighted PQ ($fw$PQ), which are specifically designed to address the unique challenges of nuclei segmentation and thereby mitigate the potential bias inherent in vanilla PQ. Experimental evaluations on two multiclass nuclei segmentation benchmark datasets, MoNuSAC2020 and NuInsSeg, demonstrate the superiority of PanopMamba for nuclei panoptic segmentation over state-of-the-art methods. Consequently, the robustness of PanopMamba is validated across various metrics, while the distinctiveness of PQ variants is also demonstrated. Code is available at https://github.com/mkang315/PanopMamba.

  </details>



- **SCHIGAND: A Synthetic Facial Generation Mode Pipeline**  
  Ananya Kadali, Sunnie Jehan-Morrison, Orasiki Wellington, Barney Evans, Precious Durojaiye, Richard Guest  
  _2026-01-23_ · https://arxiv.org/abs/2601.16627v1  
  <details><summary>Abstract</summary>

  The growing demand for diverse and high-quality facial datasets for training and testing biometric systems is challenged by privacy regulations, data scarcity, and ethical concerns. Synthetic facial images offer a potential solution, yet existing generative models often struggle to balance realism, diversity, and identity preservation. This paper presents SCHIGAND, a novel synthetic face generation pipeline integrating StyleCLIP, HyperStyle, InterfaceGAN, and Diffusion models to produce highly realistic and controllable facial datasets. SCHIGAND enhances identity preservation while generating realistic intra-class variations and maintaining inter-class distinctiveness, making it suitable for biometric testing. The generated datasets were evaluated using ArcFace, a leading facial verification model, to assess their effectiveness in comparison to real-world facial datasets. Experimental results demonstrate that SCHIGAND achieves a balance between image quality and diversity, addressing key limitations of prior generative models. This research highlights the potential of SCHIGAND to supplement and, in some cases, replace real data for facial biometric applications, paving the way for privacy-compliant and scalable solutions in synthetic dataset generation.

  </details>



- **Zero-Shot MARL Benchmark in the Cyber-Physical Mobility Lab**  
  Julius Beerwerth, Jianye Xu, Simon Schäfer, Fynn Belderink, Bassam Alrifaee  
  _2026-01-23_ · https://arxiv.org/abs/2601.16578v1  
  <details><summary>Abstract</summary>

  We present a reproducible benchmark for evaluating sim-to-real transfer of Multi-Agent Reinforcement Learning (MARL) policies for Connected and Automated Vehicles (CAVs). The platform, based on the Cyber-Physical Mobility Lab (CPM Lab) [1], integrates simulation, a high-fidelity digital twin, and a physical testbed, enabling structured zero-shot evaluation of MARL motion-planning policies. We demonstrate its use by deploying a SigmaRL-trained policy [2] across all three domains, revealing two complementary sources of performance degradation: architectural differences between simulation and hardware control stacks, and the sim-to-real gap induced by increasing environmental realism. The open-source setup enables systematic analysis of sim-to-real challenges in MARL under realistic, reproducible conditions.

  </details>



- **Understanding and Improving UMAP with Geometric and Topological Priors: The JORC-UMAP Algorithm**  
  Xiaobin Li, Run Zhang  
  _2026-01-23_ · https://arxiv.org/abs/2601.16552v1  
  <details><summary>Abstract</summary>

  Nonlinear dimensionality reduction techniques, particularly UMAP, are widely used for visualizing high-dimensional data. However, UMAP's local Euclidean distance assumption often fails to capture intrinsic manifold geometry, leading to topological tearing and structural collapse. We identify UMAP's sensitivity to the k-nearest neighbor graph as a key cause. To address this, we introduce Ollivier-Ricci curvature as a geometric prior, reinforcing edges at geometric bottlenecks and reducing redundant links. Since curvature estimation is noise-sensitive, we also incorporate a topological prior using Jaccard similarity to ensure neighborhood consistency. The resulting method, JORC-UMAP, better distinguishes true manifold structure from spurious connections. Experiments on synthetic and real-world datasets show that JORC-UMAP reduces tearing and collapse more effectively than standard UMAP and other DR methods, as measured by SVM accuracy and triplet preservation scores, while maintaining computational efficiency. This work offers a geometry-aware enhancement to UMAP for more faithful data visualization.

  </details>



- **Semi-Supervised Hierarchical Open-Set Classification**  
  Erik Wallin, Fredrik Kahl, Lars Hammarstrand  
  _2026-01-23_ · https://arxiv.org/abs/2601.16541v1  
  <details><summary>Abstract</summary>

  Hierarchical open-set classification handles previously unseen classes by assigning them to the most appropriate high-level category in a class taxonomy. We extend this paradigm to the semi-supervised setting, enabling the use of large-scale, uncurated datasets containing a mixture of known and unknown classes to improve the hierarchical open-set performance. To this end, we propose a teacher-student framework based on pseudo-labeling. Two key components are introduced: 1) subtree pseudo-labels, which provide reliable supervision in the presence of unknown data, and 2) age-gating, a mechanism that mitigates overconfidence in pseudo-labels. Experiments show that our framework outperforms self-supervised pretraining followed by supervised adaptation, and even matches the fully supervised counterpart when using only 20 labeled samples per class on the iNaturalist19 benchmark. Our code is available at https://github.com/walline/semihoc.

  </details>



- **TangramPuzzle: Evaluating Multimodal Large Language Models with Compositional Spatial Reasoning**  
  Daixian Liu, Jiayi Kuang, Yinghui Li, Yangning Li, Di Yin, Haoyu Cao, Xing Sun, Ying Shen, Hai-Tao Zheng, Liang Lin, et al.  
  _2026-01-23_ · https://arxiv.org/abs/2601.16520v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have achieved remarkable progress in visual recognition and semantic understanding. Nevertheless, their ability to perform precise compositional spatial reasoning remains largely unexplored. Existing benchmarks often involve relatively simple tasks and rely on semantic approximations or coarse relative positioning, while their evaluation metrics are typically limited and lack rigorous mathematical formulations. To bridge this gap, we introduce TangramPuzzle, a geometry-grounded benchmark designed to evaluate compositional spatial reasoning through the lens of the classic Tangram game. We propose the Tangram Construction Expression (TCE), a symbolic geometric framework that grounds tangram assemblies in exact, machine-verifiable coordinate specifications, to mitigate the ambiguity of visual approximation. We design two complementary tasks: Outline Prediction, which demands inferring global shapes from local components, and End-to-End Code Generation, which requires solving inverse geometric assembly problems. We conduct extensive evaluation experiments on advanced open-source and proprietary models, revealing an interesting insight: MLLMs tend to prioritize matching the target silhouette while neglecting geometric constraints, leading to distortions or deformations of the pieces.

  </details>



- **Expert Knowledge-Guided Decision Calibration for Accurate Fine-Grained Tree Species Classification**  
  Chen Long, Dian Chen, Ruifei Ding, Zhe Chen, Zhen Dong, Bisheng Yang  
  _2026-01-23_ · https://arxiv.org/abs/2601.16498v1  
  <details><summary>Abstract</summary>

  Accurate fine-grained tree species classification is critical for forest inventory and biodiversity monitoring. Existing methods predominantly focus on designing complex architectures to fit local data distributions. However, they often overlook the long-tailed distributions and high inter-class similarity inherent in limited data, thereby struggling to distinguish between few-shot or confusing categories. In the process of knowledge dissemination in the human world, individuals will actively seek expert assistance to transcend the limitations of local thinking. Inspired by this, we introduce an external "Domain Expert" and propose an Expert Knowledge-Guided Classification Decision Calibration Network (EKDC-Net) to overcome these challenges. Our framework addresses two core issues: expert knowledge extraction and utilization. Specifically, we first develop a Local Prior Guided Knowledge Extraction Module (LPKEM). By leveraging Class Activation Map (CAM) analysis, LPKEM guides the domain expert to focus exclusively on discriminative features essential for classification. Subsequently, to effectively integrate this knowledge, we design an Uncertainty-Guided Decision Calibration Module (UDCM). This module dynamically corrects the local model's decisions by considering both overall category uncertainty and instance-level prediction uncertainty. Furthermore, we present a large-scale classification dataset covering 102 tree species, named CU-Tree102 to address the issue of scarce diversity in current benchmarks. Experiments on three benchmark datasets demonstrate that our approach achieves state-of-the-art performance. Crucially, as a lightweight plug-and-play module, EKDC-Net improves backbone accuracy by 6.42% and precision by 11.46% using only 0.08M additional learnable parameters. The dataset, code, and pre-trained models are available at https://github.com/WHU-USI3DV/TreeCLS.

  </details>


