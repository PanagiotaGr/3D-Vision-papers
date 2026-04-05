# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-05 07:21 UTC_

Total papers shown: **43**


---

- **ActionParty: Multi-Subject Action Binding in Generative Video Games**  
  Alexander Pondaven, Ziyi Wu, Igor Gilitschenski, Philip Torr, Sergey Tulyakov, Fabio Pizzati, Aliaksandr Siarohin  
  _2026-04-02_ · https://arxiv.org/abs/2604.02330v1  
  <details><summary>Abstract</summary>

  Recent advances in video diffusion have enabled the development of "world models" capable of simulating interactive environments. However, these models are largely restricted to single-agent settings, failing to control multiple agents simultaneously in a scene. In this work, we tackle a fundamental issue of action binding in existing video diffusion models, which struggle to associate specific actions with their corresponding subjects. For this purpose, we propose ActionParty, an action controllable multi-subject world model for generative video games. It introduces subject state tokens, i.e. latent variables that persistently capture the state of each subject in the scene. By jointly modeling state tokens and video latents with a spatial biasing mechanism, we disentangle global video frame rendering from individual action-controlled subject updates. We evaluate ActionParty on the Melting Pot benchmark, demonstrating the first video world model capable of controlling up to seven players simultaneously across 46 diverse environments. Our results show significant improvements in action-following accuracy and identity consistency, while enabling robust autoregressive tracking of subjects through complex interactions.

  </details>



- **Generative World Renderer**  
  Zheng-Hui Huang, Zhixiang Wang, Jiaming Tan, Ruihan Yu, Yidan Zhang, Bo Zheng, Yu-Lun Liu, Yung-Yu Chuang, Kaipeng Zhang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02329v1  
  <details><summary>Abstract</summary>

  Scaling generative inverse and forward rendering to real-world scenarios is bottlenecked by the limited realism and temporal coherence of existing synthetic datasets. To bridge this persistent domain gap, we introduce a large-scale, dynamic dataset curated from visually complex AAA games. Using a novel dual-screen stitched capture method, we extracted 4M continuous frames (720p/30 FPS) of synchronized RGB and five G-buffer channels across diverse scenes, visual effects, and environments, including adverse weather and motion-blur variants. This dataset uniquely advances bidirectional rendering: enabling robust in-the-wild geometry and material decomposition, and facilitating high-fidelity G-buffer-guided video generation. Furthermore, to evaluate the real-world performance of inverse rendering without ground truth, we propose a novel VLM-based assessment protocol measuring semantic, spatial, and temporal consistency. Experiments demonstrate that inverse renderers fine-tuned on our data achieve superior cross-dataset generalization and controllable generation, while our VLM evaluation strongly correlates with human judgment. Combined with our toolkit, our forward renderer enables users to edit styles of AAA games from G-buffers using text prompts.

  </details>



- **Modulate-and-Map: Crossmodal Feature Mapping with Cross-View Modulation for 3D Anomaly Detection**  
  Alex Costanzino, Pierluigi Zama Ramirez, Giuseppe Lisanti, Luigi Di Stefano  
  _2026-04-02_ · https://arxiv.org/abs/2604.02328v1  
  <details><summary>Abstract</summary>

  We present ModMap, a natively multiview and multimodal framework for 3D anomaly detection and segmentation. Unlike existing methods that process views independently, our method draws inspiration from the crossmodal feature mapping paradigm to learn to map features across both modalities and views, while explicitly modelling view-dependent relationships through feature-wise modulation. We introduce a cross-view training strategy that leverages all possible view combinations, enabling effective anomaly scoring through multiview ensembling and aggregation. To process high-resolution 3D data, we train and publicly release a foundational depth encoder tailored to industrial datasets. Experiments on SiM3D, a recent benchmark that introduces the first multiview and multimodal setup for 3D anomaly detection and segmentation, demonstrate that ModMap attains state-of-the-art performance by surpassing previous methods by wide margins.

  </details>



- **Beyond Referring Expressions: Scenario Comprehension Visual Grounding**  
  Ruozhen He, Nisarg A. Shah, Qihua Dong, Zilin Xiao, Jaywon Koo, Vicente Ordonez  
  _2026-04-02_ · https://arxiv.org/abs/2604.02323v1  
  <details><summary>Abstract</summary>

  Existing visual grounding benchmarks primarily evaluate alignment between image regions and literal referring expressions, where models can often succeed by matching a prominent named category. We explore a complementary and more challenging setting of scenario-based visual grounding, where the target must be inferred from roles, intentions, and relational context rather than explicit naming. We introduce Referring Scenario Comprehension (RSC), a benchmark designed for this setting. The queries in this benchmark are paragraph-length texts that describe object roles, user goals, and contextual cues, including deliberate references to distractor objects that often require deep understanding to resolve. Each instance is annotated with interpretable difficulty tags for uniqueness, clutter, size, overlap, and position which expose distinct failure modes and support fine-grained analysis. RSC contains approximately 31k training examples, 4k in-domain test examples, and a 3k out-of-distribution split with unseen object categories. We further propose ScenGround, a curriculum reasoning method serving as a reference point for this setting, combining supervised warm-starting with difficulty-aware reinforcement learning. Experiments show that scenario-based queries expose systematic failures in current models that standard benchmarks do not reveal, and that curriculum training improves performance on challenging slices and transfers to standard benchmarks.

  </details>



- **VOID: Video Object and Interaction Deletion**  
  Saman Motamed, William Harvey, Benjamin Klein, Luc Van Gool, Zhuoning Yuan, Ta-Ying Cheng  
  _2026-04-02_ · https://arxiv.org/abs/2604.02296v1  
  <details><summary>Abstract</summary>

  Existing video object removal methods excel at inpainting content "behind" the object and correcting appearance-level artifacts such as shadows and reflections. However, when the removed object has more significant interactions, such as collisions with other objects, current models fail to correct them and produce implausible results. We present VOID, a video object removal framework designed to perform physically-plausible inpainting in these complex scenarios. To train the model, we generate a new paired dataset of counterfactual object removals using Kubric and HUMOTO, where removing an object requires altering downstream physical interactions. During inference, a vision-language model identifies regions of the scene affected by the removed object. These regions are then used to guide a video diffusion model that generates physically consistent counterfactual outcomes. Experiments on both synthetic and real data show that our approach better preserves consistent scene dynamics after object removal compared to prior video object removal methods. We hope this framework sheds light on how to make video editing models better simulators of the world through high-level causal reasoning.

  </details>



- **Deep Neural Network Based Roadwork Detection for Autonomous Driving**  
  Sebastian Wullrich, Nicolai Steinke, Daniel Goehring  
  _2026-04-02_ · https://arxiv.org/abs/2604.02282v1  
  <details><summary>Abstract</summary>

  Road construction sites create major challenges for both autonomous vehicles and human drivers due to their highly dynamic and heterogeneous nature. This paper presents a real-time system that detects and localizes roadworks by combining a YOLO neural network with LiDAR data. The system identifies individual roadwork objects while driving, merges them into coherent construction sites and records their outlines in world coordinates. The model training was based on an adapted US dataset and a new dataset collected from test drives with a prototype vehicle in Berlin, Germany. Evaluations on real-world road construction sites showed a localization accuracy below 0.5 m. The system can support traffic authorities with up-to-date roadwork data and could enable autonomous vehicles to navigate construction sites more safely in the future.

  </details>



- **A virtual-variable-length method for robust inverse kinematics of multi-segment continuum robots**  
  Weiting Feng, Federico Renda, Yunjie Yang, Francesco Giorgio-Serchi  
  _2026-04-02_ · https://arxiv.org/abs/2604.02256v1  
  <details><summary>Abstract</summary>

  This paper proposes a new, robust method to solve the inverse kinematics (IK) of multi-segment continuum manipulators. Conventional Jacobian-based solvers, especially when initialized from neutral/rest configurations, often exhibit slow convergence and, in certain conditions, may fail to converge (deadlock). The Virtual-Variable-Length (VVL) method proposed here introduces fictitious variations of segments' length during the solution iteration, conferring virtual axial degrees of freedom that alleviate adverse behaviors and constraints, thus enabling or accelerating convergence. Comprehensive numerical experiments were conducted to compare the VVL method against benchmark Jacobian-based and Damped Least Square IK solvers. Across more than $1.8\times 10^6$ randomized trials covering manipulators with two to seven segments, the proposed approach achieved up to a 20$\%$ increase in convergence success rate over the benchmark and a 40-80$\%$ reduction in average iteration count under equivalent accuracy thresholds ($10^{-4}-10^{-8}$). While deadlocks are not restricted to workspace boundaries and may occur at arbitrary poses, our empirical study identifies boundary-proximal configurations as a frequent cause of failed convergence and the VVL method mitigates such occurrences over a statistical sample of test cases.

  </details>



- **UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models**  
  Qiyao Zhang, Shuhua Zheng, Jianli Sun, Chengxiang Li, Xianke Wu, Zihan Song, Zhiyong Cui, Yisheng Lv, Yonglin Tian  
  _2026-04-02_ · https://arxiv.org/abs/2604.02241v1  
  <details><summary>Abstract</summary>

  Embodied visual tracking is crucial for Unmanned Aerial Vehicles (UAVs) executing complex real-world tasks. In dynamic urban scenarios with complex semantic requirements, Vision-Language-Action (VLA) models show great promise due to their cross-modal fusion and continuous action generation capabilities. To benchmark multimodal tracking in such environments, we construct a dedicated evaluation benchmark and a large-scale dataset encompassing over 890K frames, 176 tasks, and 85 diverse objects. Furthermore, to address temporal feature redundancy and the lack of spatial geometric priors in existing VLA models, we propose an improved VLA tracking model, UAV-Track VLA. Built upon the $π_{0.5}$ architecture, our model introduces a temporal compression net to efficiently capture inter-frame dynamics. Additionally, a parallel dual-branch decoder comprising a spatial-aware auxiliary grounding head and a flow matching action expert is designed to decouple cross-modal features and generate fine-grained continuous actions. Systematic experiments in the CARLA simulator validate the superior end-to-end performance of our method. Notably, in challenging long-distance pedestrian tracking tasks, UAV-Track VLA achieves a 61.76\% success rate and 269.65 average tracking frames, significantly outperforming existing baselines. Furthermore, it demonstrates robust zero-shot generalization in unseen environments and reduces single-step inference latency by 33.4\% (to 0.0571s) compared to the original $π_{0.5}$, enabling highly efficient, real-time UAV control. Data samples and demonstration videos are available at: https://github.com/Hub-Tian/UAV-Track\_VLA.

  </details>



- **Lightweight Spatiotemporal Highway Lane Detection via 3D-ResNet and PINet with ROI-Aware Attention**  
  Sorna Shanmuga Raja, Abdelhafid Zenati  
  _2026-04-02_ · https://arxiv.org/abs/2604.02188v1  
  <details><summary>Abstract</summary>

  This paper presents a lightweight, end-to-end highway lane detection architecture that jointly captures spatial and temporal information for robust performance in real-world driving scenarios. Building on the strengths of 3D convolutional neural networks and instance segmentation, we propose two models that integrate a 3D-ResNet encoder with a Point Instance Network (PINet) decoder. The first model enhances multi-scale feature representation using a Feature Pyramid Network (FPN) and Self-Attention mechanism to refine spatial dependencies. The second model introduces a Region of Interest (ROI) detection head to selectively focus on lane-relevant regions, thereby improving precision and reducing computational complexity. Experiments conducted on the TuSimple dataset (highway driving scenarios) demonstrate that the proposed second model achieves 93.40% accuracy while significantly reducing false negatives. Compared to existing 2D and 3D baselines, our approach achieves improved performance with fewer parameters and reduced latency. The architecture has been validated through offline training and real-time inference in the Autonomous Systems Laboratory at City, St George's University of London. These results suggest that the proposed models are well-suited for integration into Advanced Driver Assistance Systems (ADAS), with potential scalability toward full Lane Assist Systems (LAS).

  </details>



- **Reflection Generation for Composite Image Using Diffusion Model**  
  Haonan Zhao, Qingyang Liu, Jiaxuan Chen, Li Niu  
  _2026-04-02_ · https://arxiv.org/abs/2604.02168v1  
  <details><summary>Abstract</summary>

  Image composition involves inserting a foreground object into the background while synthesizing environment-consistent effects such as shadows and reflections. Although shadow generation has been extensively studied, reflection generation remains largely underexplored. In this work, we focus on reflection generation. We inject the prior information of reflection placement and reflection appearance into foundation diffusion model. We also divide reflections into two types and adopt type-aware model design. To support training, we construct the first large-scale object reflection dataset DEROBA. Experiments demonstrate that our method generates reflections that are physically coherent and visually realistic, establishing a new benchmark for reflection generation.

  </details>



- **Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation**  
  Saurabh Hinduja, Gurmeet Kaur, Maneesh Bilalpur, Jeffrey Cohn, Shaun Canavan  
  _2026-04-02_ · https://arxiv.org/abs/2604.02162v1  
  <details><summary>Abstract</summary>

  Subject-exclusive cross-validation is the standard evaluation protocol for facial Action Unit (AU) detection, yet reported improvements are often small. We show that cross-validation itself introduces measurable stochastic variance. On BP4D+, repeated 3-fold subject-exclusive splits produce an empirical noise floor of $\pm 0.065$ in average F1, with substantially larger variation for low-prevalence AUs. Operating-point metrics such as F1 fluctuate more than threshold-independent measures such as AUC, and model ranking can change under different fold assignments. We further evaluate cross-dataset robustness using a Leave-One-Dataset-Out (LODO) protocol across five AU datasets. LODO removes partition randomness and exposes domain-level instability that is not visible under single-dataset cross-validation. Together, these results suggest that gains often reported in cross-fold validation may fall within protocol variance. Leave-one-dataset-out cross-validation yields more stable and interpretable findings

  </details>



- **HyVGGT-VO: Tightly Coupled Hybrid Dense Visual Odometry with Feed-Forward Models**  
  Junxiang Pan, Lipu Zhou, Baojie Chen  
  _2026-04-02_ · https://arxiv.org/abs/2604.02107v1  
  <details><summary>Abstract</summary>

  Dense visual odometry (VO), which provides pose estimation and dense 3D reconstruction, serves as the cornerstone for applications ranging from robotics to augmented reality. Recently, feed-forward models have demonstrated remarkable capabilities in dense mapping. However, when these models are used in dense visual SLAM systems, their heavy computational burden restricts them to yielding sparse pose outputs at keyframes while still failing to achieve real-time pose estimation. In contrast, traditional sparse methods provide high computational efficiency and high-frequency pose outputs, but lack the capability for dense reconstruction. To address these limitations, we propose HyVGGT-VO, a novel framework that combines the computational efficiency of sparse VO with the dense reconstruction capabilities of feed-forward models. To the best of our knowledge, this is the first work to tightly couple a traditional VO framework with VGGT, a state-of-the-art feed-forward model. Specifically, we design an adaptive hybrid tracking frontend that dynamically switches between traditional optical flow and the VGGT tracking head to ensure robustness. Furthermore, we introduce a hierarchical optimization framework that jointly refines VO poses and the scale of VGGT predictions to ensure global scale consistency. Our approach achieves an approximately 5x processing speedup compared to existing VGGT-based methods, while reducing the average trajectory error by 85% on the indoor EuRoC dataset and 12% on the outdoor KITTI benchmark. Our code will be publicly available upon acceptance. Project page: https://geneta2580.github.io/HyVGGT-VO.io.

  </details>



- **CASHG: Context-Aware Stylized Online Handwriting Generation**  
  Jinsu Shin, Sungeun Hong, Jin Yeong Bak  
  _2026-04-02_ · https://arxiv.org/abs/2604.02103v1  
  <details><summary>Abstract</summary>

  Online handwriting represents strokes as time-ordered trajectories, which makes handwritten content easier to transform and reuse in a wide range of applications. However, generating natural sentence-level online handwriting that faithfully reflects a writer's style remains challenging, since sentence synthesis demands context-dependent characters with stroke continuity and spacing. Prior methods treat these boundary properties as implicit outcomes of sequence modeling, which becomes unreliable at the sentence scale and under limited compositional diversity. We propose CASHG, a context-aware stylized online handwriting generator that explicitly models inter-character connectivity for style-consistent sentence-level trajectory synthesis. CASHG uses a Character Context Encoder to obtain character identity and sentence-dependent context memory and fuses them in a bigram-aware sliding-window Transformer decoder that emphasizes local predecessor--current transitions, complemented by gated context fusion for sentence-level context.Training proceeds through a three-stage curriculum from isolated glyphs to full sentences, improving robustness under sparse transition coverage. We further introduce Connectivity and Spacing Metrics (CSM), a boundary-aware evaluation suite that quantifies cursive connectivity and spacing similarity. Under benchmark-matched evaluation protocols, CASHG consistently improves CSM over comparison methods while remaining competitive in DTW-based trajectory similarity, with gains corroborated by a human evaluation.

  </details>



- **LatentUM: Unleashing the Potential of Interleaved Cross-Modal Reasoning via a Latent-Space Unified Model**  
  Jiachun Jin, Zetong Zhou, Xiao Yang, Hao Zhang, Pengfei Liu, Jun Zhu, Zhijie Deng  
  _2026-04-02_ · https://arxiv.org/abs/2604.02097v1  
  <details><summary>Abstract</summary>

  Unified models (UMs) hold promise for their ability to understand and generate content across heterogeneous modalities. Compared to merely generating visual content, the use of UMs for interleaved cross-modal reasoning is more promising and valuable, e.g., for solving understanding problems that require dense visual thinking, improving visual generation through self-reflection, or modeling visual dynamics of the physical world guided by stepwise action interventions. However, existing UMs necessitate pixel decoding as a bridge due to their disjoint visual representations for understanding and generation, which is both ineffective and inefficient. In this paper, we introduce LatentUM, a novel unified model that represents all modalities within a shared semantic latent space, eliminating the need for pixel-space mediation between visual understanding and generation. This design naturally enables flexible interleaved cross-modal reasoning and generation. Beyond improved computational efficiency, the shared representation substantially alleviates codec bias and strengthens cross-modal alignment, allowing LatentUM to achieve state-of-the-art performance on the Visual Spatial Planning benchmark, push the limits of visual generation through self-reflection, and support world modeling by predicting future visual states within the shared semantic latent space.

  </details>



- **Center-Aware Detection with Swin-based Co-DETR Framework for Cervical Cytology**  
  Yan Kong, Yuan Yin, Hongan Chen, Yuqi Fang, Caifeng Shan  
  _2026-04-02_ · https://arxiv.org/abs/2604.02090v1  
  <details><summary>Abstract</summary>

  Automated analysis of Pap smear images is critical for cervical cancer screening but remains challenging due to dense cell distribution and complex morphology. In this paper, we present our winning solution for the RIVA Cervical Cytology Challenge, achieving 1st place in Track B and 2nd place in Track A. Our approach leverages a powerful baseline, integrating the Co-DINO framework with a Swin-Large backbone for robust multi-scale feature extraction. To address the dataset's unique fixed-size bounding box annotations, we formulate the detection task as a center-point prediction problem. Tailoring our approach to this formulation, we introduce a center-preserving data augmentation strategy and an analytical geometric box optimization to effectively absorb localization jitter. Finally, we apply track-specific loss tuning to adapt the loss weights for each task. Experiments demonstrate that our targeted optimizations improve detection performance, providing an effective pipeline for cytology image analysis. Our code is available at https://github.com/YanKong0408/Center-DETR.

  </details>



- **PLUME: Latent Reasoning Based Universal Multimodal Embedding**  
  Chenwei He, Xiangzhao Hao, Tianyu Yang, Yuxiang Ma, Yuheng Jia, Lingxiang Wu, Chaoyang Zhao, Haiyun Guo, Jinqiao Wang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02073v1  
  <details><summary>Abstract</summary>

  Universal multimodal embedding (UME) maps heterogeneous inputs into a shared retrieval space with a single model. Recent approaches improve UME by generating explicit chain-of-thought (CoT) rationales before extracting embeddings, enabling multimodal large language models to better infer complex query intent. However, explicit CoT incurs substantial inference overhead and can compress rich multimodal evidence into a narrow textual bottleneck. We propose PLUME, a latent reasoning framework that advances UME by replacing verbalized CoT with a short autoregressive rollout of continuous latent states. To support diverse multimodal queries, PLUME further introduces a semantic-anchor-guided transition adapter that steers latent rollout along different reasoning trajectories under the same fixed computation budget. To stabilize training, PLUME adopts a progressive explicit-to-latent curriculum that uses verbalized reasoning only as a temporary training scaffold and gradually transfers this behavior into hidden-state computation, eliminating explicit CoT at inference. On the 78-task MMEB-v2 benchmark, PLUME outperforms strong explicit-CoT UME baselines while reducing reasoning from hundreds of generated tokens to fewer than 10 latent steps, delivering over 30x faster inference. PLUME is especially well suited to retrieval settings where relevant evidence is dense, structurally complex, and difficult to organize through verbalized intermediate rationales, such as video and visual document retrieval. These results show that structured latent computation can preserve the benefits of intermediate reasoning without the overhead of explicit rationale generation, providing a stronger and more efficient paradigm for practical retrieval systems.

  </details>



- **CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects**  
  Jingliang Li, Jindou Jia, Tuo An, Chuhao Zhou, Xiangyu Chen, Shilin Shan, Boyu Ma, Bofan Lyu, Gen Li, Jianfei Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02060v1  
  <details><summary>Abstract</summary>

  When told to "cut the apple," a robot must choose the knife over nearby scissors, despite both objects affording the same cutting function. In real-world scenes, multiple objects may share identical affordances, yet only one is appropriate under the given task context. We call such cases confusing pairs. However, existing 3D affordance methods largely sidestep this challenge by evaluating isolated single objects, often with explicit category names provided in the query. We formalize Multi-Object Affordance Grounding under Intent-Driven Instructions, a new 3D affordance setting that requires predicting a per-point affordance mask on the correct object within a cluttered multi-object point cloud, conditioned on implicit natural language intent. To study this problem, we construct CompassAD, the first benchmark centered on implicit intent in confusable multi-object scenes. It comprises 30 confusing object pairs spanning 16 affordance types, 6,422 scenes, and 88K+ query-answer pairs. Furthermore, we propose CompassNet, a framework that incorporates two dedicated modules tailored to this task. Instance-bounded Cross Injection (ICI) constrains language-geometry alignment within object boundaries to prevent cross-object semantic leakage. Bi-level Contrastive Refinement (BCR) enforces discrimination at both geometric-group and point levels, sharpening distinctions between target and confusable surfaces. Extensive experiments demonstrate state-of-the-art results on both seen and unseen queries, and deployment on a robotic manipulator confirms effective transfer to real-world grasping in confusing multi-object scenes.

  </details>



- **Jagle: Building a Large-Scale Japanese Multimodal Post-Training Dataset for Vision-Language Models**  
  Issa Sugiura, Keito Sasagawa, Keisuke Nakao, Koki Maeda, Ziqi Yin, Zhishen Yang, Shuhei Kurita, Yusuke Oda, Ryoko Tokuhisa, Daisuke Kawahara, et al.  
  _2026-04-02_ · https://arxiv.org/abs/2604.02048v1  
  <details><summary>Abstract</summary>

  Developing vision-language models (VLMs) that generalize across diverse tasks requires large-scale training datasets with diverse content. In English, such datasets are typically constructed by aggregating and curating numerous existing visual question answering (VQA) resources. However, this strategy does not readily extend to other languages, where VQA datasets remain limited in both scale and domain coverage, posing a major obstacle to building high-quality multilingual and non-English VLMs. In this work, we introduce Jagle, the largest Japanese multimodal post-training dataset to date, comprising approximately 9.2 million instances across diverse tasks. Rather than relying on existing VQA datasets, we collect heterogeneous source data, including images, image-text pairs, and PDF documents, and generate VQA pairs through multiple strategies such as VLM-based QA generation, translation, and text rendering. Experiments demonstrate that a 2.2B model trained with Jagle achieves strong performance on Japanese tasks, surpassing InternVL3.5-2B in average score across ten Japanese evaluation tasks and approaching within five points of Qwen3-VL-2B-Instruct. Furthermore, combining Jagle with FineVision does not degrade English performance; instead, it improves English performance compared to training with FineVision alone. To facilitate reproducibility and future research, we release the dataset, trained models, and code.

  </details>



- **Efficient Reasoning via Thought Compression for Language Segmentation**  
  Qing Zhou, Shiyu Zhang, Yuyu Jia, Junyu Gao, Weiping Ni, Junzheng Wu, Qi Wang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02040v1  
  <details><summary>Abstract</summary>

  Chain-of-thought (CoT) reasoning has significantly improved the performance of large multimodal models in language-guided segmentation, yet its prohibitive computational cost, stemming from generating verbose rationales, limits real-world applicability. We introduce WISE (Wisdom from Internal Self-Exploration), a novel paradigm for efficient reasoning guided by the principle of \textit{thinking twice -- once for learning, once for speed}. WISE trains a model to generate a structured sequence: a concise rationale, the final answer, and then a detailed explanation. By placing the concise rationale first, our method leverages autoregressive conditioning to enforce that the concise rationale acts as a sufficient summary for generating the detailed explanation. This structure is reinforced by a self-distillation objective that jointly rewards semantic fidelity and conciseness, compelling the model to internalize its detailed reasoning into a compact form. At inference, the detailed explanation is omitted. To address the resulting conditional distribution shift, our inference strategy, WISE-S, employs a simple prompting technique that injects a brevity-focused instruction into the user's query. This final adjustment facilitates the robust activation of the learned concise policy, unlocking the full benefits of our framework. Extensive experiments show that WISE-S achieves state-of-the-art zero-shot performance on the ReasonSeg benchmark with 58.3 cIoU, while reducing the average reasoning length by nearly \textbf{5$\times$} -- from 112 to just 23 tokens. Code is available at \href{https://github.com/mrazhou/WISE}{WISE}.

  </details>



- **O-ConNet: Geometry-Aware End-to-End Inference of Over-Constrained Spatial Mechanisms**  
  Haoyu Sun, Meng Zhao, Tianhao Wang, Jianxu Wu  
  _2026-04-02_ · https://arxiv.org/abs/2604.02038v1  
  <details><summary>Abstract</summary>

  Deep learning has shown strong potential for scientific discovery, but its ability to model macroscopic rigid-body kinematic constraints remains underexplored. We study this problem on spatial over-constrained mechanisms and propose O-ConNet, an end-to-end framework that infers mechanism structural parameters from only three sparse reachable points while reconstructing the full motion trajectory, without explicitly solving constraint equations during inference. On a self-constructed Bennett 4R dataset of 42,860 valid samples, O-ConNet achieves Param-MAE 0.276 +/- 0.077 and Traj-MAE 0.145 +/- 0.018 (mean +/- std over 10 runs), outperforming the strongest sequence baseline (LSTM-Seq2Seq) by 65.1 percent and 88.2 percent, respectively. These results suggest that end-to-end learning can capture closed-loop geometric structure and provide a practical route for inverse design of spatial over-constrained mechanisms under extremely sparse observations.

  </details>



- **IndoorCrowd: A Multi-Scene Dataset for Human Detection, Segmentation, and Tracking with an Automated Annotation Pipeline**  
  Sebastian-Ion Nae, Radu Moldoveanu, Alexandra Stefania Ghita, Adina Magda Florea  
  _2026-04-02_ · https://arxiv.org/abs/2604.02032v1  
  <details><summary>Abstract</summary>

  Understanding human behaviour in crowded indoor environments is central to surveillance, smart buildings, and human-robot interaction, yet existing datasets rarely capture real-world indoor complexity at scale. We introduce IndoorCrowd, a multi-scene dataset for indoor human detection, instance segmentation, and multi-object tracking, collected across four campus locations (ACS-EC, ACS-EG, IE-Central, R-Central). It comprises $31$ videos ($9{,}913$ frames at $5$fps) with human-verified, per-instance segmentation masks. A $620$-frame control subset benchmarks three foundation-model auto-annotators: SAM3, GroundingSAM, and EfficientGroundingSAM, against human labels using Cohen's $κ$, AP, precision, recall, and mask IoU. A further $2{,}552$-frame subset supports multi-object tracking with continuous identity tracks in MOTChallenge format. We establish detection, segmentation, and tracking baselines using YOLOv8n, YOLOv26n, and RT-DETR-L paired with ByteTrack, BoT-SORT, and OC-SORT. Per-scene analysis reveals substantial difficulty variation driven by crowd density, scale, and occlusion: ACS-EC, with $79.3\%$ dense frames and a mean instance scale of $60.8$px, is the most challenging scene. The project page is available at https://sheepseb.github.io/IndoorCrowd/.

  </details>



- **Rare-Aware Autoencoding: Reconstructing Spatially Imbalanced Data**  
  Alejandro Castañeda Garcia, Jan van Gemert, Daan Brinks, Nergis Tömen  
  _2026-04-02_ · https://arxiv.org/abs/2604.02031v1  
  <details><summary>Abstract</summary>

  Autoencoders can be challenged by spatially non-uniform sampling of image content. This is common in medical imaging, biology, and physics, where informative patterns occur rarely at specific image coordinates, as background dominates these locations in most samples, biasing reconstructions toward the majority appearance. In practice, autoencoders are biased toward dominant patterns resulting in the loss of fine-grained detail and causing blurred reconstructions for rare spatial inputs especially under spatial data imbalance. We address spatial imbalance by two complementary components: (i) self-entropy-based loss that upweights statistically uncommon spatial locations and (ii) Sample Propagation, a replay mechanism that selectively re-exposes the model to hard to reconstruct samples across batches during training. We benchmark existing data balancing strategies, originally developed for supervised classification, in the unsupervised reconstruction setting. Drawing on the limitations of these approaches, our method specifically targets spatial imbalance by encouraging models to focus on statistically rare locations, improving reconstruction consistency compared to existing baselines. We validate in a simulated dataset with controlled spatial imbalance conditions, and in three, uncontrolled, diverse real-world datasets spanning physical, biological, and astronomical domains. Our approach outperforms baselines on various reconstruction metrics, particularly under spatial imbalance distributions. These results highlight the importance of data representation in a batch and emphasize rare samples in unsupervised image reconstruction. We will make all code and related data available.

  </details>



- **Are VLMs Lost Between Sky and Space? LinkS$^2$Bench for UAV-Satellite Dynamic Cross-View Spatial Intelligence**  
  Dian Liu, Jie Feng, Di Li, Yuhui Zheng, Guanbin Li, Weisheng Dong, Guangming Shi  
  _2026-04-02_ · https://arxiv.org/abs/2604.02020v1  
  <details><summary>Abstract</summary>

  Synergistic spatial intelligence between UAVs and satellites is indispensable for emergency response and security operations, as it uniquely integrates macro-scale global coverage with dynamic, real-time local perception. However, the capacity of Vision-Language Models (VLMs) to master this complex interplay remains largely unexplored. This gap persists primarily because existing benchmarks are confined to isolated Unmanned Aerial Vehicle (UAV) videos or static satellite imagery, failing to evaluate the dynamic local-to-global spatial mapping essential for comprehensive cross-view reasoning. To bridge this gap, we introduce LinkS$^2$Bench, the first comprehensive benchmark designed to evaluate VLMs' wide-area, dynamic cross-view spatial intelligence. LinkS$^2$Bench links 1,022 minutes of dynamic UAV footage with high-resolution satellite imagery covering over 200 km$^2$. Through an LMM-assisted pipeline and rigorous human annotation, we constructed 17.9k high-quality question-answer pairs comprising 12 fine-grained tasks across four dimensions: perception, localization, relation, and reasoning. Evaluations of 18 representative VLMs reveal a substantial gap compared to human baselines, identifying accurate cross-view dynamic alignment as the critical bottleneck. To alleviate this, we design a Cross-View Alignment Adapter, demonstrating that explicit alignment significantly improves model performance. Furthermore, fine-tuning experiments underscore the potential of LinkS$^2$Bench in advancing VLM adaptation for complex spatial reasoning.

  </details>



- **ProDiG: Progressive Diffusion-Guided Gaussian Splatting for Aerial to Ground Reconstruction**  
  Sirshapan Mitra, Yogesh S. Rawat  
  _2026-04-02_ · https://arxiv.org/abs/2604.02003v1  
  <details><summary>Abstract</summary>

  Generating ground-level views and coherent 3D site models from aerial-only imagery is challenging due to extreme viewpoint changes, missing intermediate observations, and large scale variations. Existing methods either refine renderings post-hoc, often producing geometrically inconsistent results, or rely on multi-altitude ground-truth, which is rarely available. Gaussian Splatting and diffusion-based refinements improve fidelity under small variations but fail under wide aerial-to-ground gaps. To address these limitations, we introduce ProDiG (Progressive Altitude Gaussian Splatting), a diffusion-guided framework that progressively transforms aerial 3D representations toward ground-level fidelity. ProDiG synthesizes intermediate-altitude views and refines the Gaussian representation at each stage using a geometry-aware causal attention module that injects epipolar structure into reference-view diffusion. A distance-adaptive Gaussian module dynamically adjusts Gaussian scale and opacity based on camera distance, ensuring stable reconstruction across large viewpoint gaps. Together, these components enable progressive, geometrically grounded refinement without requiring additional ground-truth viewpoints. Extensive experiments on synthetic and real-world datasets demonstrate that ProDiG produces visually realistic ground-level renderings and coherent 3D geometry, significantly outperforming existing approaches in terms of visual quality, geometric consistency, and robustness to extreme viewpoint changes.

  </details>



- **Interactive Tracking: A Human-in-the-Loop Paradigm with Memory-Augmented Adaptation**  
  Yuqing Huang, Guotian Zeng, Zhenqiao Yuan, Zhenyu He, Xin Li, Yaowei Wang, Ming-Hsuan Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01974v1  
  <details><summary>Abstract</summary>

  Existing visual trackers mainly operate in a non-interactive, fire-and-forget manner, making them impractical for real-world scenarios that require human-in-the-loop adaptation. To overcome this limitation, we introduce Interactive Tracking, a new paradigm that allows users to guide the tracker at any time using natural language commands. To support research in this direction, we make three main contributions. First, we present InteractTrack, the first large-scale benchmark for interactive tracking, containing 150 videos with dense bounding box annotations and timestamped language instructions. Second, we propose a comprehensive evaluation protocol and evaluate 25 representative trackers, showing that state-of-the-art methods fail in interactive scenarios; strong performance on conventional benchmarks does not transfer. Third, we introduce Interactive Memory-Augmented Tracking (IMAT), a new baseline that employs a dynamic memory mechanism to learn from user feedback and update tracking behavior accordingly. Our benchmark, protocol, and baseline establish a foundation for developing more intelligent, adaptive, and collaborative tracking systems, bridging the gap between automated perception and human guidance. The full benchmark, tracking results, and analysis are available at https://github.com/NorahGreen/InteractTrack.git.

  </details>



- **NearID: Identity Representation Learning via Near-identity Distractors**  
  Aleksandar Cvejic, Rameen Abdal, Abdelrahman Eldesokey, Bernard Ghanem, Peter Wonka  
  _2026-04-02_ · https://arxiv.org/abs/2604.01973v1  
  <details><summary>Abstract</summary>

  When evaluating identity-focused tasks such as personalized generation and image editing, existing vision encoders entangle object identity with background context, leading to unreliable representations and metrics. We introduce the first principled framework to address this vulnerability using Near-identity (NearID) distractors, where semantically similar but distinct instances are placed on the exact same background as a reference image, eliminating contextual shortcuts and isolating identity as the sole discriminative signal. Based on this principle, we present the NearID dataset (19K identities, 316K matched-context distractors) together with a strict margin-based evaluation protocol. Under this setting, pre-trained encoders perform poorly, achieving Sample Success Rates (SSR), a strict margin-based identity discrimination metric, as low as 30.7% and often ranking distractors above true cross-view matches. We address this by learning identity-aware representations on a frozen backbone using a two-tier contrastive objective enforcing the hierarchy: same identity > NearID distractor > random negative. This improves SSR to 99.2%, enhances part-level discrimination by 28.0%, and yields stronger alignment with human judgments on DreamBench++, a human-aligned benchmark for personalization. Project page: https://gorluxor.github.io/NearID/

  </details>



- **Ego-Grounding for Personalized Question-Answering in Egocentric Videos**  
  Junbin Xiao, Shenglang Zhang, Pengxiang Zhu, Angela Yao  
  _2026-04-02_ · https://arxiv.org/abs/2604.01966v1  
  <details><summary>Abstract</summary>

  We present the first systematic analysis of multimodal large language models (MLLMs) in personalized question-answering requiring ego-grounding - the ability to understand the camera-wearer in egocentric videos. To this end, we introduce MyEgo, the first egocentric VideoQA dataset designed to evaluate MLLMs' ability to understand, remember, and reason about the camera wearer. MyEgo comprises 541 long videos and 5K personalized questions asking about "my things", "my activities", and "my past". Benchmarking reveals that competitive MLLMs across variants, including open-source vs. proprietary, thinking vs. non-thinking, small vs. large scales all struggle on MyEgo. Top closed- and open-source models (e.g., GPT-5 and Qwen3-VL) achieve only~46% and 36% accuracy, trailing human performance by near 40% and 50% respectively. Surprisingly, neither explicit reasoning nor model scaling yield consistent improvements. Models improve when relevant evidence is explicitly provided, but gains drop over time, indicating limitations in tracking and remembering "me" and "my past". These findings collectively highlight the crucial role of ego-grounding and long-range memory in enabling personalized QA in egocentric videos. We hope MyEgo and our analyses catalyze further progress in these areas for egocentric personalized assistance. Data and code are available at https://github.com/Ryougetsu3606/MyEgo

  </details>



- **Automated Prostate Gland Segmentation in MRI Using nnU-Net**  
  Pablo Rodriguez-Belenguer, Gloria Ribas, Javier Aquerreta Escribano, Rafael Moreno-Calatayud, Leonor Cerda-Alberich, Luis Marti-Bonmati  
  _2026-04-02_ · https://arxiv.org/abs/2604.01964v1  
  <details><summary>Abstract</summary>

  Accurate segmentation of the prostate gland in multiparametric MRI (mpMRI) is a fundamental step for a wide range of clinical and research applications, including image registration, volume estimation, and radiomic analysis. However, manual delineation is time-consuming and subject to inter-observer variability, while general-purpose segmentation tools often fail to provide sufficient accuracy for prostate-specific tasks. In this work, we propose a dedicated deep learning-based approach for automatic prostate gland segmentation using the nnU-Net v2 framework. The model leverages multimodal mpMRI data, including T2-weighted imaging, diffusion-weighted imaging (DWI), and apparent diffusion coefficient (ADC) maps, to exploit complementary tissue information. Training was performed on 981 cases from the PI-CAI dataset using whole-gland annotations, and model performance was assessed through 5-fold cross-validation and external validation on an independent cohort of 54 patients from Hospital La Fe. The proposed model achieved a mean Dice score of 0.96 +/- 0.00 in cross-validation and 0.82 on the external test set, demonstrating strong generalization despite domain shift. In comparison, a general-purpose approach (TotalSegmentator) showed substantially lower performance, with a Dice score of 0.15, primarily due to under-segmentation of the gland. These results highlight the importance of task-specific, multimodal segmentation strategies and demonstrate the potential of the proposed approach for reliable integration into clinical research workflows. To facilitate reproducibility and deployment, the model has been fully containerized and is available as a ready-to-use inference tool.

  </details>



- **A Self supervised learning framework for imbalanced medical imaging datasets**  
  Yash Kumar Sharma, Charan Ramtej Kodi, Vineet Padmanabhan  
  _2026-04-02_ · https://arxiv.org/abs/2604.01947v1  
  <details><summary>Abstract</summary>

  Two problems often plague medical imaging analysis: 1) Non-availability of large quantities of labeled training data, and 2) Dealing with imbalanced data, i.e., abundant data are available for frequent classes, whereas data are highly limited for the rare class. Self supervised learning (SSL) methods have been proposed to deal with the first problem to a certain extent, but the issue of investigating the robustness of SSL to imbalanced data has rarely been addressed in the domain of medical image classification. In this work, we make the following contributions: 1) The MIMV method proposed by us in an earlier work is extended with a new augmentation strategy to construct asymmetric multi-image, multi-view (AMIMV) pairs to address both data scarcity and dataset imbalance in medical image classification. 2) We carry out a data analysis to evaluate the robustness of AMIMV under varying degrees of class imbalance in medical imaging . 3) We evaluate eight representative SSL methods in 11 medical imaging datasets (MedMNIST) under long-tailed distributions and limited supervision. Our experimental results on the MedMNIST dataset show an improvement of 4.25% on retinaMNIST, 1.88% on tissueMNIST, and 3.1% on DermaMNIST.

  </details>



- **Captioning Daily Activity Images in Early Childhood Education: Benchmark and Algorithm**  
  Sixing Li, Zhibin Gu, Ziqi Zhang, Weiguo Pan, Bing Li, Ying Wang, Hongzhe Liu  
  _2026-04-02_ · https://arxiv.org/abs/2604.01941v1  
  <details><summary>Abstract</summary>

  Image captioning for Early Childhood Education (ECE) is essential for automated activity understanding and educational assessment. However, existing methods face two key challenges. First, the lack of large-scale, domain-specific datasets limits the model's ability to capture fine-grained semantic concepts unique to ECE scenarios, resulting in generic and imprecise descriptions. Second, conventional training paradigms exhibit limitations in enhancing professional object description capability, as supervised learning tends to favor high-frequency expressions, while reinforcement learning may suffer from unstable optimization on difficult samples. To address these limitations, we introduce ECAC, a large-scale benchmark for ECE daily activity image captioning, comprising 256,121 real-world images annotated with expert-level captions and fine-grained labels. ECAC is further equipped with a domain-oriented evaluation protocol, the Teaching Toy Recognition Score (TTS), to explicitly measure professional object naming accuracy. Furthermore, we propose RSRS (Reward-Conditional Switch of Reinforcement Learning and Supervised Fine-Tuning), a hybrid training framework that dynamically alternates between RL and supervised optimization. By rerouting hard samples with zero rewards to supervised fine-tuning, RSRS effectively mitigates advantage collapse and enables stable optimization for fine-grained recognition. Leveraging ECAC and RSRS, we develop KinderMM-Cap-3B, a domain-adapted multimodal large language model. Extensive experiments demonstrate that our model achieves a TTS of 51.06, substantially outperforming state-of-the-art baselines while maintaining superior caption quality, highlighting its potential for specialized educational applications.

  </details>



- **Night Eyes: A Reproducible Framework for Constellation-Based Corneal Reflection Matching**  
  Virmarie Maquiling, Yasmeen Abdrabou, Enkelejda Kasneci  
  _2026-04-02_ · https://arxiv.org/abs/2604.01909v1  
  <details><summary>Abstract</summary>

  Corneal reflection (glint) detection plays an important role in pupil-corneal reflection (P-CR) eye tracking, but in practice it is often handled as heuristics embedded within larger systems, making reproducibility difficult across hardware setups. We introduce a 2D geometry-driven, constellation-based pipeline for mulit-glint detection and matching, focusing on reproducibility and clear evaluation. Inspired by lost-in-space star identification, we treat glints as structured constellations rather than independent blobs. We propose a Similarity-Layout Alignment (SLA) procedure which adapts constellation matching to the specific constraints of multi-LED eye tracking. The framework brings together controlled over-detection, adaptive candidate fallback, appearance-aware scoring, and optional semantic layout priors while keeping detection and correspondence explicitly separated. Evaluated on a public multi-LED dataset, the system provides stable identity-preserving correspondence under noisy conditions. We release code, presets, and evaluation scripts to enable transparent replication, comparison, and dataset annotation.

  </details>



- **GeoAI Agency Primitives**  
  Akram Zaytar, Rohan Sawahn, Caleb Robinson, Gilles Q. Hacheme, Girmaw A. Tadesse, Inbal Becker-Reshef, Rahul Dodhia, Juan Lavista Ferres  
  _2026-04-02_ · https://arxiv.org/abs/2604.01869v1  
  <details><summary>Abstract</summary>

  We present ongoing research on agency primitives for GeoAI assistants -- core capabilities that connect Foundation models to the artifact-centric, human-in-the-loop workflows where GIS practitioners actually work. Despite advances in satellite image captioning, visual question answering, and promptable segmentation, these capabilities have not translated into productivity gains for practitioners who spend most of their time producing vector layers, raster maps, and cartographic products. The gap is not model capability alone but the absence of an agency layer that supports iterative collaboration. We propose a vocabulary of $9$ primitives for such a layer -- including navigation, perception, geo-referenced memory, and dual modeling -- along with a benchmark that measures human productivity. Our goal is a vocabulary that makes agentic assistance in GIS implementable, testable, and comparable.

  </details>



- **MAR-MAER: Metric-Aware and Ambiguity-Adaptive Autoregressive Image Generation**  
  Kai Dong, Tingting Bai  
  _2026-04-02_ · https://arxiv.org/abs/2604.01864v1  
  <details><summary>Abstract</summary>

  Autoregressive (AR) models have demonstrated significant success in the realm of text-to-image generation. However, they usually face two major challenges. Firstly, the generated images may not always meet the quality standards expected by humans. Furthermore, these models face difficulty when dealing with ambiguous prompts that could be interpreted in several valid ways. To address these issues, we introduce MAR-MAER, an innovative hierarchical autoregressive framework. It combines two main components. It is a metric-aware embedding regularization method. The other one is a probabilistic latent model used for handling ambiguous semantics. Our method utilizes a lightweight projection head, which is trained with an adaptive kernel regression loss function. This aligns the model's internal representations with human-preferred quality metrics, such as CLIPScore and HPSv2. As a result, the embedding space that is learned more accurately reflects human judgment. We are also introducing a conditional variational module. This approach incorporates an aspect of controlled randomness within the hierarchical token generation process. This capability allows the model to produce a diverse array of coherent images based on ambiguous or open-ended prompts. We conducted extensive experiments using COCO and a newly developed Ambiguous-Prompt Benchmark. The results show that MAR-MAER achieves excellent performance in both metric consistency and semantic flexibility. It exceeds the baseline Hi-MAR model's performance, showing an improvement of +1.6 in CLIPScore and +5.3 in HPSv2. For unclear inputs, it produces a notably wider range of outputs. These findings have been confirmed through both human evaluation and automated metrics.

  </details>



- **Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation**  
  Hinako Mitsuoka, Kazuhiro Hotta  
  _2026-04-02_ · https://arxiv.org/abs/2604.01859v1  
  <details><summary>Abstract</summary>

  Recent progress in Temporal Action Segmentation (TAS) has increasingly relied on complex architectures, which can hinder practical deployment. We present a lightweight dual-loss training framework that improves fine-grained segmentation quality with only one additional output channel and two auxiliary loss terms, requiring minimal architectural modification. Our approach combines a boundary-regression loss that promotes accurate temporal localization via a single-channel boundary prediction and a CDF-based segment-level regularization loss that encourages coherent within-segment structure by matching cumulative distributions over predicted and ground-truth segments. The framework is architecture-agnostic and can be integrated into existing TAS models (e.g., MS-TCN, C2F-TCN, FACT) as a training-time loss function. Across three benchmark datasets, the proposed method improves segment-level consistency and boundary quality, yielding higher F1 and Edit scores across three different models. Frame-wise accuracy remains largely unchanged, highlighting that precise segmentation can be achieved through simple loss design rather than heavier architectures or inference-time refinements.

  </details>



- **Semantic Segmentation of Textured Non-manifold 3D Meshes using Transformers**  
  Mohammadreza Heidarianbaei, Max Mehltretter, Franz Rottensteiner  
  _2026-04-02_ · https://arxiv.org/abs/2604.01836v1  
  <details><summary>Abstract</summary>

  Textured 3D meshes jointly represent geometry, topology, and appearance, yet their irregular structure poses significant challenges for deep-learning-based semantic segmentation. While a few recent methods operate directly on meshes without imposing geometric constraints, they typically overlook the rich textural information also provided by such meshes. We introduce a texture-aware transformer that learns directly from raw pixels associated with each mesh face, coupled with a new hierarchical learning scheme for multi-scale feature aggregation. A texture branch summarizes all face-level pixels into a learnable token, which is fused with geometrical descriptors and processed by a stack of Two-Stage Transformer Blocks (TSTB), which allow for both a local and a global information flow. We evaluate our model on the Semantic Urban Meshes (SUM) benchmark and a newly curated cultural-heritage dataset comprising textured roof tiles with triangle-level annotations for damage types. Our method achieves 81.9\% mF1 and 94.3\% OA on SUM and 49.7\% mF1 and 72.8\% OA on the new dataset, substantially outperforming existing approaches.

  </details>



- **A deep learning pipeline for PAM50 subtype classification using histopathology images and multi-objective patch selection**  
  Arezoo Borji, Gernot Kronreif, Bernhard Angermayr, Francisco Mario Calisto, Wolfgang Birkfellner, Inna Servetnyk, Yinyin Yuan, Sepideh Hatamikia  
  _2026-04-02_ · https://arxiv.org/abs/2604.01798v1  
  <details><summary>Abstract</summary>

  Breast cancer is a highly heterogeneous disease with diverse molecular profiles. The PAM50 gene signature is widely recognized as a standard for classifying breast cancer into intrinsic subtypes, enabling more personalized treatment strategies. In this study, we introduce a novel optimization-driven deep learning framework that aims to reduce reliance on costly molecular assays by directly predicting PAM50 subtypes from H&E-stained whole-slide images (WSIs). Our method jointly optimizes patch informativeness, spatial diversity, uncertainty, and patch count by combining the non-dominated sorting genetic algorithm II (NSGA-II) with Monte Carlo dropout-based uncertainty estimation. The proposed method can identify a small but highly informative patch subset for classification. We used a ResNet18 backbone for feature extraction and a custom CNN head for classification. For evaluation, we used the internal TCGA-BRCA dataset as the training cohort and the external CPTAC-BRCA dataset as the test cohort. On the internal dataset, an F1-score of 0.8812 and an AUC of 0.9841 using 627 WSIs from the TCGA-BRCA cohort were achieved. The performance of the proposed approach on the external validation dataset showed an F1-score of 0.7952 and an AUC of 0.9512. These findings indicate that the proposed optimization-guided, uncertainty-aware patch selection can achieve high performance and improve the computational efficiency of histopathology-based PAM50 classification compared to existing methods, suggesting a scalable imaging-based replacement that has the potential to support clinical decision-making.

  </details>



- **PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency**  
  Leezy Han, Seunggyu Kim, Dongseok Shim, Hyeonbeom Lee  
  _2026-04-02_ · https://arxiv.org/abs/2604.01791v1  
  <details><summary>Abstract</summary>

  Monocular depth estimation (MDE) has been widely adopted in the perception systems of autonomous vehicles and mobile robots. However, existing approaches often struggle to maintain temporal consistency in depth estimation across consecutive frames. This inconsistency not only causes jitter but can also lead to estimation failures when the depth range changes abruptly. To address these challenges, this paper proposes a consistency-aware monocular depth estimation framework that leverages wheel odometry from a mobile robot to achieve stable and coherent depth predictions over time. Specifically, we estimate camera pose and sparse depth from triangulation using optical flow between consecutive frames. The sparse depth estimates are used to update a recursive Bayesian estimate of the metric scale, which is then applied to rescale the relative depth predicted by a pre-trained depth estimation foundation model. The proposed method is evaluated on the KITTI, TartanAir, MS2, and our own dataset, demonstrating robust and accurate depth estimation performance.

  </details>



- **Hidden Meanings in Plain Sight: RebusBench for Evaluating Cognitive Visual Reasoning**  
  Seyed Amir Kasaei, Arash Marioriyad, Mahbod Khaleti, MohammadAmin Fazli, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban  
  _2026-04-02_ · https://arxiv.org/abs/2604.01764v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (LVLMs) have achieved remarkable proficiency in explicit visual recognition, effectively describing what is directly visible in an image. However, a critical cognitive gap emerges when the visual input serves only as a clue rather than the answer. We identify that current models struggle with the complex, multi-step reasoning required to solve problems where information is not explicitly depicted. Successfully solving a rebus puzzle requires a distinct cognitive workflow: the model must extract visual and textual attributes, retrieve linguistic prior knowledge (such as idioms), and perform abstract mapping to synthesize these elements into a meaning that exists outside the pixel space. To evaluate this neurosymbolic capability, we introduce RebusBench, a benchmark of 1,164 puzzles designed to test this specific integration of perception and knowledge. Our evaluation of state-of-the-art models (including Qwen, InternVL, and LLaVA) shows a severe deficiency: performance saturates below 10% Exact Match and 20% semantic accuracy, with no significant improvement observed from model scaling or In-Context Learning (ICL). These findings suggest that while models possess the necessary visual and linguistic components, they lack the cognitive reasoning glue to connect them. Project page available at https://amirkasaei.com/rebusbench/.

  </details>



- **Cosine-Normalized Attention for Hyperspectral Image Classification**  
  Muhammad Ahmad, Manuel Mazzara  
  _2026-04-02_ · https://arxiv.org/abs/2604.01763v1  
  <details><summary>Abstract</summary>

  Transformer-based methods have improved hyperspectral image classification (HSIC) by modeling long-range spatial-spectral dependencies; however, their attention mechanisms typically rely on dot-product similarity, which mixes feature magnitude and orientation and may be suboptimal for hyperspectral data. This work revisits attention scoring from a geometric perspective and introduces a cosine-normalized attention formulation that aligns similarity computation with the angular structure of hyperspectral signatures. By projecting query and key embeddings onto a unit hypersphere and applying a squared cosine similarity, the proposed method emphasizes angular relationships while reducing sensitivity to magnitude variations. The formulation is integrated into a spatial-spectral Transformer and evaluated under extremely limited supervision. Experiments on three benchmark datasets demonstrate that the proposed approach consistently achieves higher performance, outperforming several recent Transformer- and Mamba-based models despite using a lightweight backbone. In addition, a controlled analysis of multiple attention score functions shows that cosine-based scoring provides a reliable inductive bias for hyperspectral representation learning.

  </details>



- **Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding**  
  Jiayun Jin, Haolong Chai, Xueying Huang, Xiaoqing Guo, Zengwei Zheng, Zhan Zhou, Junmei Wang, Xinyu Wang, Jie Liu, Binbin Zhou  
  _2026-04-02_ · https://arxiv.org/abs/2604.01749v1  
  <details><summary>Abstract</summary>

  Ultrasound imaging is widely used in clinical diagnostics due to its real-time capability and radiation-free nature. However, existing vision-language pre-training models, such as CLIP, are primarily designed for other modalities, and are difficult to directly apply to ultrasound data, which exhibit heterogeneous anatomical structures and diverse diagnostic attributes. To bridge this gap, we construct US-365K, a large-scale ultrasound image-text dataset containing 365k paired samples across 52 anatomical categories. We establish Ultrasonographic Diagnostic Taxonomy (UDT) containing two hierarchical knowledge frameworks. Ultrasonographic Hierarchical Anatomical Taxonomy standardizes anatomical organization, and Ultrasonographic Diagnostic Attribute Framework formalizes nine diagnostic dimensions, including body system, organ, diagnosis, shape, margins, echogenicity, internal characteristics, posterior acoustic phenomena, and vascularity. Building upon these foundations, we propose Ultrasound-CLIP, a semantic-aware contrastive learning framework that introduces semantic soft labels and semantic loss to refine sample discrimination. Moreover, we construct a heterogeneous graph modality derived from UDAF's textual representations, enabling structured reasoning over lesion-attribute relations. Extensive experiments with patient-level data splitting demonstrate that our approach achieves state-of-the-art performance on classification and retrieval benchmarks, while also delivering strong generalization to zero-shot, linear probing, and fine-tuning tasks.

  </details>



- **Unifying UAV Cross-View Geo-Localization via 3D Geometric Perception**  
  Haoyuan Li, Wen Yang, Fang Xu, Hong Tan, Haijian Zhang, Shengyang Li, Gui-Song Xia  
  _2026-04-02_ · https://arxiv.org/abs/2604.01747v1  
  <details><summary>Abstract</summary>

  Cross-view geo-localization for Unmanned Aerial Vehicles (UAVs) operating in GNSS-denied environments remains challenging due to the severe geometric discrepancy between oblique UAV imagery and orthogonal satellite maps. Most existing methods address this problem through a decoupled pipeline of place retrieval and pose estimation, implicitly treating perspective distortion as appearance noise rather than an explicit geometric transformation. In this work, we propose a geometry-aware UAV geo-localization framework that explicitly models the 3D scene geometry to unify coarse place recognition and fine-grained pose estimation within a single inference pipeline. Our approach reconstructs a local 3D scene from multi-view UAV image sequences using a Visual Geometry Grounded Transformer (VGGT), and renders a virtual Bird's-Eye View (BEV) representation that orthorectifies the UAV perspective to align with satellite imagery. This BEV serves as a geometric intermediary that enables robust cross-view retrieval and provides spatial priors for accurate 3 Degrees of Freedom (3-DoF) pose regression. To efficiently handle multiple location hypotheses, we introduce a Satellite-wise Attention Block that isolates the interaction between each satellite candidate and the reconstructed UAV scene, preventing inter-candidate interference while maintaining linear computational complexity. In addition, we release a recalibrated version of the University-1652 dataset with precise coordinate annotations and spatial overlap analysis, enabling rigorous evaluation of end-to-end localization accuracy. Extensive experiments on the refined University-1652 benchmark and SUES-200 demonstrate that our method significantly outperforms state-of-the-art baselines, achieving robust meter-level localization accuracy and improved generalization in complex urban environments.

  </details>



- **Setup-Independent Full Projector Compensation**  
  Haibo Li, Qingyue Deng, Jijiang Li, Haibin Ling, Bingyao Huang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01736v1  
  <details><summary>Abstract</summary>

  Projector compensation seeks to correct geometric and photometric distortions that occur when images are projected onto nonplanar or textured surfaces. However, most existing methods are highly setup-dependent, requiring fine-tuning or retraining whenever the surface, lighting, or projector-camera pose changes. Progress has been limited by two key challenges: (1) the absence of large, diverse training datasets and (2) existing geometric correction models are typically constrained by specific spatial setups; without further retraining or fine-tuning, they often fail to generalize directly to novel geometric configurations. We introduce SIComp, the first Setup-Independent framework for full projector Compensation, capable of generalizing to unseen setups without fine-tuning or retraining. To enable this, we construct a large-scale real-world dataset spanning 277 distinct projector-camera setups. SIComp adopts a co-adaptive design that decouples geometry and photometry: A carefully tailored optical flow module performs online geometric correction, while a novel photometric network handles photometric compensation. To further enhance robustness under varying illumination, we integrate intensity-varying surface priors into the network design. Extensive experiments demonstrate that SIComp consistently produces high-quality compensation across diverse unseen setups, substantially outperforming existing methods in terms of generalization ability and establishing the first generalizable solution to projector compensation. The code and dataset are available on our project page: https://hai-bo-li.github.io/SIComp/

  </details>



- **Hi-LOAM: Hierarchical Implicit Neural Fields for LiDAR Odometry and Mapping**  
  Zhiliu Yang, Jianyuan Zhang, Lianhui Zhao, Jinyu Dai, Zhu Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01720v1  
  <details><summary>Abstract</summary>

  LiDAR Odometry and Mapping (LOAM) is a pivotal technique for embodied-AI applications such as autonomous driving and robot navigation. Most existing LOAM frameworks are either contingent on the supervision signal, or lack of the reconstruction fidelity, which are deficient in depicting details of large-scale complex scenes. To overcome these limitations, we propose a multi-scale implicit neural localization and mapping framework using LiDAR sensor, called Hi-LOAM. Hi-LOAM receives LiDAR point cloud as the input data modality, learns and stores hierarchical latent features in multiple levels of hash tables based on an octree structure, then these multi-scale latent features are decoded into signed distance value through shallow Multilayer Perceptrons (MLPs) in the mapping procedure. For pose estimation procedure, we rely on a correspondence-free, scan-to-implicit matching paradigm to estimate optimal pose and register current scan into the submap. The entire training process is conducted in a self-supervised manner, which waives the model pre-training and manifests its generalizability when applied to diverse environments. Extensive experiments on multiple real-world and synthetic datasets demonstrate the superior performance, in terms of the effectiveness and generalization capabilities, of our Hi-LOAM compared to existing state-of-the-art methods.

  </details>


