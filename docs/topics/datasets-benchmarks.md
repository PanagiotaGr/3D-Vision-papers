# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-05 07:08 UTC_

Total papers shown: **50**


---

- **SimpliHuMoN: Simplifying Human Motion Prediction**  
  Aadya Agrawal, Alexander Schwing  
  _2026-03-04_ · https://arxiv.org/abs/2603.04399v1  
  <details><summary>Abstract</summary>

  Human motion prediction combines the tasks of trajectory forecasting and human pose prediction. For each of the two tasks, specialized models have been developed. Combining these models for holistic human motion prediction is non-trivial, and recent methods have struggled to compete on established benchmarks for individual tasks. To address this, we propose a simple yet effective transformer-based model for human motion prediction. The model employs a stack of self-attention modules to effectively capture both spatial dependencies within a pose and temporal relationships across a motion sequence. This simple, streamlined, end-to-end model is sufficiently versatile to handle pose-only, trajectory-only, and combined prediction tasks without task-specific modifications. We demonstrate that this approach achieves state-of-the-art results across all tasks through extensive experiments on a wide range of benchmark datasets, including Human3.6M, AMASS, ETH-UCY, and 3DPW.

  </details>



- **TaxonRL: Reinforcement Learning with Intermediate Rewards for Interpretable Fine-Grained Visual Reasoning**  
  Maximilian von Klinski, Maximilian Schall  
  _2026-03-04_ · https://arxiv.org/abs/2603.04380v1  
  <details><summary>Abstract</summary>

  Traditional vision-language models struggle with contrastive fine-grained taxonomic reasoning, particularly when distinguishing between visually similar species within the same genus or family. We introduce TaxonRL, a reinforcement learning approach using Group Relative Policy Optimization with intermediate rewards that decomposes the reasoning process into hierarchical taxonomic predictions. Our method incentivizes models to explicitly reason about species-level, genus-level, and family-level features before making final classifications. This structured approach is designed not only to boost accuracy but also to yield a transparent, verifiable decision-making process. On the challenging Birds-to-Words dataset, TaxonRL achieves 91.7\% average accuracy, exceeding human performance (77.3\%) while generating interpretable reasoning traces. We demonstrate strong cross-domain generalization, showing substantial gains in primate and marine species verification. Our results establish that enforcing structured, hierarchical reasoning provides a powerful and transferable framework for fine-grained visual discrimination.

  </details>



- **ManipulationNet: An Infrastructure for Benchmarking Real-World Robot Manipulation with Physical Skill Challenges and Embodied Multimodal Reasoning**  
  Yiting Chen, Kenneth Kimble, Edward H. Adelson, Tamim Asfour, Podshara Chanrungmaneekul, Sachin Chitta, Yash Chitambar, Ziyang Chen, Ken Goldberg, Danica Kragic, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.04363v1  
  <details><summary>Abstract</summary>

  Dexterous manipulation enables robots to purposefully alter the physical world, transforming them from passive observers into active agents in unstructured environments. This capability is the cornerstone of physical artificial intelligence. Despite decades of advances in hardware, perception, control, and learning, progress toward general manipulation systems remains fragmented due to the absence of widely adopted standard benchmarks. The central challenge lies in reconciling the variability of the real world with the reproducibility and authenticity required for rigorous scientific evaluation. To address this, we introduce ManipulationNet, a global infrastructure that hosts real-world benchmark tasks for robotic manipulation. ManipulationNet delivers reproducible task setups through standardized hardware kits, and enables distributed performance evaluation via a unified software client that delivers real-time task instructions and collects benchmarking results. As a persistent and scalable infrastructure, ManipulationNet organizes benchmark tasks into two complementary tracks: 1) the Physical Skills Track, which evaluates low-level physical interaction skills, and 2) the Embodied Reasoning Track, which tests high-level reasoning and multimodal grounding abilities. This design fosters the systematic growth of an interconnected network of real-world abilities and skills, paving the path toward general robotic manipulation. By enabling comparable manipulation research in the real world at scale, this infrastructure establishes a sustainable foundation for measuring long-term scientific progress and identifying capabilities ready for real-world deployment.

  </details>



- **RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots**  
  Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, Yuke Zhu  
  _2026-03-04_ · https://arxiv.org/abs/2603.04356v1  
  <details><summary>Abstract</summary>

  Recent advances in robot learning have accelerated progress toward generalist robots that can perform everyday tasks in human environments. Yet it remains difficult to gauge how close we are to this vision. The field lacks a reproducible, large-scale benchmark for systematic evaluation. To fill this gap, we present RoboCasa365, a comprehensive simulation benchmark for household mobile manipulation. Built on the RoboCasa platform, RoboCasa365 introduces 365 everyday tasks across 2,500 diverse kitchen environments, with over 600 hours of human demonstration data and over 1600 hours of synthetically generated demonstration data -- making it one of the most diverse and large-scale resources for studying generalist policies. RoboCasa365 is designed to support systematic evaluations for different problem settings, including multi-task learning, robot foundation model training, and lifelong learning. We conduct extensive experiments on this benchmark with state-of-the-art methods and analyze the impacts of task diversity, dataset scale, and environment variation on generalization. Our results provide new insights into what factors most strongly affect the performance of generalist robots and inform strategies for future progress in the field.

  </details>



- **RANGER: Sparsely-Gated Mixture-of-Experts with Adaptive Retrieval Re-ranking for Pathology Report Generation**  
  Yixin Chen, Ziyu Su, Hikmat Khan, Muhammad Khalid Khan Niazi  
  _2026-03-04_ · https://arxiv.org/abs/2603.04348v1  
  <details><summary>Abstract</summary>

  Pathology report generation remains a relatively under-explored downstream task, primarily due to the gigapixel scale and complex morphological heterogeneity of Whole Slide Images (WSIs). Existing pathology report generation frameworks typically employ transformer architectures, relying on a homogeneous decoder architecture and static knowledge retrieval integration. Such architectures limit generative specialization and may introduce noisy external guidance during the report generation process. To address these limitations, we propose RANGER, a sparsely-gated Mixture-of-Experts (MoE) framework with adaptive retrieval re-ranking for pathology report generation. Specifically, we integrate a sparsely gated MoE into the decoder, along with noisy top-$k$ routing and load-balancing regularization, to enable dynamic expert specialization across various diagnostic patterns. Additionally, we introduce an adaptive retrieval re-ranking module that selectively refines retrieved memory from a knowledge base before integration, reducing noise and improving semantic alignment based on visual feature representations. We perform extensive experiments on the PathText-BRCA dataset and demonstrate consistent improvements over existing approaches across standard natural language generation metrics. Our full RANGER model achieves optimal performance on PathText dataset, reaching BLEU-1 to BLEU-4 scores of 0.4598, 0.3044, 0.2036, and 0.1435, respectively, with METEOR of 0.1883, and ROUGE-L of 0.3038, validating the effectiveness of dynamic expert routing and adaptive knowledge refinement for semantically grounded pathology report generation.

  </details>



- **Underrepresented in Foundation Model Pretraining Data? A One-Shot Probe**  
  Chris Vorster, Mayug Maniparambil, Noel E. O'Connor, Noel Murphy, Derek Molloy  
  _2026-03-04_ · https://arxiv.org/abs/2603.04346v1  
  <details><summary>Abstract</summary>

  Large-scale Vision-Language Foundation Models (VLFMs), such as CLIP, now underpin a wide range of computer vision research and applications. VLFMs are often adapted to various domain-specific tasks. However, VLFM performance on novel, specialised, or underrepresented domains remains inconsistent. Evaluating VLFMs typically requires labelled test sets, which are often unavailable for niche domains of interest, particularly those from the Global South. We address this gap by proposing a highly data-efficient method to predict a VLFM's zero-shot accuracy on a target domain using only a single labelled image per class. Our approach uses a Large Language Model to generate plausible counterfactual descriptions of a given image. By measuring the VLFM's ability to distinguish the correct description from these hard negatives, we engineer features that capture the VLFM's discriminative power in its shared embedding space. A linear regressor trained on these similarity scores estimates the VLFM's zero-shot test accuracy across various visual domains with a Pearson-r correlation of 0.96. We demonstrate our method's performance across five diverse datasets, including standard benchmark datasets and underrepresented datasets from Africa. Our work provides a low-cost, reliable tool for probing VLFMs, enabling researchers and practitioners to make informed decisions about data annotation efforts before committing significant resources. The model training code, generated captions and counterfactuals are released here: https://github.com/chris-vorster/PreLabellingProbe.

  </details>



- **Hold-One-Shot-Out (HOSO) for Validation-Free Few-Shot CLIP Adapters**  
  Chris Vorster, Mayug Maniparambil, Noel E. O'Connor, Noel Murphy, Derek Molloy  
  _2026-03-04_ · https://arxiv.org/abs/2603.04341v1  
  <details><summary>Abstract</summary>

  In many CLIP adaptation methods, a blending ratio hyperparameter controls the trade-off between general pretrained CLIP knowledge and the limited, dataset-specific supervision from the few-shot cases. Most few-shot CLIP adaptation techniques report results by ablation of the blending ratio on the test set or require additional validation sets to select the blending ratio per dataset, and thus are not strictly few-shot. We present a simple, validation-free method for learning the blending ratio in CLIP adaptation. Hold-One-Shot-Out (HOSO) presents a novel approach for CLIP-Adapter-style methods to compete in the newly established validation-free setting. CLIP-Adapter with HOSO (HOSO-Adapter) learns the blending ratio using a one-shot, hold-out set, while the adapter trains on the remaining few-shot support examples. Under the validation-free few-shot protocol, HOSO-Adapter outperforms the CLIP-Adapter baseline by more than 4 percentage points on average across 11 standard few-shot datasets. Interestingly, in the 8- and 16-shot settings, HOSO-Adapter outperforms CLIP-Adapter even with the optimal blending ratio selected on the test set. Ablation studies validate the use of a one-shot hold-out mechanism, decoupled training, and improvements over the naively learnt blending ratio baseline. Code is released here: https://github.com/chris-vorster/HOSO-Adapter

  </details>



- **Pointer-CAD: Unifying B-Rep and Command Sequences via Pointer-based Edges & Faces Selection**  
  Dacheng Qi, Chenyu Wang, Jingwei Xu, Tianzhe Chu, Zibo Zhao, Wen Liu, Wenrui Ding, Yi Ma, Shenghua Gao  
  _2026-03-04_ · https://arxiv.org/abs/2603.04337v1  
  <details><summary>Abstract</summary>

  Constructing computer-aided design (CAD) models is labor-intensive but essential for engineering and manufacturing. Recent advances in Large Language Models (LLMs) have inspired the LLM-based CAD generation by representing CAD as command sequences. But these methods struggle in practical scenarios because command sequence representation does not support entity selection (e.g. faces or edges), limiting its ability to support complex editing operations such as chamfer or fillet. Further, the discretization of a continuous variable during sketch and extrude operations may result in topological errors. To address these limitations, we present Pointer-CAD, a novel LLM-based CAD generation framework that leverages a pointer-based command sequence representation to explicitly incorporate the geometric information of B-rep models into sequential modeling. In particular, Pointer-CAD decomposes CAD model generation into steps, conditioning the generation of each subsequent step on both the textual description and the B-rep generated from previous steps. Whenever an operation requires the selection of a specific geometric entity, the LLM predicts a Pointer that selects the most feature-consistent candidate from the available set. Such a selection operation also reduces the quantization error in the command sequence-based representation. To support the training of Pointer-CAD, we develop a data annotation pipeline that produces expert-level natural language descriptions and apply it to build a dataset of approximately 575K CAD models. Extensive experimental results demonstrate that Pointer-CAD effectively supports the generation of complex geometric structures and reduces segmentation error to an extremely low level, achieving a significant improvement over prior command sequence methods, thereby significantly mitigating the topological inaccuracies introduced by quantization error.

  </details>



- **MOO: A Multi-view Oriented Observations Dataset for Viewpoint Analysis in Cattle Re-Identification**  
  William Grolleau, Achraf Chaouch, Astrid Sabourin, Guillaume Lapouge, Catherine Achard  
  _2026-03-04_ · https://arxiv.org/abs/2603.04314v1  
  <details><summary>Abstract</summary>

  Animal re-identification (ReID) faces critical challenges due to viewpoint variations, particularly in Aerial-Ground (AG-ReID) settings where models must match individuals across drastic elevation changes. However, existing datasets lack the precise angular annotations required to systematically analyze these geometric variations. To address this, we introduce the Multi-view Oriented Observation (MOO) dataset, a large-scale synthetic AG-ReID dataset of $1,000$ cattle individuals captured from $128$ uniformly sampled viewpoints ($128,000$ annotated images). Using this controlled dataset, we quantify the influence of elevation and identify a critical elevation threshold, above which models generalize significantly better to unseen views. Finally, we validate the transferability to real-world applications in both zero-shot and supervised settings, demonstrating performance gains across four real-world cattle datasets and confirming that synthetic geometric priors effectively bridge the domain gap. Collectively, this dataset and analysis lay the foundation for future model development in cross-view animal ReID. MOO is publicly available at https://github.com/TurtleSmoke/MOO.

  </details>



- **Dual Diffusion Models for Multi-modal Guided 3D Avatar Generation**  
  Hong Li, Yutang Feng, Minqi Meng, Yichen Yang, Xuhui Liu, Baochang Zhang  
  _2026-03-04_ · https://arxiv.org/abs/2603.04307v1  
  <details><summary>Abstract</summary>

  Generating high-fidelity 3D avatars from text or image prompts is highly sought after in virtual reality and human-computer interaction. However, existing text-driven methods often rely on iterative Score Distillation Sampling (SDS) or CLIP optimization, which struggle with fine-grained semantic control and suffer from excessively slow inference. Meanwhile, image-driven approaches are severely bottlenecked by the scarcity and high acquisition cost of high-quality 3D facial scans, limiting model generalization. To address these challenges, we first construct a novel, large-scale dataset comprising over 100,000 pairs across four modalities: fine-grained textual descriptions, in-the-wild face images, high-quality light-normalized texture UV maps, and 3D geometric shapes. Leveraging this comprehensive dataset, we propose PromptAvatar, a framework featuring dual diffusion models. Specifically, it integrates a Texture Diffusion Model (TDM) that supports flexible multi-condition guidance from text and/or image prompts, alongside a Geometry Diffusion Model (GDM) guided by text prompts. By learning the direct mapping from multi-modal prompts to 3D representations, PromptAvatar eliminates the need for time-consuming iterative optimization, successfully generating high-fidelity, shading-free 3D avatars in under 10 seconds. Extensive quantitative and qualitative experiments demonstrate that our method significantly outperforms existing state-of-the-art approaches in generation quality, fine-grained detail alignment, and computational efficiency.

  </details>



- **Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight**  
  Chao Qin, Jiaxu Xing, Rudolf Reiter, Angel Romero, Yifan Lin, Hugh H. -T. Liu, Davide Scaramuzza  
  _2026-03-04_ · https://arxiv.org/abs/2603.04305v1  
  <details><summary>Abstract</summary>

  Agile quadrotor flight pushes the limits of control, actuation, and onboard perception. While time-optimal trajectory planning has been extensively studied, existing approaches typically neglect the tight coupling between vehicle dynamics, environmental geometry, and the visual requirements of onboard state estimation. As a result, trajectories that are dynamically feasible may fail in closed-loop execution due to degraded visual quality. This paper introduces a unified time-optimal trajectory optimization framework for vision-based quadrotors that explicitly incorporates perception constraints alongside full nonlinear dynamics, rotor actuation limits, aerodynamic effects, camera field-of-view constraints, and convex geometric gate representations. The proposed formulation solves minimum-time lap trajectories for arbitrary racetracks with diverse gate shapes and orientations, while remaining numerically robust and computationally efficient. We derive an information-theoretic position uncertainty metric to quantify visual state-estimation quality and integrate it into the planner through three perception objectives: position uncertainty minimization, sequential field-of-view constraints, and look-ahead alignment. This enables systematic exploration of the trade-offs between speed and perceptual reliability. To accurately track the resulting perception-aware trajectories, we develop a model predictive contouring tracking controller that separates lateral and progress errors. Experiments demonstrate real-world flight speeds up to 9.8 m/s with 0.07 m average tracking error, and closed-loop success rates improved from 55% to 100% on a challenging Split-S course. The proposed system provides a scalable benchmark for studying the fundamental limits of perception-aware, time-optimal autonomous flight.

  </details>



- **CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video**  
  Lingen Li, Guangzhi Wang, Xiaoyu Li, Zhaoyang Zhang, Qi Dou, Jinwei Gu, Tianfan Xue, Ying Shan  
  _2026-03-04_ · https://arxiv.org/abs/2603.04291v1  
  <details><summary>Abstract</summary>

  Generating high-quality 360° panoramic videos from perspective input is one of the crucial applications for virtual reality (VR), whereby high-resolution videos are especially important for immersive experience. Existing methods are constrained by computational limitations of vanilla diffusion models, only supporting $\leq$ 1K resolution native generation and relying on suboptimal post super-resolution to increase resolution. We introduce CubeComposer, a novel spatio-temporal autoregressive diffusion model that natively generates 4K-resolution 360° videos. By decomposing videos into cubemap representations with six faces, CubeComposer autoregressively synthesizes content in a well-planned spatio-temporal order, reducing memory demands while enabling high-resolution output. Specifically, to address challenges in multi-dimensional autoregression, we propose: (1) a spatio-temporal autoregressive strategy that orchestrates 360° video generation across cube faces and time windows for coherent synthesis; (2) a cube face context management mechanism, equipped with a sparse context attention design to improve efficiency; and (3) continuity-aware techniques, including cube-aware positional encoding, padding, and blending to eliminate boundary seams. Extensive experiments on benchmark datasets demonstrate that CubeComposer outperforms state-of-the-art methods in native resolution and visual quality, supporting practical VR application scenarios. Project page: https://lg-li.github.io/project/cubecomposer

  </details>



- **A multi-center analysis of deep learning methods for video polyp detection and segmentation**  
  Noha Ghatwary, Pedro Chavarias Solano, Mohamed Ramzy Ibrahim, Adrian Krenzer, Frank Puppe, Stefano Realdon, Renato Cannizzaro, Jiacheng Wang, Liansheng Wang, Thuy Nuong Tran, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.04288v1  
  <details><summary>Abstract</summary>

  Colonic polyps are well-recognized precursors to colorectal cancer (CRC), typically detected during colonoscopy. However, the variability in appearance, location, and size of these polyps complicates their detection and removal, leading to challenges in effective surveillance, intervention, and subsequently CRC prevention. The processes of colonoscopy surveillance and polyp removal are highly reliant on the expertise of gastroenterologists and occur within the complexities of the colonic structure. As a result, there is a high rate of missed detections and incomplete removal of colonic polyps, which can adversely impact patient outcomes. Recently, automated methods that use machine learning have been developed to enhance polyps detection and segmentation, thus helping clinical processes and reducing missed rates. These advancements highlight the potential for improving diagnostic accuracy in real-time applications, which ultimately facilitates more effective patient management. Furthermore, integrating sequence data and temporal information could significantly enhance the precision of these methods by capturing the dynamic nature of polyp growth and the changes that occur over time. To rigorously investigate these challenges, data scientists and experts gastroenterologists collaborated to compile a comprehensive dataset that spans multiple centers and diverse populations. This initiative aims to underscore the critical importance of incorporating sequence data and temporal information in the development of robust automated detection and segmentation methods. This study evaluates the applicability of deep learning techniques developed in real-time clinical colonoscopy tasks using sequence data, highlighting the critical role of temporal relationships between frames in improving diagnostic precision.

  </details>



- **VANGUARD: Vehicle-Anchored Ground Sample Distance Estimation for UAVs in GPS-Denied Environments**  
  Yifei Chen, Xupeng Chen, Feng Wang, Niangang Jiao, Jiayin Liu  
  _2026-03-04_ · https://arxiv.org/abs/2603.04277v1  
  <details><summary>Abstract</summary>

  Autonomous aerial robots operating in GPS-denied or communication-degraded environments frequently lose access to camera metadata and telemetry, leaving onboard perception systems unable to recover the absolute metric scale of the scene. As LLM/VLM-based planners are increasingly adopted as high-level agents for embodied systems, their ability to reason about physical dimensions becomes safety-critical -- yet our experiments show that five state-of-the-art VLMs suffer from spatial scale hallucinations, with median area estimation errors exceeding 50%. We propose VANGUARD, a lightweight, deterministic Geometric Perception Skill designed as a callable tool that any LLM-based agent can invoke to recover Ground Sample Distance (GSD) from ubiquitous environmental anchors: small vehicles detected via oriented bounding boxes, whose modal pixel length is robustly estimated through kernel density estimation and converted to GSD using a pre-calibrated reference length. The tool returns both a GSD estimate and a composite confidence score, enabling the calling agent to autonomously decide whether to trust the measurement or fall back to alternative strategies. On the DOTA~v1.5 benchmark, VANGUARD achieves 6.87% median GSD error on 306~images. Integrated with SAM-based segmentation for downstream area measurement, the pipeline yields 19.7% median error on a 100-entry benchmark -- with 2.6x lower category dependence and 4x fewer catastrophic failures than the best VLM baseline -- demonstrating that equipping agents with deterministic geometric tools is essential for safe autonomous spatial reasoning.

  </details>



- **RoboLight: A Dataset with Linearly Composable Illumination for Robotic Manipulation**  
  Shutong Jin, Jin Yang, Muhammad Zahid, Florian T. Pokorny  
  _2026-03-04_ · https://arxiv.org/abs/2603.04249v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce RoboLight, the first real-world robotic manipulation dataset capturing synchronized episodes under systematically varied lighting conditions. RoboLight consists of two components. (a) RoboLight-Real contains 2,800 real-world episodes collected in our custom Light Cube setup, a calibrated system equipped with eight programmable RGB LED lights. It includes structured illumination variation along three independently controlled dimensions: color, direction, and intensity. Each dimension is paired with a dedicated task featuring objects of diverse geometries and materials to induce perceptual challenges. All image data are recorded in high-dynamic-range (HDR) format to preserve radiometric accuracy. Leveraging the linearity of light transport, we introduce (b) RoboLight-Synthetic, comprising 196,000 episodes synthesized through interpolation in the HDR image space of RoboLight-Real. In principle, RoboLight-Synthetic can be arbitrarily expanded by refining the interpolation granularity. We further verify the dataset quality through qualitative analysis and real-world policy roll-outs, analyzing task difficulty, distributional diversity, and the effectiveness of synthesized data. We additionally demonstrate three representative use cases of the proposed dataset. The full dataset, along with the system software and hardware design, will be released as open-source to support continued research.

  </details>



- **A Unified Framework for Joint Detection of Lacunes and Enlarged Perivascular Spaces**  
  Lucas He, Krinos Li, Hanyuan Zhang, Runlong He, Silvia Ingala, Luigi Lorenzini, Marleen de Bruijne, Frederik Barkhof, Rhodri Davies, Carole Sudre  
  _2026-03-04_ · https://arxiv.org/abs/2603.04243v1  
  <details><summary>Abstract</summary>

  Cerebral small vessel disease (CSVD) markers, specifically enlarged perivascular spaces (EPVS) and lacunae, present a unique challenge in medical image analysis due to their radiological mimicry. Standard segmentation networks struggle with feature interference and extreme class imbalance when handling these divergent targets simultaneously. To address these issues, we propose a morphology-decoupled framework where Zero-Initialized Gated Cross-Task Attention exploits dense EPVS context to guide sparse lacune detection. Furthermore, biological and topological consistency are enforced via a mixed-supervision strategy integrating Mutual Exclusion and Centerline Dice losses. Finally, we introduce an Anatomically-Informed Inference Calibration mechanism to dynamically suppress false positives based on tissue semantics. Extensive 5-folds cross-validation on the VALDO 2021 dataset (N=40) demonstrates state-of-the-art performance, notably surpassing task winners in lacunae detection precision (71.1%, p=0.01) and F1-score (62.6%, p=0.03). Furthermore, evaluation on the external EPAD cohort (N=1762) confirms the model's robustness for large-scale population studies. Code will be released upon acceptance.

  </details>



- **AMP2026: A Multi-Platform Marine Robotics Dataset for Tracking and Mapping**  
  Edwin Meriaux, Shuo Wen, David Widhalm, Zhizun Wang, Junming Shi, Mariana Sosa Guzmán, Kalvik Jakkala, Bennett Carley, Elias Sokolova, Yogesh Girdhar, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.04225v1  
  <details><summary>Abstract</summary>

  Marine environments present significant challenges for perception and autonomy due to dynamic surfaces, limited visibility, and complex interactions between aerial, surface, and submerged sensing modalities. This paper introduces the Aerial Marine Perception Dataset (AMP2026), a multi-platform marine robotics dataset collected across multiple field deployments designed to support research in two primary areas: multi-view tracking and marine environment mapping. The dataset includes synchronized data from aerial drones, boat-mounted cameras, and submerged robotic platforms, along with associated localization and telemetry information. The goal of this work is to provide a publicly available dataset enabling research in marine perception and multi-robot observation scenarios. This paper describes the data collection methodology, sensor configurations, dataset organization, and intended research tasks supported by the dataset.

  </details>



- **PRAM-R: A Perception-Reasoning-Action-Memory Framework with LLM-Guided Modality Routing for Adaptive Autonomous Driving**  
  Yi Zhang, Xian Zhang, Saisi Zhao, Yinglei Song, Chengdong Wu, Nenad Petrovic, Alois Knoll  
  _2026-03-04_ · https://arxiv.org/abs/2603.04222v1  
  <details><summary>Abstract</summary>

  Multimodal perception enables robust autonomous driving but incurs unnecessary computational cost when all sensors remain active. This paper presents PRAM-R, a unified Perception-Reasoning-Action-Memory framework with LLM-Guided Modality Routing for adaptive autonomous driving. PRAM-R adopts an asynchronous dual-loop design: a fast reactive loop for perception and control, and a slow deliberative loop for reasoning-driven modality selection and memory updates. An LLM router selects and weights modalities using environmental context and sensor diagnostics, while a hierarchical memory module preserves temporal consistency and supports long-term adaptation. We conduct a two-stage evaluation: (1) synthetic stress tests for stability analysis and (2) real-world validation on the nuScenes dataset. Synthetic stress tests confirm 87.2% reduction in routing oscillations via hysteresis-based stabilization. Real-world validation on nuScenes shows 6.22% modality reduction with 20% memory recall while maintaining comparable trajectory accuracy to full-modality baselines in complex urban scenarios. Our work demonstrates that LLM-augmented architectures with hierarchical memory achieve efficient, adaptive multimodal perception in autonomous driving.

  </details>



- **Real5-OmniDocBench: A Full-Scale Physical Reconstruction Benchmark for Robust Document Parsing in the Wild**  
  Changda Zhou, Ziyue Gao, Xueqing Wang, Tingquan Gao, Cheng Cui, Jing Tang, Yi Liu  
  _2026-03-04_ · https://arxiv.org/abs/2603.04205v1  
  <details><summary>Abstract</summary>

  While Vision-Language Models (VLMs) achieve near-perfect scores on digital document benchmarks like OmniDocBench, their performance in the unpredictable physical world remains largely unknown due to the lack of controlled yet realistic evaluations. We introduce Real5-OmniDocBench, the first benchmark that performs a full-scale, one-to-one physical reconstruction of the entire OmniDocBench v1.5 (1,355 images) across five critical real-world scenarios: Scanning, Warping, Screen-Photography, Illumination, and Skew. Unlike prior benchmark that either lack digital correspondence or employ partial sampling, our complete ground-truth mapping enables, for the first time, rigorous factor-wise attribution of performance degradation-allowing us to pinpoint whether failures stem from geometric distortions, optical artifacts, or model limitations. Our benchmark establishes a challenging new standard for the community, demonstrating that the 'reality gap' in document parsing is far from closed, and provides a diagnostic tool to guide the development of truly resilient document intelligence.

  </details>



- **Degradation-based augmented training for robust individual animal re-identification**  
  Thanos Polychronou, Lukáš Adam, Viktor Penchev, Kostas Papafitsoros  
  _2026-03-04_ · https://arxiv.org/abs/2603.04163v1  
  <details><summary>Abstract</summary>

  Wildlife re-identification aims to recognise individual animals by matching query images to a database of previously identified individuals, based on their fine-scale unique morphological characteristics. Current state-of-the-art models for multispecies re- identification are based on deep metric learning representing individual identities by fea- ture vectors in an embedding space, the similarity of which forms the basis for a fast automated identity retrieval. Yet very often, the discriminative information of individual wild animals gets significantly reduced due to the presence of several degradation factors in images, leading to reduced retrieval performance and limiting the downstream eco- logical studies. Here, starting by showing that the extent of this performance reduction greatly varies depending on the animal species (18 wild animal datasets), we introduce an augmented training framework for deep feature extractors, where we apply artificial but diverse degradations in images in the training set. We show that applying this augmented training only to a subset of individuals, leads to an overall increased re-identification performance, under the same type of degradations, even for individuals not seen during training. The introduction of diverse degradations during training leads to a gain of up to 8.5% Rank-1 accuracy to a dataset of real-world degraded animal images, selected using human re-ID expert annotations provided here for the first time. Our work is the first to systematically study image degradation in wildlife re-identification, while introducing all the necessary benchmarks, publicly available code and data, enabling further research on this topic.

  </details>



- **LISTA-Transformer Model Based on Sparse Coding and Attention Mechanism and Its Application in Fault Diagnosis**  
  Shuang Liu, Lina Zhao, Tian Wang, Huaqing Wang  
  _2026-03-04_ · https://arxiv.org/abs/2603.04146v1  
  <details><summary>Abstract</summary>

  Driven by the continuous development of models such as Multi-Layer Perceptron, Convolutional Neural Network (CNN), and Transformer, deep learning has made breakthrough progress in fields such as computer vision and natural language processing, and has been successfully applied in practical scenarios such as image classification and industrial fault diagnosis. However, existing models still have certain limitations in local feature modeling and global dependency capture. Specifically, CNN is limited by local receptive fields, while Transformer has shortcomings in effectively modeling local structures, and both face challenges of high model complexity and insufficient interpretability. In response to the above issues, we proposes the following innovative work: A sparse Transformer based on Learnable Iterative Shrinkage Threshold Algorithm (LISTA-Transformer) was designed, which deeply integrates LISTA sparse encoding with visual Transformer to construct a model architecture with adaptive local and global feature collaboration mechanism. This method utilizes continuous wavelet transform to convert vibration signals into time-frequency maps and inputs them into LISTA-Transformer for more effective feature extraction. On the CWRU dataset, the fault recognition rate of our method reached 98.5%, which is 3.3% higher than traditional methods and exhibits certain superiority over existing Transformer-based approaches.

  </details>



- **Crab$^{+}$: A Scalable and Unified Audio-Visual Scene Understanding Model with Explicit Cooperation**  
  Dongnuan Cai, Henghui Du, Chang Zhou, Xi Chen, Dan Guo, Hongyuan Zhang, Xuelong Li, Di Hu  
  _2026-03-04_ · https://arxiv.org/abs/2603.04128v1  
  <details><summary>Abstract</summary>

  Developing Audio-Visual Large Language Models (AV-LLMs) for unified scene understanding is pivotal in multimodal intelligence. While instruction tuning enables pre-trained models with multi-task abilities, we observe that conventional multi-task unification methods often suffer from severe negative transfer, where nearly 55% of tasks degrade compared to single-task training. We attribute this phenomenon to audio-visual task heterogeneity, characterized by disparate task granularity and divergent capability demands, which lead to negative interference under joint training. To tackle this, we present Crab$^{+}$, a scalable and unified audio-visual scene understanding model that addresses task heterogeneity through explicit cooperation from both data and model perspectives. On the data side, we introduce AV-UIE v2, a comprehensive Audio-Visual Unified Instruction-tuning dataset with Explicit reasoning processes. It contains approximately 222K samples spanning 17 datasets and 7 tasks, enabling the model to capture cross-task relationships at different levels of granularity. On the model side, we design a unified interface to align heterogeneous task formulations, and propose Interaction-aware LoRA (I-LoRA), which explicitly models inter-task relationships via dynamic routing to coordinate distinct audio-visual interaction patterns, mitigating parameter interference. Extensive experiments show Crab$^{+}$ covers broader tasks than existing unified models while outperforming specialized models on various benchmarks. We successfully reverse the negative transfer trend, achieving positive transfer where multi-task learning surpasses single-task baselines in nearly 88% of tasks. These results hold across diverse AV-LLM paradigms and are validated through in-depth visualization, positioning Crab$^{+}$ as a robust step towards holistic audio-visual scene understanding.

  </details>



- **A Baseline Study and Benchmark for Few-Shot Open-Set Action Recognition with Feature Residual Discrimination**  
  Stefano Berti, Giulia Pasquale, Lorenzo Natale  
  _2026-03-04_ · https://arxiv.org/abs/2603.04125v1  
  <details><summary>Abstract</summary>

  Few-Shot Action Recognition (FS-AR) has shown promising results but is often limited by a closed-set assumption that fails in real-world open-set scenarios. While Few-Shot Open-Set (FSOS) recognition is well-established for images, its extension to spatio-temporal video data remains underexplored. To address this, we propose an architectural extension based on a Feature-Residual Discriminator (FR-Disc), adapting previous work on skeletal data to the more complex video domain. Extensive experiments on five datasets demonstrate that while common open-set techniques provide only marginal gains, our FR-Disc significantly enhances unknown rejection capabilities without compromising closed-set accuracy, setting a new state-of-the-art for FSOS-AR. The project website, code, and benchmark are available at: https://hsp-iit.github.io/fsosar/.

  </details>



- **Any2Any: Unified Arbitrary Modality Translation for Remote Sensing**  
  Haoyang Chen, Jing Zhang, Hebaixu Wang, Shiqin Wang, Pohsun Huang, Jiayuan Li, Haonan Guo, Di Wang, Zheng Wang, Bo Du  
  _2026-03-04_ · https://arxiv.org/abs/2603.04114v1  
  <details><summary>Abstract</summary>

  Multi-modal remote sensing imagery provides complementary observations of the same geographic scene, yet such observations are frequently incomplete in practice. Existing cross-modal translation methods treat each modality pair as an independent task, resulting in quadratic complexity and limited generalization to unseen modality combinations. We formulate Any-to-Any translation as inference over a shared latent representation of the scene, where different modalities correspond to partial observations of the same underlying semantics. Based on this formulation, we propose Any2Any, a unified latent diffusion framework that projects heterogeneous inputs into a geometrically aligned latent space. Such structure performs anchored latent regression with a shared backbone, decoupling modality-specific representation learning from semantic mapping. Moreover, lightweight target-specific residual adapters are used to correct systematic latent mismatches without increasing inference complexity. To support learning under sparse but connected supervision, we introduce RST-1M, the first million-scale remote sensing dataset with paired observations across five sensing modalities, providing supervision anchors for any-to-any translation. Experiments across 14 translation tasks show that Any2Any consistently outperforms pairwise translation methods and exhibits strong zero-shot generalization to unseen modality pairs. Code and models will be available at https://github.com/MiliLab/Any2Any.

  </details>



- **Understanding Sources of Demographic Predictability in Brain MRI via Disentangling Anatomy and Contrast**  
  Mehmet Yigit Avci, Akshit Achara, Andrew King, Jorge Cardoso  
  _2026-03-04_ · https://arxiv.org/abs/2603.04113v1  
  <details><summary>Abstract</summary>

  Demographic attributes such as age, sex, and race can be predicted from medical images, raising concerns about bias in clinical AI systems. In brain MRI, this signal may arise from anatomical variation, acquisition-dependent contrast differences, or both, yet these sources remain entangled in conventional analyses. Without disentangling them, mitigation strategies risk failing to address the underlying causes. We propose a controlled framework based on disentangled representation learning, decomposing brain MRI into anatomy-focused representations that suppress acquisition influence and contrast embeddings that capture acquisition-dependent characteristics. Training predictive models for age, sex, and race on full images, anatomical representations, and contrast-only embeddings allows us to quantify the relative contributions of structure and acquisition to the demographic signal. Across three datasets and multiple MRI sequences, we find that demographic predictability is primarily rooted in anatomical variation: anatomy-focused representations largely preserve the performance of models trained on raw images. Contrast-only embeddings retain a weaker but systematic signal that is dataset-specific and does not generalise across sites. These findings suggest that effective mitigation must explicitly account for the distinct anatomical and acquisition-dependent origins of the demographic signal, ensuring that any bias reduction generalizes robustly across domains.

  </details>



- **Real Eyes Realize Faster: Gaze Stability and Pupil Novelty for Efficient Egocentric Learning**  
  Ajan Subramanian, Sumukh Bettadapura, Rohan Sathish  
  _2026-03-04_ · https://arxiv.org/abs/2603.04098v1  
  <details><summary>Abstract</summary>

  Always-on egocentric cameras are increasingly used as demonstrations for embodied robotics, imitation learning, and assistive AR, but the resulting video streams are dominated by redundant and low-quality frames. Under the storage and battery constraints of wearable devices, choosing which frames to keep is as important as how to learn from them. We observe that modern eye-tracking headsets provide a continuous, training-free side channel that decomposes into two complementary axes: gaze fixation captures visual stability (quality), while pupil response captures arousal-linked moments (novelty). We operationalize this insight as a Dual-Criterion Frame Curator that first gates frames by gaze quality and then ranks the survivors by pupil-derived novelty. On the Visual Experience Dataset (VEDB), curated frames at 10% budget match the classification performance of the full stream, and naive signal fusion consistently destroys both contributions. The benefit is task-dependent: pupil ranking improves activity recognition, while gaze-only selection already dominates for scene recognition, confirming that the two signals serve genuinely different roles. Our method requires no model inference and operates at capture time, offering a path toward efficient, always-on egocentric data curation.

  </details>



- **CLIP-Guided Multi-Task Regression for Multi-View Plant Phenotyping**  
  Simon Warmers, Muhammad Zawish, Fayaz Ali Dharejo, Steven Davy, Radu Timofte  
  _2026-03-04_ · https://arxiv.org/abs/2603.04091v1  
  <details><summary>Abstract</summary>

  Modeling plant growth dynamics plays a central role in modern agricultural research. However, learning robust predictors from multi-view plant imagery remains challenging due to strong viewpoint redundancy and viewpoint-dependent appearance changes. We propose a level-aware vision language framework that jointly predicts plant age and leaf count using a single multi-task model built on CLIP embeddings. Our method aggregates rotational views into angle-invariant representations and conditions visual features on lightweight text priors encoding viewpoint level for stable prediction under incomplete or unordered inputs. On the GroMo25 benchmark, our approach reduces mean age MAE from 7.74 to 3.91 and mean leaf-count MAE from 5.52 to 3.08 compared to the GroMo baseline, corresponding to improvements of 49.5% and 44.2%, respectively. The unified formulation simplifies the pipeline by replacing the conventional dual-model setup while improving robustness to missing views. The models and code is available at: https://github.com/SimonWarmers/CLIP-MVP

  </details>



- **EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR**  
  Zhenyu Li, Sai Kumar Dwivedi, Filip Maric, Carlos Chacon, Nadine Bertsch, Filippo Arcadu, Tomas Hodan, Michael Ramamonjisoa, Peter Wonka, Amy Zhao, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.04090v1  
  <details><summary>Abstract</summary>

  Egocentric human motion estimation is essential for AR/VR experiences, yet remains challenging due to limited body coverage from the egocentric viewpoint, frequent occlusions, and scarce labeled data. We present EgoPoseFormer v2, a method that addresses these challenges through two key contributions: (1) a transformer-based model for temporally consistent and spatially grounded body pose estimation, and (2) an auto-labeling system that enables the use of large unlabeled datasets for training. Our model is fully differentiable, introduces identity-conditioned queries, multi-view spatial refinement, causal temporal attention, and supports both keypoints and parametric body representations under a constant compute budget. The auto-labeling system scales learning to tens of millions of unlabeled frames via uncertainty-aware semi-supervised training. The system follows a teacher-student schema to generate pseudo-labels and guide training with uncertainty distillation, enabling the model to generalize to different environments. On the EgoBody3M benchmark, with a 0.8 ms latency on GPU, our model outperforms two state-of-the-art methods by 12.2% and 19.4% in accuracy, and reduces temporal jitter by 22.2% and 51.7%. Furthermore, our auto-labeling system further improves the wrist MPJPE by 13.1%.

  </details>



- **SaFeR: Safety-Critical Scenario Generation for Autonomous Driving Test via Feasibility-Constrained Token Resampling**  
  Jinlong Cui, Fenghua Liang, Guo Yang, Chengcheng Tang, Jianxun Cui  
  _2026-03-04_ · https://arxiv.org/abs/2603.04071v1  
  <details><summary>Abstract</summary>

  Safety-critical scenario generation is crucial for evaluating autonomous driving systems. However, existing approaches often struggle to balance three conflicting objectives: adversarial criticality, physical feasibility, and behavioral realism. To bridge this gap, we propose SaFeR: safety-critical scenario generation for autonomous driving test via feasibility-constrained token resampling. We first formulate traffic generation as a discrete next token prediction problem, employing a Transformer-based model as a realism prior to capture naturalistic driving distributions. To capture complex interactions while effectively mitigating attention noise, we propose a novel differential attention mechanism within the realism prior. Building on this prior, SaFeR implements a novel resampling strategy that induces adversarial behaviors within a high-probability trust region to maintain naturalism, while enforcing a feasibility constraint derived from the Largest Feasible Region (LFR). By approximating the LFR via offline reinforcement learning, SaFeR effectively prevents the generation of theoretically inevitable collisions. Closed-loop experiments on the Waymo Open Motion Dataset and nuPlan demonstrate that SaFeR significantly outperforms state-of-the-art baselines, achieving a higher solution rate and superior kinematic realism while maintaining strong adversarial effectiveness.

  </details>



- **Long-Term Visual Localization in Dynamic Benthic Environments: A Dataset, Footprint-Based Ground Truth, and Visual Place Recognition Benchmark**  
  Martin Kvisvik Larsen, Oscar Pizarro  
  _2026-03-04_ · https://arxiv.org/abs/2603.04056v1  
  <details><summary>Abstract</summary>

  Long-term visual localization has the potential to reduce cost and improve mapping quality in optical benthic monitoring with autonomous underwater vehicles (AUVs). Despite this potential, long-term visual localization in benthic environments remains understudied, primarily due to the lack of curated datasets for benchmarking. Moreover, limited georeferencing accuracy and image footprints necessitate precise geometric information for accurate ground-truthing. In this work, we address these gaps by presenting a curated dataset for long-term visual localization in benthic environments and a novel method to ground-truth visual localization results for near-nadir underwater imagery. Our dataset comprises georeferenced AUV imagery from five benthic reference sites, revisited over periods up to six years, and includes raw and color-corrected stereo imagery, camera calibrations, and sub-decimeter registered camera poses. To our knowledge, this is the first curated underwater dataset for long-term visual localization spanning multiple sites and photic-zone habitats. Our ground-truthing method estimates 3D seafloor image footprints and links camera views with overlapping footprints, ensuring that ground-truth links reflect shared visual content. Building on this dataset and ground truth, we benchmark eight state-of-the-art visual place recognition (VPR) methods and find that Recall@K is significantly lower on our dataset than on established terrestrial and underwater benchmarks. Finally, we compare our footprint-based ground truth to a traditional location-based ground truth and show that distance-threshold ground-truthing can overestimate VPR Recall@K at sites with rugged terrain and altitude variations. Together, the curated dataset, ground-truthing method, and VPR benchmark provide a stepping stone for advancing long-term visual localization in dynamic benthic environments.

  </details>



- **Force-Aware Residual DAgger via Trajectory Editing for Precision Insertion with Impedance Control**  
  Yiou Huang, Ma Ning, Weichu Zhao, Zinuo Liu, Jun Sun, Qiufeng Wang, Yaran Chen  
  _2026-03-04_ · https://arxiv.org/abs/2603.04038v1  
  <details><summary>Abstract</summary>

  Imitation learning (IL) has shown strong potential for contact-rich precision insertion tasks. However, its practical deployment is often hindered by covariate shift and the need for continuous expert monitoring to recover from failures during execution. In this paper, we propose Trajectory Editing Residual Dataset Aggregation (TER-DAgger), a scalable and force-aware human-in-the-loop imitation learning framework that mitigates covariate shift by learning residual policies through optimization-based trajectory editing. This approach smoothly fuses policy rollouts with human corrective trajectories, providing consistent and stable supervision. Second, we introduce a force-aware failure anticipation mechanism that triggers human intervention only when discrepancies arise between predicted and measured end-effector forces, significantly reducing the requirement for continuous expert monitoring. Third, all learned policies are executed within a Cartesian impedance control framework, ensuring compliant and safe behavior during contact-rich interactions. Extensive experiments in both simulation and real-world precision insertion tasks show that TER-DAgger improves the average success rate by over 37\% compared to behavior cloning, human-guided correction, retraining, and fine-tuning baselines, demonstrating its effectiveness in mitigating covariate shift and enabling scalable deployment in contact-rich manipulation.

  </details>



- **Weakly Supervised Patch Annotation for Improved Screening of Diabetic Retinopathy**  
  Shramana Dey, Abhirup Banerjee, B. Uma Shankar, Ramachandran Rajalakshmi, Sushmita Mitra  
  _2026-03-04_ · https://arxiv.org/abs/2603.03991v1  
  <details><summary>Abstract</summary>

  Diabetic Retinopathy (DR) requires timely screening to prevent irreversible vision loss. However, its early detection remains a significant challenge since often the subtle pathological manifestations (lesions) get overlooked due to insufficient annotation. Existing literature primarily focuses on image-level supervision, weakly-supervised localization, and clustering-based representation learning, which fail to systematically annotate unlabeled lesion region(s) for refining the dataset. Expert-driven lesion annotation is labor-intensive and often incomplete, limiting the performance of deep learning models. We introduce Similarity-based Annotation via Feature-space Ensemble (SAFE), a two-stage framework that unifies weak supervision, contrastive learning, and patch-wise embedding inference, to systematically expand sparse annotations in the pathology. SAFE preserves fine-grained details of the lesion(s) under partial clinical supervision. In the first stage, a dual-arm Patch Embedding Network learns semantically structured, class-discriminative embeddings from expert annotated patches. Next, an ensemble of independent embedding spaces extrapolates labels to the unannotated regions based on spatial and semantic proximity. An abstention mechanism ensures trade-off between highly reliable annotation and noisy coverage. Experimental results demonstrate reliable separation of healthy and diseased patches, achieving upto 0.9886 accuracy. The annotation generated from SAFE substantially improves downstream tasks such as DR classification, demonstrating a substantial increase in F1-score of the diseased class and a performance gain as high as 0.545 in Area Under the Precision-Recall Curve (AUPRC). Qualitative analysis, with explainability, confirms that SAFE focuses on clinically relevant lesion patterns; and is further validated by ophthalmologists.

  </details>



- **RIVER: A Real-Time Interaction Benchmark for Video LLMs**  
  Yansong Shi, Qingsong Zhao, Tianxiang Jiang, Xiangyu Zeng, Yi Wang, Limin Wang  
  _2026-03-04_ · https://arxiv.org/abs/2603.03985v1  
  <details><summary>Abstract</summary>

  The rapid advancement of multimodal large language models has demonstrated impressive capabilities, yet nearly all operate in an offline paradigm, hindering real-time interactivity. Addressing this gap, we introduce the Real-tIme Video intERaction Bench (RIVER Bench), designed for evaluating online video comprehension. RIVER Bench introduces a novel framework comprising Retrospective Memory, Live-Perception, and Proactive Anticipation tasks, closely mimicking interactive dialogues rather than responding to entire videos at once. We conducted detailed annotations using videos from diverse sources and varying lengths, and precisely defined the real-time interactive format. Evaluations across various model categories reveal that while offline models perform well in single question-answering tasks, they struggle with real-time processing. Addressing the limitations of existing models in online video interaction, especially their deficiencies in long-term memory and future perception, we proposed a general improvement method that enables models to interact with users more flexibly in real time. We believe this work will significantly advance the development of real-time interactive video understanding models and inspire future research in this emerging field. Datasets and code are publicly available at https://github.com/OpenGVLab/RIVER.

  </details>



- **GeoSeg: Training-Free Reasoning-Driven Segmentation in Remote Sensing Imagery**  
  Lifan Jiang, Yuhang Pei, oxi Wu, Yan Zhao, Tianrun Wu, Shulong Yu, Lihui Zhang, Deng Cai  
  _2026-03-04_ · https://arxiv.org/abs/2603.03983v1  
  <details><summary>Abstract</summary>

  Recent advances in MLLMs are reframing segmentation from fixed-category prediction to instruction-grounded localization. While reasoning based segmentation has progressed rapidly in natural scenes, remote sensing lacks a generalizable solution due to the prohibitive cost of reasoning-oriented data and domain-specific challenges like overhead viewpoints. We present GeoSeg, a zero-shot, training-free framework that bypasses the supervision bottleneck for reasoning-driven remote sensing segmentation. GeoSeg couples MLLM reasoning with precise localization via: (i) bias-aware coordinate refinement to correct systematic grounding shifts and (ii) a dual-route prompting mechanism to fuse semantic intent with fine-grained spatial cues. We also introduce GeoSeg-Bench, a diagnostic benchmark of 810 image--query pairs with hierarchical difficulty levels. Experiments show that GeoSeg consistently outperforms all baselines, with extensive ablations confirming the effectiveness and necessity of each component.

  </details>



- **Scaling Dense Event-Stream Pretraining from Visual Foundation Models**  
  Zhiwen Chen, Junhui Hou, Zhiyu Zhu, Jinjian Wu, Guangming Shi  
  _2026-03-04_ · https://arxiv.org/abs/2603.03969v1  
  <details><summary>Abstract</summary>

  Learning versatile, fine-grained representations from irregular event streams is pivotal yet nontrivial, primarily due to the heavy annotation that hinders scalability in dataset size, semantic richness, and application scope. To mitigate this dilemma, we launch a novel self-supervised pretraining method that distills visual foundation models (VFMs) to push the boundaries of event representation at scale. Specifically, we curate an extensive synchronized image-event collection to amplify cross-modal alignment. Nevertheless, due to inherent mismatches in sparsity and granularity between image-event domains, existing distillation paradigms are prone to semantic collapse in event representations, particularly at high resolutions. To bridge this gap, we propose to extend the alignment objective to semantic structures provided off-the-shelf by VFMs, indicating a broader receptive field and stronger supervision. The key ingredient of our method is a structure-aware distillation loss that grounds higher-quality image-event correspondences for alignment, optimizing dense event representations. Extensive experiments demonstrate that our approach takes a great leap in downstream benchmarks, significantly surpassing traditional methods and existing pretraining techniques. This breakthrough manifests in enhanced generalization, superior data efficiency and elevated transferability.

  </details>



- **UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization**  
  Qianfeng Yang, Qiyuan Guan, Xiang Chen, Jiyu Jin, Guiyue Jin, Jiangxin Dong  
  _2026-03-04_ · https://arxiv.org/abs/2603.03967v1  
  <details><summary>Abstract</summary>

  Despite significant progress has been made in image deraining, we note that most existing methods are often developed for only specific types of rain degradation and fail to generalize across diverse real-world rainy scenes. How to effectively model different rain degradations within a universal framework is important for real-world image deraining. In this paper, we propose UniRain, an effective unified image deraining framework capable of restoring images degraded by rain streak and raindrop under both daytime and nighttime conditions. To better enhance unified model generalization, we construct an intelligent retrieval augmented generation (RAG)-based dataset distillation pipeline that selects high-quality training samples from all public deraining datasets for better mixed training. Furthermore, we incorporate a simple yet effective multi-objective reweighted optimization strategy into the asymmetric mixture-of-experts (MoE) architecture to facilitate consistent performance and improve robustness across diverse scenes. Extensive experiments show that our framework performs favorably against the state-of-the-art models on our proposed benchmarks and multiple public datasets.

  </details>



- **ArthroCut: Autonomous Policy Learning for Robotic Bone Resection in Knee Arthroplasty**  
  Xu Lu, Yiling Zhang, Wenquan Cheng, Longfei Ma, Fang Chen, Hongen Liao  
  _2026-03-04_ · https://arxiv.org/abs/2603.03957v1  
  <details><summary>Abstract</summary>

  Despite rapid commercialization of surgical robots, their autonomy and real-time decision-making remain limited in practice. To address this gap, we propose ArthroCut, an autonomous policy learning framework that upgrades knee arthroplasty robots from assistive execution to context-aware action generation. ArthroCut fine-tunes a Qwen--VL backbone on a self-built, time-synchronized multimodal dataset from 21 complete cases (23,205 RGB--D pairs), integrating preoperative CT/MR, intraoperative NDI tracking of bones and end effector, RGB--D surgical video, robot state, and textual intent. The method operates on two complementary token families -- Preoperative Imaging Tokens (PIT) to encode patient-specific anatomy and planned resection planes, and Time-Aligned Surgical Tokens (TAST) to fuse real-time visual, geometric, and kinematic evidence -- and emits an interpretable action grammar under grammar/safety-constrained decoding. In bench-top experiments on a knee prosthesis across seven trials, ArthroCut achieves an average success rate of 86% over the six standard resections, significantly outperforming strong baselines trained under the same protocol. Ablations show that TAST is the principal driver of reliability while PIT provides essential anatomical grounding, and their combination yields the most stable multi-plane execution. These results indicate that aligning preoperative geometry with time-aligned intraoperative perception and translating that alignment into tokenized, constrained actions is an effective path toward robust, interpretable autonomy in orthopedic robotic surgery.

  </details>



- **RVN-Bench: A Benchmark for Reactive Visual Navigation**  
  Jaewon Lee, Jaeseok Heo, Gunmin Lee, Howoong Jun, Jeongwoo Oh, Songhwai Oh  
  _2026-03-04_ · https://arxiv.org/abs/2603.03953v1  
  <details><summary>Abstract</summary>

  Safe visual navigation is critical for indoor mobile robots operating in cluttered environments. Existing benchmarks, however, often neglect collisions or are designed for outdoor scenarios, making them unsuitable for indoor visual navigation. To address this limitation, we introduce the reactive visual navigation benchmark (RVN-Bench), a collision-aware benchmark for indoor mobile robots. In RVN-Bench, an agent must reach sequential goal positions in previously unseen environments using only visual observations and no prior map, while avoiding collisions. Built on the Habitat 2.0 simulator and leveraging high-fidelity HM3D scenes, RVN-Bench provides large-scale, diverse indoor environments, defines a collision-aware navigation task and evaluation metrics, and offers tools for standardized training and benchmarking. RVN-Bench supports both online and offline learning by offering an environment for online reinforcement learning, a trajectory image dataset generator, and tools for producing negative trajectory image datasets that capture collision events. Experiments show that policies trained on RVN-Bench generalize effectively to unseen environments, demonstrating its value as a standardized benchmark for safe and robust visual navigation. Code and additional materials are available at: https://rvn-bench.github.io/.

  </details>



- **Spatial Causal Prediction in Video**  
  Yanguang Zhao, Jie Yang, Shengqiong Wu, Shutong Hu, Hongbo Qiu, Yu Wang, Guijia Zhang, Tan Kai Ze, Hao Fei, Chia-Wen Lin, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.03944v1  
  <details><summary>Abstract</summary>

  Spatial reasoning, the ability to understand spatial relations, causality, and dynamic evolution, is central to human intelligence and essential for real-world applications such as autonomous driving and robotics. Existing studies, however, primarily assess models on visible spatio-temporal understanding, overlooking their ability to infer unseen past or future spatial states. In this work, we introduce Spatial Causal Prediction (SCP), a new task paradigm that challenges models to reason beyond observation and predict spatial causal outcomes. We further construct SCP-Bench, a benchmark comprising 2,500 QA pairs across 1,181 videos spanning diverse viewpoints, scenes, and causal directions, to support systematic evaluation. Through comprehensive experiments on {23} state-of-the-art models, we reveal substantial gaps between human and model performance, limited temporal extrapolation, and weak causal grounding. We further analyze key factors influencing performance and propose perception-enhancement and reasoning-guided strategies toward advancing spatial causal intelligence. The project page is https://guangstrip.github.io/SCP-Bench.

  </details>



- **Lightweight Visual Reasoning for Socially-Aware Robots**  
  Alessio Galatolo, Ronald Cumbal, Alexandros Rouchitsas, Katie Winkle, Didem Gürdür Broo, Ginevra Castellano  
  _2026-03-04_ · https://arxiv.org/abs/2603.03942v1  
  <details><summary>Abstract</summary>

  Robots operating in shared human environments must not only navigate, interact, and detect their surroundings, they must also interpret and respond to dynamic, and often unpredictable, human behaviours. Although recent advances have shown promise in enhancing robotic perception and instruction-following using Vision-Language Models (VLMs), they remain limited in addressing the complexities of multimodal human-robot interactions (HRI). Motivated by this challenge, we introduce a lightweight language-to-vision feedback module that closes the loop between an LLM and the vision encoder in VLMs. The module projects image-token hidden states through a gated Multi-Layer Perceptron (MLP) back into the encoder input, prompting a second pass that reinterprets the scene under text context. We evaluate this approach on three robotics-centred tasks: navigation in a simulated environment (Habitat), sequential scene description (Mementos-Robotics), and human-intention recognition (our HRI dataset). Results show that our method improves Qwen 2.5 (7B) by $3.3\%$ (less distance), $+0.057$ description score, and $+2.93\%$ accuracy, with less than $3\%$ extra parameters; Gemma 3 (4B) and LLaVA OV 1.5 (4B) show mixed navigation results but gains $+0.111,+0.055$ and $+10.81\%,+4.79\%$ on the latter two tasks. Code is available at https://github.com/alessioGalatolo/VLM-Reasoning-for-Robotics

  </details>



- **Slice-wise quality assessment of high b-value breast DWI via deep learning-based artifact detection**  
  Ameya Markale, Luise Brock, Ihor Horishnyi, Dominika Skwierawska, Tri-Thien Nguyen, Hannes Schreiter, Shirin Heidarikahkesh, Lorenz A. Kapsner, Michael Uder, Sabine Ohlmeyer, et al.  
  _2026-03-04_ · https://arxiv.org/abs/2603.03941v1  
  <details><summary>Abstract</summary>

  Diffusion-weighted imaging (DWI) can support lesion detection and characterization in breast magnetic resonance imaging (MRI), however especially high b-value diffusion-weighted acquisitions can be prone to intensity artifacts that can affect diagnostic image assessment. This study aims to detect both hyper- and hypointense artifacts on high b-value diffusion-weighted images (b=1500 s/mm2) using deep learning, employing either a binary classification (artifact presence) or a multiclass classification (artifact intensity) approach on a slice-wise dataset.This IRB-approved retrospective study used the single-center dataset comprising n=11806 slices from routine 3T breast MRI examinations performed between 2022 and mid-2023. Three convolutional neural network (CNN) architectures (DenseNet121, ResNet18, and SEResNet50) were trained for binary classification of hyper- and hypointense artifacts. The best performing model (DenseNet121) was applied to an independent holdout test set and was further trained separately for multiclass classification. Evaluation included area under receiver operating characteristic curve (AUROC), area under precision recall curve (AUPRC), precision, and recall, as well as analysis of predicted bounding box positions, derived from the network Grad-CAM heatmaps. DenseNet121 achieved AUROCs of 0.92 and 0.94 for hyper- and hypointense artifact detection, respectively, and weighted AUROCs of 0.85 and 0.88 for multiclass classification on single-slice high b-value diffusion-weighted images. A radiologist evaluated bounding box precision on a 1-5 Likert-like scale across 200 slices, achieving mean scores of 3.33+-1.04 for hyperintense artifacts and 2.62+-0.81 for hypointense artifacts. Hyper- and hypointense artifact detection in slice-wise breast DWI MRI dataset (b=1500 s/mm2) using CNNs particularly DenseNet121, seems promising and requires further validation.

  </details>



- **Cross-Modal Mapping and Dual-Branch Reconstruction for 2D-3D Multimodal Industrial Anomaly Detection**  
  Radia Daci, Vito Renò, Cosimo Patruno, Angelo Cardellicchio, Abdelmalik Taleb-Ahmed, Marco Leo, Cosimo Distante  
  _2026-03-04_ · https://arxiv.org/abs/2603.03939v1  
  <details><summary>Abstract</summary>

  Multimodal industrial anomaly detection benefits from integrating RGB appearance with 3D surface geometry, yet existing \emph{unsupervised} approaches commonly rely on memory banks, teacher-student architectures, or fragile fusion schemes, limiting robustness under noisy depth, weak texture, or missing modalities. This paper introduces \textbf{CMDR-IAD}, a lightweight and modality-flexible unsupervised framework for reliable anomaly detection in 2D+3D multimodal as well as single-modality (2D-only or 3D-only) settings. \textbf{CMDR-IAD} combines bidirectional 2D$\leftrightarrow$3D cross-modal mapping to model appearance-geometry consistency with dual-branch reconstruction that independently captures normal texture and geometric structure. A two-part fusion strategy integrates these cues: a reliability-gated mapping anomaly highlights spatially consistent texture-geometry discrepancies, while a confidence-weighted reconstruction anomaly adaptively balances appearance and geometric deviations, yielding stable and precise anomaly localization even in depth-sparse or low-texture regions. On the MVTec 3D-AD benchmark, CMDR-IAD achieves state-of-the-art performance while operating without memory banks, reaching 97.3\% image-level AUROC (I-AUROC), 99.6\% pixel-level AUROC (P-AUROC), and 97.6\% AUPRO. On a real-world polyurethane cutting dataset, the 3D-only variant attains 92.6\% I-AUROC and 92.5\% P-AUROC, demonstrating strong effectiveness under practical industrial conditions. These results highlight the framework's robustness, modality flexibility, and the effectiveness of the proposed fusion strategies for industrial visual inspection. Our source code is available at https://github.com/ECGAI-Research/CMDR-IAD/

  </details>



- **DISC: Dense Integrated Semantic Context for Large-Scale Open-Set Semantic Mapping**  
  Felix Igelbrink, Lennart Niecksch, Martin Atzmueller, Joachim Hertzberg  
  _2026-03-04_ · https://arxiv.org/abs/2603.03935v1  
  <details><summary>Abstract</summary>

  Open-set semantic mapping enables language-driven robotic perception, but current instance-centric approaches are bottlenecked by context-depriving and computationally expensive crop-based feature extraction. To overcome this fundamental limitation, we introduce DISC (Dense Integrated Semantic Context), featuring a novel single-pass, distance-weighted extraction mechanism. By deriving high-fidelity CLIP embeddings directly from the vision transformer's intermediate layers, our approach eliminates the latency and domain-shift artifacts of traditional image cropping, yielding pure, mask-aligned semantic representations. To fully leverage these features in large-scale continuous mapping, DISC is built upon a fully GPU-accelerated architecture that replaces periodic offline processing with precise, on-the-fly voxel-level instance refinement. We evaluate our approach on standard benchmarks (Replica, ScanNet) and a newly generated large-scale-mapping dataset based on Habitat-Matterport 3D (HM3DSEM) to assess scalability across complex scenes in multi-story buildings. Extensive evaluations demonstrate that DISC significantly surpasses current state-of-the-art zero-shot methods in both semantic accuracy and query retrieval, providing a robust, real-time capable framework for robotic deployment. The full source code, data generation and evaluation pipelines will be made available at https://github.com/DFKI-NI/DISC.

  </details>



- **Architecture and evaluation protocol for transformer-based visual object tracking in UAV applications**  
  Augustin Borne, Pierre Notin, Christophe Hennequin, Sebastien Changey, Stephane Bazeille, Christophe Cudel, Franz Quint  
  _2026-03-04_ · https://arxiv.org/abs/2603.03904v1  
  <details><summary>Abstract</summary>

  Object tracking from Unmanned Aerial Vehicles (UAVs) is challenged by platform dynamics, camera motion, and limited onboard resources. Existing visual trackers either lack robustness in complex scenarios or are too computationally demanding for real-time embedded use. We propose an Modular Asynchronous Tracking Architecture (MATA) that combines a transformer-based tracker with an Extended Kalman Filter, integrating ego-motion compensation from sparse optical flow and an object trajectory model. We further introduce a hardware-independent, embedded oriented evaluation protocol and a new metric called Normalized time to Failure (NT2F) to quantify how long a tracker can sustain a tracking sequence without external help. Experiments on UAV benchmarks, including an augmented UAV123 dataset with synthetic occlusions, show consistent improvements in Success and NT2F metrics across multiple tracking processing frequency. A ROS 2 implementation on a Nvidia Jetson AGX Orin confirms that the evaluation protocol more closely matches real-time performance on embedded systems.

  </details>



- **From Misclassifications to Outliers: Joint Reliability Assessment in Classification**  
  Yang Li, Youyang Sha, Yinzhi Wang, Timothy Hospedales, Xi Shen, Shell Xu Hu, Xuanlong Yu  
  _2026-03-04_ · https://arxiv.org/abs/2603.03903v1  
  <details><summary>Abstract</summary>

  Building reliable classifiers is a fundamental challenge for deploying machine learning in real-world applications. A reliable system should not only detect out-of-distribution (OOD) inputs but also anticipate in-distribution (ID) errors by assigning low confidence to potentially misclassified samples. Yet, most prior work treats OOD detection and failure prediction as separated problems, overlooking their closed connection. We argue that reliability requires evaluating them jointly. To this end, we propose a unified evaluation framework that integrates OOD detection and failure prediction, quantified by our new metrics DS-F1 and DS-AURC, where DS denotes double scoring functions. Experiments on the OpenOOD benchmark show that double scoring functions yield classifiers that are substantially more reliable than traditional single scoring approaches. Our analysis further reveals that OOD-based approaches provide notable gains under simple or far-OOD shifts, but only marginal benefits under more challenging near-OOD conditions. Beyond evaluation, we extend the reliable classifier SURE and introduce SURE+, a new approach that significantly improves reliability across diverse scenarios. Together, our framework, metrics, and method establish a new benchmark for trustworthy classification and offer practical guidance for deploying robust models in real-world settings. The source code is publicly available at https://github.com/Intellindust-AI-Lab/SUREPlus.

  </details>



- **UniSync: Towards Generalizable and High-Fidelity Lip Synchronization for Challenging Scenarios**  
  Ruidi Fan, Yang Zhou, Siyuan Wang, Tian Yu, Yutong Jiang, Xusheng Liu  
  _2026-03-04_ · https://arxiv.org/abs/2603.03882v1  
  <details><summary>Abstract</summary>

  Lip synchronization aims to generate realistic talking videos that match given audio, which is essential for high-quality video dubbing. However, current methods have fundamental drawbacks: mask-based approaches suffer from local color discrepancies, while mask-free methods struggle with global background texture misalignment. Furthermore, most methods struggle with diverse real-world scenarios such as stylized avatars, face occlusion, and extreme lighting conditions. In this paper, we propose UniSync, a unified framework designed for achieving high-fidelity lip synchronization in diverse scenarios. Specifically, UniSync uses a mask-free pose-anchored training strategy to keep head motion and eliminate synthesis color artifacts, while employing mask-based blending consistent inference to ensure structural precision and smooth blending. Notably, fine-tuning on compact but diverse videos empowers our model with exceptional domain adaptability, handling complex corner cases effectively. We also introduce the RealWorld-LipSync benchmark to evaluate models under real-world demands, which covers diverse application scenarios including both human faces and stylized avatars. Extensive experiments demonstrate that UniSync significantly outperforms state-of-the-art methods, advancing the field towards truly generalizable and production-ready lip synchronization.

  </details>



- **Bridging Human Evaluation to Infrared and Visible Image Fusion**  
  Jinyuan Liu, Xingyuan Li, Qingyun Mei, Haoyuan Xu, Zhiying Jiang, Long Ma, Risheng Liu, Xin Fan  
  _2026-03-04_ · https://arxiv.org/abs/2603.03871v1  
  <details><summary>Abstract</summary>

  Infrared and visible image fusion (IVIF) integrates complementary modalities to enhance scene perception. Current methods predominantly focus on optimizing handcrafted losses and objective metrics, often resulting in fusion outcomes that do not align with human visual preferences. This challenge is further exacerbated by the ill-posed nature of IVIF, which severely limits its effectiveness in human perceptual environments such as security surveillance and driver assistance systems. To address these limitations, we propose a feedback reinforcement framework that bridges human evaluation to infrared and visible image fusion. To address the lack of human-centric evaluation metrics and data, we introduce the first large-scale human feedback dataset for IVIF, containing multidimensional subjective scores and artifact annotations, and enriched by a fine-tuned large language model with expert review. Based on this dataset, we design a domain-specific reward function and train a reward model to quantify perceptual quality. Guided by this reward, we fine-tune the fusion network through Group Relative Policy Optimization, achieving state-of-the-art performance that better aligns fused images with human aesthetics. Code is available at https://github.com/ALKA-Wind/EVAFusion.

  </details>



- **Universal Pansharpening Foundation Model**  
  Hebaixu Wang, Jing Zhang, Haonan Guo, Di Wang, Jiayi Ma, Bo Du, Liangpei Zhang  
  _2026-03-04_ · https://arxiv.org/abs/2603.03831v1  
  <details><summary>Abstract</summary>

  Pansharpening generates the high-resolution multi-spectral (MS) image by integrating spatial details from a texture-rich panchromatic (PAN) image and spectral attributes from a low-resolution MS image. Existing methods are predominantly satellite-specific and scene-dependent, which severely limits their generalization across heterogeneous sensors and varied scenes, thereby reducing their real-world practicality. To address these challenges, we present FoundPS, a universal pansharpening foundation model for satellite-agnostic and scene-robust fusion. Specifically, we introduce a modality-interleaved transformer that learns band-wise modal specializations to form reversible spectral affine bases, mapping arbitrary-band MS into a unified latent space via tensor multiplication. Building upon this, we construct a latent diffusion bridge model to progressively evolve latent representations, and incorporate bridge posterior sampling to couple latent diffusion with pixel-space observations, enabling stable and controllable fusion. Furthermore, we devise infinite-dimensional pixel-to-latent interaction mechanisms to comprehensively capture the cross-domain dependencies between PAN observations and MS representations, thereby facilitating complementary information fusion. In addition, to support large-scale training and evaluation, we construct a comprehensive pansharpening benchmark, termed PSBench, consisting of worldwide MS and PAN image pairs from multiple satellites across diverse scenes. Extensive experiments demonstrate that FoundPS consistently outperforms state-of-the-art methods, exhibiting superior generalization and robustness across a wide range of pansharpening tasks.

  </details>



- **Vector-Quantized Soft Label Compression for Dataset Distillation**  
  Ali Abbasi, Ashkan Shahbazi, Hamed Pirsiavash, Soheil Kolouri  
  _2026-03-04_ · https://arxiv.org/abs/2603.03808v1  
  <details><summary>Abstract</summary>

  Dataset distillation is an emerging technique for reducing the computational and storage costs of training machine learning models by synthesizing a small, informative subset of data that captures the essential characteristics of a much larger dataset. Recent methods pair synthetic samples and their augmentations with soft labels from a teacher model, enabling student models to generalize effectively despite the small size of the distilled dataset. While soft labels are critical for effective distillation, the storage and communication overhead they incur, especially when accounting for augmentations, is often overlooked. In practice, each distilled sample is associated with multiple soft labels, making them the dominant contributor to storage costs, particularly in large-class settings such as ImageNet-1K. In this paper, we present a rigorous analysis of bit requirements across dataset distillation frameworks, quantifying the storage demands of both distilled samples and their soft labels. To address the overhead, we introduce a vector-quantized autoencoder (VQAE) for compressing soft labels, achieving substantial compression while preserving the effectiveness of the distilled data. We validate our method on both vision and language distillation benchmarks. On ImageNet-1K, our proposed VQAE achieves 30--40x additional compression over RDED, LPLD, SRE2L, and CDA baselines while retaining over $90\%$ of their original performance.

  </details>



- **Adaptive Enhancement and Dual-Pooling Sequential Attention for Lightweight Underwater Object Detection with YOLOv10**  
  Md. Mushibur Rahman, Umme Fawzia Rahim, Enam Ahmed Taufik  
  _2026-03-04_ · https://arxiv.org/abs/2603.03807v1  
  <details><summary>Abstract</summary>

  Underwater object detection constitutes a pivotal endeavor within the realms of marine surveillance and autonomous underwater systems; however, it presents significant challenges due to pronounced visual impairments arising from phenomena such as light absorption, scattering, and diminished contrast. In response to these formidable challenges, this manuscript introduces a streamlined yet robust framework for underwater object detection, grounded in the YOLOv10 architecture. The proposed method integrates a Multi-Stage Adaptive Enhancement module to improve image quality, a Dual-Pooling Sequential Attention (DPSA) mechanism embedded into the backbone to strengthen multi-scale feature representation, and a Focal Generalized IoU Objectness (FGIoU) loss to jointly improve localization accuracy and objectness prediction under class imbalance. Comprehensive experimental evaluations conducted on the RUOD and DUO benchmark datasets substantiate that the proposed DPSA_FGIoU_YOLOv10n attains exceptional performance, achieving mean Average Precision (mAP) scores of 88.9% and 88.0% at IoU threshold 0.5, respectively. In comparison to the baseline YOLOv10n, this represents enhancements of 6.7% for RUOD and 6.2% for DUO, all while preserving a compact model architecture comprising merely 2.8M parameters. These findings validate that the proposed framework establishes an efficacious equilibrium among accuracy, robustness, and real-time operational efficiency, making it suitable for deployment in resource-constrained underwater settings.

  </details>


