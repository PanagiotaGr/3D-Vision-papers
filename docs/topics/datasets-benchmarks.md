# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-02 07:39 UTC_

Total papers shown: **50**


---

- **HippoCamp: Benchmarking Contextual Agents on Personal Computers**  
  Zhe Yang, Shulin Tian, Kairui Hu, Shuai Liu, Hoang-Nhat Nguyen, Yichi Zhang, Zujin Guo, Mengying Yu, Zinan Zhang, Jingkang Yang, et al.  
  _2026-04-01_ · https://arxiv.org/abs/2604.01221v1  
  <details><summary>Abstract</summary>

  We present HippoCamp, a new benchmark designed to evaluate agents' capabilities on multimodal file management. Unlike existing agent benchmarks that focus on tasks like web interaction, tool use, or software automation in generic settings, HippoCamp evaluates agents in user-centric environments to model individual user profiles and search massive personal files for context-aware reasoning. Our benchmark instantiates device-scale file systems over real-world profiles spanning diverse modalities, comprising 42.4 GB of data across over 2K real-world files. Building upon the raw files, we construct 581 QA pairs to assess agents' capabilities in search, evidence perception, and multi-step reasoning. To facilitate fine-grained analysis, we provide 46.1K densely annotated structured trajectories for step-wise failure diagnosis. We evaluate a wide range of state-of-the-art multimodal large language models (MLLMs) and agentic methods on HippoCamp. Our comprehensive experiments reveal a significant performance gap: even the most advanced commercial models achieve only 48.3% accuracy in user profiling, struggling particularly with long-horizon retrieval and cross-modal reasoning within dense personal file systems. Furthermore, our step-wise failure diagnosis identifies multimodal perception and evidence grounding as the primary bottlenecks. Ultimately, HippoCamp exposes the critical limitations of current agents in realistic, user-centric environments and provides a robust foundation for developing next-generation personal AI assistants.

  </details>



- **Collaborative Task and Path Planning for Heterogeneous Robotic Teams using Multi-Agent PPO**  
  Matthias Rubio, Julia Richter, Hendrik Kolvenbach, Marco Hutter  
  _2026-04-01_ · https://arxiv.org/abs/2604.01213v1  
  <details><summary>Abstract</summary>

  Efficient robotic extraterrestrial exploration requires robots with diverse capabilities, ranging from scientific measurement tools to advanced locomotion. A robotic team enables the distribution of tasks over multiple specialized subsystems, each providing specific expertise to complete the mission. The central challenge lies in efficiently coordinating the team to maximize utilization and the extraction of scientific value. Classical planning algorithms scale poorly with problem size, leading to long planning cycles and high inference costs due to the combinatorial growth of possible robot-target allocations and possible trajectories. Learning-based methods are a viable alternative that move the scaling concern from runtime to training time, setting a critical step towards achieving real-time planning. In this work, we present a collaborative planning strategy based on Multi-Agent Proximal Policy Optimization (MAPPO) to coordinate a team of heterogeneous robots to solve a complex target allocation and scheduling problem. We benchmark our approach against single-objective optimal solutions obtained through exhaustive search and evaluate its ability to perform online replanning in the context of a planetary exploration scenario.

  </details>



- **TRACE: High-Fidelity 3D Scene Editing via Tangible Reconstruction and Geometry-Aligned Contextual Video Masking**  
  Jiyuan Hu, Zechuan Zhang, Zongxin Yang, Yi Yang  
  _2026-04-01_ · https://arxiv.org/abs/2604.01207v1  
  <details><summary>Abstract</summary>

  We present TRACE, a mesh-guided 3DGS editing framework that achieves automated, high-fidelity scene transformation. By anchoring video diffusion with explicit 3D geometry, TRACE uniquely enables fine-grained, part-level manipulatio--such as local pose shifting or component replacemen--while preserving the structural integrity of the central subject, a capability largely absent in existing editing methods. Our approach comprises three key stages: (1) Multi-view 3D-Anchor Synthesis, which leverages a sparse-view editor trained on our MV-TRACE datase--the first multi-view consistent dataset dedicated to scene-coherent object addition and modificatio--to generate spatially consistent 3D-anchors; (2) Tangible Geometry Anchoring (TGA), which ensures precise spatial synchronization between inserted meshes and the 3DGS scene via two-phase registration; and (3) Contextual Video Masking (CVM), which integrates 3D projections into an autoregressive video pipeline to achieve temporally stable, physically-grounded rendering. Extensive experiments demonstrate that TRACE consistently outperforms existing methods especially in editing versatility and structural integrity.

  </details>



- **True (VIS) Lies: Analyzing How Generative AI Recognizes Intentionality, Rhetoric, and Misleadingness in Visualization Lies**  
  Graziano Blasilli, Marco Angelini  
  _2026-04-01_ · https://arxiv.org/abs/2604.01181v1  
  <details><summary>Abstract</summary>

  This study investigates the ability of multimodal Large Language Models (LLMs) to identify and interpret misleading visualizations, and recognize these observations along with their underlying causes and potential intentionality. Our analysis leverages concepts from visualization rhetoric and a newly developed taxonomy of authorial intents as explanatory lenses. We formulated three research questions and addressed them experimentally using a dataset of 2,336 COVID-19-related tweets, half of which contain misleading visualizations, and supplemented it with real-world examples of perceptual, cognitive, and conceptual errors drawn from VisLies, the IEEE VIS community event dedicated to showcasing deceptive and misleading visualizations. To ensure broad coverage of the current LLM landscape, we evaluated 16 state-of-the-art models. Among them, 15 are open-weight models, spanning a wide range of model sizes, architectural families, and reasoning capabilities. The selection comprises small models, namely Nemotron-Nano-V2-VL (12B parameters), Mistral-Small-3.2 (24B), DeepSeek-VL2 (27B), Gemma3 (27B), and GTA1 (32B); medium-sized models, namely Qianfan-VL (70B), Molmo (72B), GLM-4.5V (108B), LLaVA-NeXT (110B), and Pixtral-Large (124B); and large models, namely Qwen3-VL (235B), InternVL3.5 (241B), Step3 (321B), Llama-4-Maverick (400B), and Kimi-K2.5 (1000B). In addition, we employed OpenAI GPT-5.4, a frontier proprietary model. To establish a human perspective on these tasks, we also conducted a user study with visualization experts to assess how people perceive rhetorical techniques and the authorial intentions behind the same misleading visualizations. This allows comparison between model and expert behavior, revealing similarities and differences that provide insights into where LLMs align with human judgment and where they diverge.

  </details>



- **Open-Set Supervised 3D Anomaly Detection: An Industrial Dataset and a Generalisable Framework for Unknown Defects**  
  Hanzhe Liang, Luocheng Zhang, Junyang Xia, HanLiang Zhou, Bingyang Guo, Yingxi Xie, Can Gao, Ruiyun Yu, Jinbao Wang, Pan Li  
  _2026-04-01_ · https://arxiv.org/abs/2604.01171v1  
  <details><summary>Abstract</summary>

  Although self-supervised 3D anomaly detection assumes that acquiring high-precision point clouds is computationally expensive, in real manufacturing scenarios it is often feasible to collect a limited number of anomalous samples. Therefore, we study open-set supervised 3D anomaly detection, where the model is trained with only normal samples and a small number of known anomalous samples, aiming to identify unknown anomalies at test time. We present Open-Industry, a high-quality industrial dataset containing 15 categories, each with five real anomaly types collected from production lines. We first adapt general open-set anomaly detection methods to accommodate 3D point cloud inputs better. Building upon this, we propose Open3D-AD, a point-cloud-oriented approach that leverages normal samples, simulated anomalies, and partially observed real anomalies to model the probability density distributions of normal and anomalous data. Then, we introduce a simple Correspondence Distributions Subsampling to reduce the overlap between normal and non-normal distributions, enabling stronger dual distributions modeling. Based on these contributions, we establish a comprehensive benchmark and evaluate the proposed method extensively on Open-Industry as well as established datasets including Real3D-AD and Anomaly-ShapeNet. Benchmark results and ablation studies demonstrate the effectiveness of Open3D-AD and further reveal the potential of open-set supervised 3D anomaly detection.

  </details>



- **VRUD: A Drone Dataset for Complex Vehicle-VRU Interactions within Mixed Traffic**  
  Ziyu Wang, Hongrui Kou, Cheng Wang, Ruochen Li, Hubert P. H. Shum, Amir Atapour-Abarghouei, Yuxin Zhang  
  _2026-04-01_ · https://arxiv.org/abs/2604.01134v1  
  <details><summary>Abstract</summary>

  The Operational Design Domain (ODD) of urbanoriented Level 4 (L4) autonomous driving, especially for autonomous robotaxis, confronts formidable challenges in complex urban mixed traffic environments. These challenges stem mainly from the high density of Vulnerable Road Users (VRUs) and their highly uncertain and unpredictable interaction behaviors. However, existing open-source datasets predominantly focus on structured scenarios such as highways or regulated intersections, leaving a critical gap in data representing chaotic, unstructured urban environments. To address this, this paper proposes an efficient, high-precision method for constructing drone-based datasets and establishes the Vehicle-Vulnerable Road User Interaction Dataset (VRUD), as illustrated in Figure 1. Distinct from prior works, VRUD is collected from typical "Urban Villages" in Shenzhen, characterized by loose traffic supervision and extreme occlusion. The dataset comprises 4 hours of 4K/30Hz recording, containing 11,479 VRU trajectories and 1,939 vehicle trajectories. A key characteristic of VRUD is its composition: VRUs account for about 87% of all traffic participants, significantly exceeding the proportions in existing benchmarks. Furthermore, unlike datasets that only provide raw trajectories, we extracted 4,002 multi-agent interaction scenarios based on a novel Vector Time to Collision (VTTC) threshold, supported by standard OpenDRIVE HD maps. This study provides valuable, rare edge-case resources for enhancing the safety performance of ADS in complex, unstructured urban environments. To facilitate further research, we have made the VRUD dataset open-source at: https://zzi4.github.io/VRUD/.

  </details>



- **Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation**  
  Reyhaneh Ahani Manghotay, Jie Liang  
  _2026-04-01_ · https://arxiv.org/abs/2604.01118v1  
  <details><summary>Abstract</summary>

  Leveraging the rich semantic features of vision-language models (VLMs) like CLIP for monocular depth estimation tasks is a promising direction, yet often requires extensive fine-tuning or lacks geometric precision. We present a parameter-efficient framework, named MoA-DepthCLIP, that adapts pretrained CLIP representations for monocular depth estimation with minimal supervision. Our method integrates a lightweight Mixture-of-Adapters (MoA) module into the pretrained Vision Transformer (ViT-B/32) backbone combined with selective fine-tuning of the final layers. This design enables spatially-aware adaptation, guided by a global semantic context vector and a hybrid prediction architecture that synergizes depth bin classification with direct regression. To enhance structural accuracy, we employ a composite loss function that enforces geometric constraints. On the NYU Depth V2 benchmark, MoA-DepthCLIP achieves competitive results, significantly outperforming the DepthCLIP baseline by improving the $δ_1$ accuracy from 0.390 to 0.745 and reducing the RMSE from 1.176 to 0.520. These results are achieved while requiring substantially few trainable parameters, demonstrating that lightweight, prompt-guided MoA is a highly effective strategy for transferring VLM knowledge to fine-grained monocular depth estimation tasks.

  </details>



- **ProTPS: Prototype-Guided Text Prompt Selection for Continual Learning**  
  Jie Mei, Li-Leng Peng, Keith Fuller, Jenq-Neng Hwang  
  _2026-04-01_ · https://arxiv.org/abs/2604.01116v1  
  <details><summary>Abstract</summary>

  For continual learning, text-prompt-based methods leverage text encoders and learnable prompts to encode semantic features for sequentially arrived classes over time. A common challenge encountered by existing works is how to learn unique text prompts, which implicitly carry semantic information of new classes, so that the semantic features of newly arrived classes do not overlap with those of trained classes, thereby mitigating the catastrophic forgetting problem. To address this challenge, we propose a novel approach Prototype-guided Text Prompt Selection (ProTPS)'' to intentionally increase the training flexibility thus encouraging the learning of unique text prompts. Specifically, our ProTPS learns class-specific vision prototypes and text prompts. Vision prototypes guide the selection and learning of text prompts for each class. We first evaluate our ProTPS in both class incremental (CI) setting and cross-datasets continual (CDC) learning setting. Because our ProTPS achieves performance close to the upper bounds, we further collect a real-world dataset with 112 marine species collected over a span of six years, named Marine112, to bring new challenges to the community. Marine112 is authentically suited for the class and domain incremental (CDI) learning setting and is under natural long-tail distribution. The results under three settings show that our ProTPS performs favorably against the recent state-of-the-art methods. The implementation code and Marine112 dataset will be released upon the acceptance of our paper.

  </details>



- **TRACE: Training-Free Partial Audio Deepfake Detection via Embedding Trajectory Analysis of Speech Foundation Models**  
  Awais Khan, Muhammad Umar Farooq, Kutub Uddin, Khalid Malik  
  _2026-04-01_ · https://arxiv.org/abs/2604.01083v1  
  <details><summary>Abstract</summary>

  Partial audio deepfakes, where synthesized segments are spliced into genuine recordings, are particularly deceptive because most of the audio remains authentic. Existing detectors are supervised: they require frame-level annotations, overfit to specific synthesis pipelines, and must be retrained as new generative models emerge. We argue that this supervision is unnecessary. We hypothesize that speech foundation models implicitly encode a forensic signal: genuine speech forms smooth, slowly varying embedding trajectories, while splice boundaries introduce abrupt disruptions in frame-level transitions. Building on this, we propose TRACE (Training-free Representation-based Audio Countermeasure via Embedding dynamics), a training-free framework that detects partial audio deepfakes by analyzing the first-order dynamics of frozen speech foundation model representations without any training, labeled data, or architectural modification. We evaluate TRACE on four benchmarks that span two languages using six speech foundation models. In PartialSpoof, TRACE achieves 8.08% EER, competitive with fine-tuned supervised baselines. In LlamaPartialSpoof, the most challenging benchmark featuring LLM-driven commercial synthesis, TRACE surpasses a supervised baseline outright (24.12% vs. 24.49% EER) without any target-domain data. These results show that temporal dynamics in speech foundation models provide an effective, generalize signal for training-free audio forensics.

  </details>



- **A global dataset of continuous urban dashcam driving**  
  Md Shadab Alam, Olena Bazilinska, Pavlo Bazilinskyy  
  _2026-04-01_ · https://arxiv.org/abs/2604.01044v1  
  <details><summary>Abstract</summary>

  We introduce CROWD (City Road Observations With Dashcams), a manually curated dataset of ordinary, minute scale, temporally contiguous, unedited, front facing urban dashcam segments screened and segmented from publicly available YouTube videos. CROWD is designed to support cross-domain robustness and interaction analysis by prioritising routine driving and explicitly excluding crashes, crash aftermath, and other edited or incident-focused content. The release contains 51,753 segment records spanning 20,275.56 hours (42,032 videos), covering 7,103 named inhabited places in 238 countries and territories across all six inhabited continents (Africa, Asia, Europe, North America, South America and Oceania), with segment level manual labels for time of day (day or night) and vehicle type. To lower the barrier for benchmarking, we provide per-segment CSV files of machine-generated detections for all 80 MS-COCO classes produced with YOLOv11x, together with segment-local multi-object tracks (BoT-SORT); e.g. person, bicycle, motorcycle, car, bus, truck, traffic light, stop sign, etc. CROWD is distributed as video identifiers with segment boundaries and derived annotations, enabling reproducible research without redistributing the underlying videos.

  </details>



- **Foundation Model-guided Iteratively Prompting and Pseudo-Labeling for Partially Labeled Medical Image Segmentation**  
  Qiaochu Zhao, Wei Wei, David Horowitz, Richard Bakst, Yading Yuan  
  _2026-04-01_ · https://arxiv.org/abs/2604.01038v1  
  <details><summary>Abstract</summary>

  Automated medical image segmentation has achieved remarkable progress with fully labeled data. However, site-specific clinical priorities and the high cost of manual annotation often yield scans with only a subset of organs labeled, leading to the partially labeled problem that degrades performance. To address this issue, we propose IPnP, an Iteratively Prompting and Pseudo-labeling framework, for partially labeled medical image segmentation. IPnP iteratively generates and refines pseudo-labels for unlabeled organs through collaboration between a trainable segmentation network (specialist) and a frozen foundation model (generalist), progressively recovering full-organ supervision. On the public dataset AMOS with the simulated partial-label setting, IPnP consistently improves segmentation performance over prior methods and approaches the performance of the fully labeled reference. We further evaluate on a private, partially labeled dataset of 210 head-and-neck cancer patients and demonstrate our effectiveness in real-world clinical settings.

  </details>



- **Customizing Large Vision Model-Guided Low-Rank Approximation for Ground-Roll Denoise**  
  Jiacheng Liao, Feng Qian, Ziyin Fan, Yongjian Guo  
  _2026-04-01_ · https://arxiv.org/abs/2604.00998v1  
  <details><summary>Abstract</summary>

  Ground-roll is a dominant source of coherent noise in land and vertical seismic profiling (VSP) data, severely masking reflection events and degrading subsequent imaging and interpretation. Conventional attenuation methods, including transform-domain filtering, sparse representation, and deep learning, often suffer from limited adaptability, signal leakage, or dependence on labeled training data, especially under strong signal-noise overlap. To address these challenges, we propose a training-free framework that reformulates ground-roll attenuation as a semantic-guided signal separation problem. Specifically, a promptable large vision model is employed to extract high-level semantic priors by converting seismic gathers into visual representations and localizing ground-roll-dominant regions via text or image prompts. The resulting semantic response is transformed into a continuous soft mask, which is embedded into a mask-conditioned low-rank inverse formulation to enable spatially adaptive suppression and reflection-preserving reconstruction. An efficient alternating direction method of multipliers (ADMM)-based solver is further developed to solve the proposed inverse problem, enabling stable and physically consistent signal recovery without requiring task-specific training or manual annotation. Extensive experiments on both synthetic and field VSP datasets demonstrate that the proposed method achieves superior ground-roll attenuation while preserving reflection continuity and waveform fidelity, consistently outperforming representative transform-domain filtering and implicit neural representation methods.

  </details>



- **YieldSAT: A Multimodal Benchmark Dataset for High-Resolution Crop Yield Prediction**  
  Miro Miranda, Deepak Pathak, Patrick Helber, Benjamin Bischke, Hiba Najjar, Francisco Mena, Cristhian Sanchez, Akshay Pai, Diego Arenas, Matias Valdenegro-Toro, et al.  
  _2026-04-01_ · https://arxiv.org/abs/2604.00940v1  
  <details><summary>Abstract</summary>

  Crop yield prediction requires substantial data to train scalable models. However, creating yield prediction datasets is constrained by high acquisition costs, heterogeneous data quality, and data privacy regulations. Consequently, existing datasets are scarce, low in quality, or limited to regional levels or single crop types, hindering the development of scalable data-driven solutions. In this work, we release YieldSAT, a large, high-quality, and multimodal dataset for high-resolution crop yield prediction. YieldSAT spans various climate zones across multiple countries, including Argentina, Brazil, Uruguay, and Germany, and includes major crop types, including corn, rapeseed, soybeans, and wheat, across 2,173 expert-curated fields. In total, over 12.2 million yield samples are available, each with a spatial resolution of 10 m. Each field is paired with multispectral satellite imagery, resulting in 113,555 labeled satellite images, complemented by auxiliary environmental data. We demonstrate the potential of large-scale and high-resolution crop yield prediction as a pixel regression task by comparing various deep learning models and data fusion architectures. Furthermore, we highlight open challenges arising from severe distribution shifts in the ground truth data under real-world conditions. To mitigate this, we explore a domain-informed Deep Ensemble approach that exhibits significant performance gains. The dataset is available at https://yieldsat.github.io/.

  </details>



- **EmoScene: A Dual-space Dataset for Controllable Affective Image Generation**  
  Li He, Longtai Zhang, Wenqiang Zhang, Yan Wang, Lizhe Qi  
  _2026-04-01_ · https://arxiv.org/abs/2604.00933v1  
  <details><summary>Abstract</summary>

  Text-to-image diffusion models have achieved high visual fidelity, yet precise control over scene semantics and fine-grained affective tone remains challenging. Human visual affect arises from the rapid integration of contextual meaning, including valence, arousal, and dominance, with perceptual cues such as color harmony, luminance contrast, texture variation, curvature, and spatial layout. However, current text-to-image models rarely represent affective and perceptual factors within a unified representation, which limits their ability to synthesize scenes with coherent and nuanced emotional intent. To address this gap, we construct EmoScene, a large-scale dual-space emotion dataset that jointly encodes affective dimensions and perceptual attributes, with contextual semantics provided as supporting annotations. EmoScene contains 1.2M images across more than three hundred real-world scene categories, each annotated with discrete emotion labels, continuous VAD values, perceptual descriptors and textual captions. Multi-space analyses reveal how discrete emotions occupy the VAD space and how affect systematically correlates with scene-level perceptual factors. To benchmark EmoScene, we provide a lightweight reference baseline that injects dual-space controls into a frozen diffusion backbone via shallow cross-attention modulation, serving as a reproducible probe of affect controllability enabled by dual-space supervision.

  </details>



- **Learning Quantised Structure-Preserving Motion Representations for Dance Fingerprinting**  
  Arina Kharlamova, Bowei He, Chen Ma, Xue Liu  
  _2026-04-01_ · https://arxiv.org/abs/2604.00927v1  
  <details><summary>Abstract</summary>

  We present DANCEMATCH, an end-to-end framework for motion-based dance retrieval, the task of identifying semantically similar choreographies directly from raw video, defined as DANCE FINGERPRINTING. While existing motion analysis and retrieval methods can compare pose sequences, they rely on continuous embeddings that are difficult to index, interpret, or scale. In contrast, DANCEMATCH constructs compact, discrete motion signatures that capture the spatio-temporal structure of dance while enabling efficient large-scale retrieval. Our system integrates Skeleton Motion Quantisation (SMQ) with Spatio-Temporal Transformers (STT) to encode human poses, extracted via Apple CoMotion, into a structured motion vocabulary. We further design DANCE RETRIEVAL ENGINE (DRE), which performs sub-linear retrieval using a histogram-based index followed by re-ranking for refined matching. To facilitate reproducible research, we release DANCETYPESBENCHMARK, a pose-aligned dataset annotated with quantised motion tokens. Experiments demonstrate robust retrieval across diverse dance styles and strong generalisation to unseen choreographies, establishing a foundation for scalable motion fingerprinting and quantitative choreographic analysis.

  </details>



- **Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment**  
  Zhuchenyang Liu, Yao Zhang, Yu Xiao  
  _2026-04-01_ · https://arxiv.org/abs/2604.00913v1  
  <details><summary>Abstract</summary>

  2D assembly diagrams are often abstract and hard to follow, creating a need for intelligent assistants that can monitor progress, detect errors, and provide step-by-step guidance. In mixed reality settings, such systems must recognize completed and ongoing steps from the camera feed and align them with the diagram instructions. Vision Language Models (VLMs) show promise for this task, but face a depiction gap because assembly diagrams and video frames share few visual features. To systematically assess this gap, we construct IKEA-Bench, a benchmark of 1,623 questions across 6 task types on 29 IKEA furniture products, and evaluate 19 VLMs (2B-38B) under three alignment strategies. Our key findings: (1) assembly instruction understanding is recoverable via text, but text simultaneously degrades diagram-to-video alignment; (2) architecture family predicts alignment accuracy more strongly than parameter count; (3) video understanding remains a hard bottleneck unaffected by strategy. A three-level mechanistic analysis further reveals that diagrams and video occupy disjoint ViT subspaces, and that adding text shifts models from visual to text-driven reasoning. These results identify visual encoding as the primary target for improving cross-depiction robustness. Project page: https://ryenhails.github.io/IKEA-Bench/

  </details>



- **ProCap: Projection-Aware Captioning for Spatial Augmented Reality**  
  Zimo Cao, Yuchen Deng, Haibin Ling, Bingyao Huang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00912v1  
  <details><summary>Abstract</summary>

  Spatial augmented reality (SAR) directly projects digital content onto physical scenes using projectors, creating immersive experience without head-mounted displays. However, for SAR to support intelligent interaction, such as reasoning about the scene or answering user queries, it must semantically distinguish between the physical scene and the projected content. Standard Vision Language Models (VLMs) struggle with this virtual-physical ambiguity, often confusing the two contexts. To address this issue, we introduce ProCap, a novel framework that explicitly decouples projected content from physical scenes. ProCap employs a two-stage pipeline: first it visually isolates virtual and physical layers via automated segmentation; then it uses region-aware retrieval to avoid ambiguous semantic context due to projection distortion. To support this, we present RGBP (RGB + Projections), the first large-scale SAR semantic benchmark dataset, featuring 65 diverse physical scenes and over 180,000 projections with dense, decoupled annotations. Finally, we establish a dual-captioning evaluation protocol using task-specific tokens to assess physical scene and projection descriptions independently. Our experiments show that ProCap provides a robust semantic foundation for future SAR research. The source code, pre-trained models and the RGBP dataset are available on the project page: https://ZimoCao.github.io/ProCap/.

  </details>



- **JAMMEval: A Refined Collection of Japanese Benchmarks for Reliable VLM Evaluation**  
  Issa Sugiura, Koki Maeda, Shuhei Kurita, Yusuke Oda, Daisuke Kawahara, Naoaki Okazaki  
  _2026-04-01_ · https://arxiv.org/abs/2604.00909v1  
  <details><summary>Abstract</summary>

  Reliable evaluation is essential for the development of vision-language models (VLMs). However, Japanese VQA benchmarks have undergone far less iterative refinement than their English counterparts. As a result, many existing benchmarks contain issues such as ambiguous questions, incorrect answers, and instances that can be solved without visual grounding, undermining evaluation reliability and leading to misleading conclusions in model comparisons. To address these limitations, we introduce JAMMEval, a refined collection of Japanese benchmarks for reliable VLM evaluation. It is constructed by systematically refining seven existing Japanese benchmark datasets through two rounds of human annotation, improving both data quality and evaluation reliability. In our experiments, we evaluate open-weight and proprietary VLMs on JAMMEval and analyze the capabilities of recent models on Japanese VQA. We further demonstrate the effectiveness of our refinement by showing that the resulting benchmarks yield evaluation scores that better reflect model capability, exhibit lower run-to-run variance, and improve the ability to distinguish between models of different capability levels. We release our dataset and code to advance reliable evaluation of VLMs.

  </details>



- **A 4D Representation for Training-Free Agentic Reasoning from Monocular Laparoscopic Video**  
  Maximilian Fehrentz, Nicolas Stellwag, Robert Wiebe, Nicole Thorisch, Fabian Grob, Patrick Remerscheid, Ken-Joel Simmoteit, Benjamin D. Killeen, Christian Heiliger, Nassir Navab  
  _2026-04-01_ · https://arxiv.org/abs/2604.00867v1  
  <details><summary>Abstract</summary>

  Spatiotemporal reasoning is a fundamental capability for artificial intelligence (AI) in soft tissue surgery, paving the way for intelligent assistive systems and autonomous robotics. While 2D vision-language models show increasing promise at understanding surgical video, the spatial complexity of surgical scenes suggests that reasoning systems may benefit from explicit 4D representations. Here, we propose a framework for equipping surgical agents with spatiotemporal tools based on an explicit 4D representation, enabling AI systems to ground their natural language reasoning in both time and 3D space. Leveraging models for point tracking, depth, and segmentation, we develop a coherent 4D model with spatiotemporally consistent tool and tissue semantics. A Multimodal Large Language Model (MLLM) then acts as an agent on tools derived from the explicit 4D representation (e.g., trajectories) without any fine-tuning. We evaluate our method on a new dataset of 134 clinically relevant questions and find that the combination of a general purpose reasoning backbone and our 4D representation significantly improves spatiotemporal understanding and allows for 4D grounding. We demonstrate that spatiotemporal intelligence can be "assembled" from 2D MLLMs and 3D computer vision models without additional training. Code, data, and examples are available at https://tum-ai.github.io/surg4d/

  </details>



- **Perturb-and-Restore: Simulation-driven Structural Augmentation Framework for Imbalance Chromosomal Anomaly Detection**  
  Yilan Zhang, Hanbiao Chen, Changchun Yang, Yuetan Chu, Siyuan Chen, Jing Wu, Jingdong Hu, Na Li, Junkai Su, Yuxuan Chen, et al.  
  _2026-04-01_ · https://arxiv.org/abs/2604.00854v1  
  <details><summary>Abstract</summary>

  Detecting structural chromosomal abnormalities is crucial for accurate diagnosis and management of genetic disorders. However, collecting sufficient structural abnormality data is extremely challenging and costly in clinical practice, and not all abnormal types can be readily collected. As a result, deep learning approaches face significant performance degradation due to the severe imbalance and scarcity of abnormal chromosome data. To address this challenge, we propose a Perturb-and-Restore (P&R), a simulation-driven structural augmentation framework that effectively alleviates data imbalance in chromosome anomaly detection. The P&R framework comprises two key components: (1) Structure Perturbation and Restoration Simulation, which generates synthetic abnormal chromosomes by perturbing chromosomal banding patterns of normal chromosomes followed by a restoration diffusion network that reconstructs continuous chromosome content and edges, thus eliminating reliance on rare abnormal samples; and (2) Energy-guided Adaptive Sampling, an energy score-based online selection strategy that dynamically prioritizes high-quality synthetic samples by referencing the energy distribution of real samples. To evaluate our method, we construct a comprehensive structural anomaly dataset consisting of over 260,000 chromosome images, including 4,242 abnormal samples spanning 24 categories. Experimental results demonstrate that the P&R framework achieves state-of-the-art (SOTA) performance, surpassing existing methods with an average improvement of 8.92% in sensitivity, 8.89% in precision, and 13.79% in F1-score across all categories.

  </details>



- **PanoAir: A Panoramic Visual-Inertial SLAM with Cross-Time Real-World UAV Dataset**  
  Yiyang Wu, Xiaohu Zhang, Yanjin Du, Tongsu Zhang, Chujun Li, Siyang Chen, Guoyi Zhang, Xiangpeng Xu  
  _2026-04-01_ · https://arxiv.org/abs/2604.00852v1  
  <details><summary>Abstract</summary>

  Accurate pose estimation is fundamental for unmanned aerial vehicle (UAV) applications, where Visual-Inertial SLAM (VI-SLAM) provides a cost-effective solution for localization and mapping. However, existing VI-SLAM methods mainly rely on sensors with limited fields of view (FoV), which can lead to drift and even failure in complex UAV scenarios. Although panoramic cameras provide omnidirectional perception to improve robustness, panoramic VI-SLAM and corresponding real-world datasets for UAVs remain underexplored. To address this limitation, we first construct a real-world panoramic visual-inertial dataset covering diverse flight conditions, including varying illumination, altitudes, trajectory lengths, and motion dynamics. To achieve accurate and robust pose estimation under such challenging UAV scenarios, we propose a panoramic VI-SLAM framework that exploits the omnidirectional FoV via the proposed panoramic feature extraction and panoramic loop closure, enhancing feature constraints and ensuring global consistency. Extensive experiments on both the proposed dataset and public benchmarks demonstrate that our method achieves superior accuracy, robustness, and consistency compared to existing approaches. Moreover, deployment on embedded platform validates its practical applicability, achieving comparable computational efficiency to PC implementations. The source code and dataset are publicly available at https://drive.google.com/file/d/1lG1Upn6yi-N6tYpEHAt6dfR1uhzNtWbT/view

  </details>



- **Video Patch Pruning: Efficient Video Instance Segmentation via Early Token Reduction**  
  Patrick Glandorf, Thomas Norrenbrock, Bodo Rosenhahn  
  _2026-04-01_ · https://arxiv.org/abs/2604.00827v1  
  <details><summary>Abstract</summary>

  Vision Transformers (ViTs) have demonstrated state-ofthe-art performance in several benchmarks, yet their high computational costs hinders their practical deployment. Patch Pruning offers significant savings, but existing approaches restrict token reduction to deeper layers, leaving early-stage compression unexplored. This limits their potential for holistic efficiency. In this work, we present a novel Video Patch Pruning framework (VPP) that integrates temporal prior knowledge to enable efficient sparsity within early ViT layers. Our approach is motivated by the observation that prior features extracted from deeper layers exhibit strong foreground selectivity. Therefore we propose a fully differentiable module for temporal mapping to accurately select the most relevant patches in early network stages. Notably, the proposed method enables a patch reduction of up to 60% in dense prediction tasks, exceeding the capabilities of conventional image-based patch pruning, which typically operate around a 30% patch sparsity. VPP excels the high-sparsity regime, sustaining remarkable performance even when patch usage is reduced below 55%. Specifically, it preserves stable results with a maximal performance drop of 0.6% on the Youtube-VIS 2021 dataset.

  </details>



- **Continual Vision-Language Learning for Remote Sensing: Benchmarking and Analysis**  
  Xingxing Weng, Ruifeng Ni, Chao Pang, XiangYu Hao, Yishan Wang, Xiaokang Zhang, Wei Xu, Gui-Song Xia  
  _2026-04-01_ · https://arxiv.org/abs/2604.00820v1  
  <details><summary>Abstract</summary>

  Current remote sensing vision-language models (RS VLMs) demonstrate impressive performance in image interpretation but rely on static training data, limiting their ability to accommodate continuously emerging sensing modalities and downstream tasks. This exposes a fundamental challenge: enabling RS VLMs to continually adapt without catastrophic forgetting. Despite its practical importance, the continual learning capability of RS VLMs remains underexplored, and no dedicated benchmark currently exists. In this work, we present CLeaRS, a comprehensive benchmark for continual vision-language learning in remote sensing. CLeaRS comprises 10 curated subsets with over 207k image-text pairs, spanning diverse interpretation tasks, sensing modalities, and application scenarios. We further define three evaluation protocols: long-horizon, modality-incremental, and task-incremental settings, to systematically assess continual adaptation. Extensive benchmarking of diverse vision-language models reveals catastrophic forgetting across all settings. Moreover, representative continual learning methods, when adapted to RS VLMs, exhibit limited effectiveness in handling task, instruction, and modality transitions. Our findings underscore the need for developing continual learning methods tailored to RS VLMs.

  </details>



- **Revisiting Human-in-the-Loop Object Retrieval with Pre-Trained Vision Transformers**  
  Kawtar Zaher, Olivier Buisson, Alexis Joly  
  _2026-04-01_ · https://arxiv.org/abs/2604.00809v1  
  <details><summary>Abstract</summary>

  Building on existing approaches, we revisit Human-in-the-Loop Object Retrieval, a task that consists of iteratively retrieving images containing objects of a class-of-interest, specified by a user-provided query. Starting from a large unlabeled image collection, the aim is to rapidly identify diverse instances of an object category relying solely on the initial query and the user's Relevance Feedback, with no prior labels. The retrieval process is formulated as a binary classification task, where the system continuously learns to distinguish between relevant and non-relevant images to the query, through iterative user interaction. This interaction is guided by an Active Learning loop: at each iteration, the system selects informative samples for user annotation, thereby refining the retrieval performance. This task is particularly challenging in multi-object datasets, where the object of interest may occupy only a small region of the image within a complex, cluttered scene. Unlike object-centered settings where global descriptors often suffice, multi-object images require more adapted, localized descriptors. In this work, we formulate and revisit the Human-in-the-Loop Object Retrieval task by leveraging pre-trained ViT representations, and addressing key design questions, including which object instances to consider in an image, what form the annotations should take, how Active Selection should be applied, and which representation strategies best capture the object's features. We compare several representation strategies across multi-object datasets highlighting trade-offs between capturing the global context and focusing on fine-grained local object details. Our results offer practical insights for the design of effective interactive retrieval pipelines based on Active Learning for object class retrieval.

  </details>



- **Compact Keyframe-Optimized Multi-Agent Gaussian Splatting SLAM**  
  Monica M. Q. Li, Pierre-Yves Lajoie, Jialiang Liu, Giovanni Beltrame  
  _2026-04-01_ · https://arxiv.org/abs/2604.00804v1  
  <details><summary>Abstract</summary>

  Efficient multi-agent 3D mapping is essential for robotic teams operating in unknown environments, but dense representations hinder real-time exchange over constrained communication links. In multi-agent Simultaneous Localization and Mapping (SLAM), systems typically rely on a centralized server to merge and optimize the local maps produced by individual agents. However, sharing these large map representations, particularly those generated by recent methods such as Gaussian Splatting, becomes a bottleneck in real-world scenarios with limited bandwidth. We present an improved multi-agent RGB-D Gaussian Splatting SLAM framework that reduces communication load while preserving map fidelity. First, we incorporate a compaction step into our SLAM system to remove redundant 3D Gaussians, without degrading the rendering quality. Second, our approach performs centralized loop closure computation without initial guess, operating in two modes: a pure rendered-depth mode that requires no data beyond the 3D Gaussians, and a camera-depth mode that includes lightweight depth images for improved registration accuracy and additional Gaussian pruning. Evaluation on both synthetic and real-world datasets shows up to 85-95\% reduction in transmitted data compared to state-of-the-art approaches in both modes, bringing 3D Gaussian multi-agent SLAM closer to practical deployment in real-world scenarios. Code: https://github.com/lemonci/coko-slam

  </details>



- **HICT: High-precision 3D CBCT reconstruction from a single X-ray**  
  Wen Ma, Jiaxiang Liu, Zikai Xiao, Ziyang Wang, Feng Yang, Zuozhu Liu  
  _2026-04-01_ · https://arxiv.org/abs/2604.00792v1  
  <details><summary>Abstract</summary>

  Accurate 3D dental imaging is vital for diagnosis and treatment planning, yet CBCT's high radiation dose and cost limit its accessibility. Reconstructing 3D volumes from a single low-dose panoramic X-ray is a promising alternative but remains challenging due to geometric inconsistencies and limited accuracy. We propose HiCT, a two-stage framework that first generates geometrically consistent multi-view projections from a single panoramic image using a video diffusion model, and then reconstructs high-fidelity CBCT from the projections using a ray-based dynamic attention network and an X-ray sampling strategy. To support this, we built XCT, a large-scale dataset combining public CBCT data with 500 paired PX-CBCT cases. Extensive experiments show that HiCT achieves state-of-the-art performance, delivering accurate and geometrically consistent reconstructions for clinical use.

  </details>



- **An Approach to Enriching Surgical Video Datasets for Fine-Grained Spatial-Temporal Understanding of Vision-Language Models**  
  Lennart Maack, Alexander Schlaefer  
  _2026-04-01_ · https://arxiv.org/abs/2604.00784v1  
  <details><summary>Abstract</summary>

  Surgical video understanding is a crucial prerequisite for advancing Computer-Assisted Surgery. While vision-language models (VLMs) have recently been applied to the surgical domain, existing surgical vision-language datasets lack in capturing and evaluating complex, interleaved spatial-temporal dynamics. Creating large scale datasets that accurately represent fine-grained spatial-temporal relationships in surgical videos is challenging due to costly manual annotations or error-prone generation using large language models. To address this gap, we introduce the SurgSTU-Pipeline, a deterministic generation pipeline featuring temporal and spatial continuity filtering to reliably create surgical datasets for fine-grained spatial-temporal multimodal understanding. Applying this pipeline to publicly available surgical datasets, we create the SurgSTU dataset, comprising 7515 video clips densely extended with 150k fine-grained spatial-temporal question-answer samples. Our comprehensive evaluation shows that while state-of-the-art generalist VLMs struggle in zero-shot settings, their spatial-temporal capabilities can be improved through in-context learning. A fine-tuned VLM on the SurgSTU training dataset achieves highest performance among all spatial-temporal tasks, validating the dataset's efficacy to improve spatial-temporal understanding of VLMs in surgical videos. Code will be made publicly available.

  </details>



- **PrivHAR-Bench: A Graduated Privacy Benchmark Dataset for Video-Based Action Recognition**  
  Samar Ansari  
  _2026-04-01_ · https://arxiv.org/abs/2604.00761v1  
  <details><summary>Abstract</summary>

  Existing research on privacy-preserving Human Activity Recognition (HAR) typically evaluates methods against a binary paradigm: clear video versus a single privacy transformation. This limits cross-method comparability and obscures the nuanced relationship between privacy strength and recognition utility. We introduce \textit{PrivHAR-Bench}, a multi-tier benchmark dataset designed to standardize the evaluation of the \textit{Privacy-Utility Trade-off} in video-based action recognition. PrivHAR-Bench applies a graduated spectrum of visual privacy transformations: from lightweight spatial obfuscation to cryptographic block permutation, to a curated subset of 15 activity classes selected for human articulation diversity. Each of the 1,932 source videos is distributed across 9 parallel tiers of increasing privacy strength, with additional background-removed variants to isolate the contribution of human motion features from contextual scene bias. We provide lossless frame sequences, per-frame bounding boxes, estimated pose keypoints with joint-level confidence scores, standardized group-based train/test splits, and an evaluation toolkit computing recognition accuracy and privacy metrics. Empirical validation using R3D-18 demonstrates a measurable and interpretable degradation curve across tiers, with within-tier accuracy declining from 88.8\% (clear) to 53.5\% (encrypted, background-removed) and cross-domain accuracy collapsing to 4.8\%, establishing PrivHAR-Bench as a controlled benchmark for comparing privacy-preserving HAR methods under standardized conditions. The dataset, generation pipeline, and evaluation code are publicly available.

  </details>



- **A Benchmark of State-Space Models vs. Transformers and BiLSTM-based Models for Historical Newspaper OCR**  
  Merveilles Agbeti-messan, Thierry Paquet, Clément Chatelain, Pierrick Tranouez, Stéphane Nicolas  
  _2026-04-01_ · https://arxiv.org/abs/2604.00725v1  
  <details><summary>Abstract</summary>

  End-to-end OCR for historical newspapers remains challenging, as models must handle long text sequences, degraded print quality, and complex layouts. While Transformer-based recognizers dominate current research, their quadratic complexity limits efficient paragraph-level transcription and large-scale deployment. We investigate linear-time State-Space Models (SSMs), specifically Mamba, as a scalable alternative to Transformer-based sequence modeling for OCR. We present to our knowledge, the first OCR architecture based on SSMs, combining a CNN visual encoder with bi-directional and autoregressive Mamba sequence modeling, and conduct a large-scale benchmark comparing SSMs with Transformer- and BiLSTM-based recognizers. Multiple decoding strategies (CTC, autoregressive, and non-autoregressive) are evaluated under identical training conditions alongside strong neural baselines (VAN, DAN, DANIEL) and widely used off-the-shelf OCR engines (PERO-OCR, Tesseract OCR, TrOCR, Gemini). Experiments on historical newspapers from the Bibliothèque nationale du Luxembourg, with newly released >99% verified gold-standard annotations, and cross-dataset tests on Fraktur and Antiqua lines, show that all neural models achieve low error rates (~2% CER), making computational efficiency the main differentiator. Mamba-based models maintain competitive accuracy while halving inference time and exhibiting superior memory scaling (1.26x vs 2.30x growth at 1000 chars), reaching 6.07% CER at the severely degraded paragraph level compared to 5.24% for DAN, while remaining 2.05x faster. We release code, trained models, and standardized evaluation protocols to enable reproducible research and guide practitioners in large-scale cultural heritage OCR.

  </details>



- **TTA-Vid: Generalized Test-Time Adaptation for Video Reasoning**  
  Soumya Shamarao Jahagirdar, Edson Araujo, Anna Kukleva, M. Jehanzeb Mirza, Saurabhchand Bhati, Samuel Thomas, Brian Kingsbury, Rogerio Feris, James R. Glass, Hilde Kuehne  
  _2026-04-01_ · https://arxiv.org/abs/2604.00696v1  
  <details><summary>Abstract</summary>

  Recent video reasoning models have shown strong results on temporal and multimodal understanding, yet they depend on large-scale supervised data and multi-stage training pipelines, making them costly to train and difficult to adapt to new domains. In this work, we leverage the paradigm of Test-Time Reinforcement Learning on video-language data to allow for adapting a pretrained model to incoming video samples at test-time without explicit labels. The proposed test-time adaptation for video approach (TTA-Vid) combines two components that work simultaneously: (1) a test-time adaptation that performs step-by-step reasoning at inference time on multiple frame subsets. We then use a batch-aware frequency-based reward computed across different frame subsets as pseudo ground truth to update the model. It shows that the resulting model trained on a single batch or even a single sample from a dataset, is able to generalize at test-time to the whole dataset and even across datasets. Because the adaptation occurs entirely at test time, our method requires no ground-truth annotations or dedicated training splits. Additionally, we propose a multi-armed bandit strategy for adaptive frame selection that learns to prioritize informative frames, guided by the same reward formulation. Our evaluation shows that TTA-Vid yields consistent improvements across various video reasoning tasks and is able to outperform current state-of-the-art methods trained on large-scale data. This highlights the potential of test-time reinforcement learning for temporal multimodal understanding.

  </details>



- **MoonAnything: A Vision Benchmark with Large-Scale Lunar Supervised Data**  
  Clémentine Grethen, Yuang Shi, Simone Gasparini, Géraldine Morin  
  _2026-04-01_ · https://arxiv.org/abs/2604.00682v1  
  <details><summary>Abstract</summary>

  Accurate perception of lunar surfaces is critical for modern lunar exploration missions. However, developing robust learning-based perception systems is hindered by the lack of datasets that provide both geometric and photometric supervision. Existing lunar datasets typically lack either geometric ground truth, photometric realism, illumination diversity, or large-scale coverage. In this paper, we introduce MoonAnything, a unified benchmark built on real lunar topography with physically-based rendering, providing the first comprehensive geometric and photometric supervision under diverse illumination with large scale. The benchmark comprises two complementary sub-datasets : i) LunarGeo provides stereo images with corresponding dense depth maps and camera calibration enabling 3D reconstruction and pose estimation; ii) LunarPhoto provides photorealistic images using a spatially-varying BRDF model, along with multi-illumination renderings under real solar configurations, enabling reflectance estimation and illumination-robust perception. Together, these datasets offer over 130K samples with comprehensive supervision. Beyond lunar applications, MoonAnything offers a unique setting and challenging testbed for algorithms under low-textured, high-contrast conditions and applies to other airless celestial bodies and could generalize beyond. We establish baselines using state-of-the-art methods and release the complete dataset along with generation tools to support community extension: https://github.com/clementinegrethen/MoonAnything.

  </details>



- **CL-VISTA: Benchmarking Continual Learning in Video Large Language Models**  
  Haiyang Guo, Yichen Shi, Fei Zhu, Wenzhuo Liu, Hongbo Zhao, Fanhu Zeng, Shijie Ma, Da-Han Wang, Xu-Yao Zhang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00677v1  
  <details><summary>Abstract</summary>

  Video Large Language Models (Video-LLMs) require continual learning to adapt to non-stationary real-world data. However, existing benchmarks fall short of evaluating modern foundation models: many still rely on models without large-scale pre-training, and prevailing benchmarks typically partition a single dataset into sub-tasks, resulting in high task redundancy and negligible forgetting on pre-trained Video-LLMs. To address these limitations, we propose CL-VISTA, a benchmark tailored for continual video understanding of Video-LLMs. By curating 8 diverse tasks spanning perception, understanding, and reasoning, CL-VISTA induces substantial distribution shifts that effectively expose catastrophic forgetting. To systematically assess CL methods, we establish a comprehensive evaluation framework comprising 6 distinct protocols across 3 critical dimensions: performance, computational efficiency, and memory footprint. Notably, the performance dimension incorporates a general video understanding assessment to assess whether CL methods genuinely enhance foundational intelligence or merely induce task-specific overfitting. Extensive benchmarking of 10 mainstream CL methods reveals a fundamental trade-off: no single approach achieves universal superiority across all dimensions. Methods that successfully mitigate catastrophic forgetting tend to compromise generalization or incur prohibitive computational and memory overheads. We hope CL-VISTA provides critical insights for advancing continual learning in multimodal foundation models.

  </details>



- **Towards Viewpoint-Robust End-to-End Autonomous Driving with 3D Foundation Model Priors**  
  Hiroki Hashimoto, Hiromichi Goto, Hiroyuki Sugai, Hiroshi Kera, Kazuhiko Kawamoto  
  _2026-04-01_ · https://arxiv.org/abs/2604.00597v1  
  <details><summary>Abstract</summary>

  Robust trajectory planning under camera viewpoint changes is important for scalable end-to-end autonomous driving. However, existing models often depend heavily on the camera viewpoints seen during training. We investigate an augmentation-free approach that leverages geometric priors from a 3D foundation model. The method injects per-pixel 3D positions derived from depth estimates as positional embeddings and fuses intermediate geometric features through cross-attention. Experiments on the VR-Drive camera viewpoint perturbation benchmark show reduced performance degradation under most perturbation conditions, with clear improvements under pitch and height perturbations. Gains under longitudinal translation are smaller, suggesting that more viewpoint-agnostic integration is needed for robustness to camera viewpoint changes.

  </details>



- **HarassGuard: Detecting Harassment Behaviors in Social Virtual Reality with Vision-Language Models**  
  Junhee Lee, Minseok Kim, Hwanjo Heo, Seungwon Woo, Jinwoo Kim  
  _2026-04-01_ · https://arxiv.org/abs/2604.00592v1  
  <details><summary>Abstract</summary>

  Social Virtual Reality (VR) platforms provide immersive social experiences but also expose users to serious risks of online harassment. Existing safety measures are largely reactive, while proactive solutions that detect harassment behavior during an incident often depend on sensitive biometric data, raising privacy concerns. In this paper, we present HarassGuard, a vision-language model (VLM) based system that detects physical harassment in social VR using only visual input. We construct an IRB-approved harassment vision dataset, apply prompt engineering, and fine-tune VLMs to detect harassment behavior by considering contextual information in social VR. Experimental results demonstrate that HarassGuard achieves competitive performance compared to state-of-the-art baselines (i.e., LSTM/CNN, Transformer), reaching an accuracy of up to 88.09% in binary classification and 68.85% in multi-class classification. Notably, HarassGuard matches these baselines while using significantly fewer fine-tuning samples (200 vs. 1,115), offering unique advantages in contextual reasoning and privacy-preserving detection.

  </details>



- **FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning**  
  Tien-Yu Chi  
  _2026-04-01_ · https://arxiv.org/abs/2604.00559v1  
  <details><summary>Abstract</summary>

  Early detection of highly pathogenic avian influenza (HPAI) and endemic poultry diseases is critical for global food security. While computer vision models excel at classifying diseases from fecal imaging, deploying these systems at scale is bottlenecked by farm data privacy concerns and institutional data silos. Furthermore, existing open-source agricultural datasets frequently suffer from severe, undocumented data contamination. In this paper, we introduce $\textbf{FecalFed}$, a privacy-preserving federated learning framework for poultry disease classification. We first curate and release $\texttt{poultry-fecal-fl}$, a rigorously deduplicated dataset of 8,770 unique images across four disease classes, revealing and eliminating a 46.89$\%$ duplication rate in popular public repositories. To simulate realistic agricultural environments, we evaluate FecalFed under highly heterogeneous, non-IID conditions (Dirichlet $α=0.5$). While isolated single-farm training collapses under this data heterogeneity, yielding only 64.86$\%$ accuracy, our federated approach recovers performance without centralizing sensitive data. Specifically, utilizing server-side adaptive optimization (FedAdam) with a Swin-Small architecture achieves 90.31$\%$ accuracy, closely approaching the centralized upper bound of 95.10\%. Furthermore, we demonstrate that an edge-optimized Swin-Tiny model maintains highly competitive performance at 89.74$\%$, establishing a highly efficient, privacy-first blueprint for on-farm avian disease monitoring.

  </details>



- **STAR: Mitigating Cascading Errors in Spatial Reasoning via Turn-point Alignment and Segment-level DPO**  
  Pukun Zhao, Longxiang Wang, Chen Chen, Peicheng Wang, Fanqing Zhou, Runze Li, Haojian Huang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00558v1  
  <details><summary>Abstract</summary>

  Structured spatial navigation is a core benchmark for Large Language Models (LLMs) spatial reasoning. Existing paradigms like Visualization-of-Thought (VoT) are prone to cascading errors in complex topologies. To solve this, we propose STAR, a two-stage framework grounded on topological anchors, and introduce the RedMaze-23K dataset with human-inspired turnpoint annotations. The first stage uses supervised fine-tuning to help models internalize spatial semantics and prune redundant paths. The second adopts Spatial-aware Segment-level Direct Preference Optimization (SDPO) to refine self-correction in long-horizon navigation. Experiments show STAR achieves state-of-the-art performance among open-source models: its 32B variant outperforms DeepSeek-V3 (29.27% vs. 25.00%) and reaches 82.4% of GPT-4's performance.

  </details>



- **MATHENA: Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy**  
  Kyeonghun Kim, Jaehyung Park, Youngung Han, Anna Jung, Seongbin Park, Sumin Lee, Jiwon Yang, Jiyoon Han, Subeen Lee, Junsu Lim, et al.  
  _2026-04-01_ · https://arxiv.org/abs/2604.00537v1  
  <details><summary>Abstract</summary>

  Dental diagnosis from Orthopantomograms (OPGs) requires coordination of tooth detection, caries segmentation (CarSeg), anomaly detection (AD), and dental developmental staging (DDS). We propose Mamba-based Architectural Tooth Hierarchical Estimator and Holistic Evaluation Network for Anatomy (MATHENA), a unified framework leveraging Mamba's linear-complexity State Space Models (SSM) to address all four tasks. MATHENA integrates MATHE, a multi-resolution SSM-driven detector with four-directional Vision State Space (VSS) blocks for O(N) global context modeling, generating per-tooth crops. These crops are processed by HENA, a lightweight Mamba-UNet with a triple-head architecture and Global Context State Token (GCST). In the triple-head architecture, CarSeg is first trained as an upstream task to establish shared representations, which are then frozen and reused for downstream AD fine-tuning and DDS classification via linear probing, enabling stable, efficient learning. We also curate PARTHENON, a benchmark comprising 15,062 annotated instances from ten datasets. MATHENA achieves 93.78% mAP@50 in tooth detection, 90.11% Dice for CarSeg, 88.35% for AD, and 72.40% ACC for DDS.

  </details>



- **AceTone: Bridging Words and Colors for Conditional Image Grading**  
  Tianren Ma, Mingxiang Liao, Xijin Zhang, Qixiang Ye  
  _2026-04-01_ · https://arxiv.org/abs/2604.00530v1  
  <details><summary>Abstract</summary>

  Color affects how we interpret image style and emotion. Previous color grading methods rely on patch-wise recoloring or fixed filter banks, struggling to generalize across creative intents or align with human aesthetic preferences. In this study, we propose AceTone, the first approach that supports multimodal conditioned color grading within a unified framework. AceTone formulates grading as a generative color transformation task, where a model directly produces 3D-LUTs conditioned on text prompts or reference images. We develop a VQ-VAE based tokenizer which compresses a $3\times32^3$ LUT vector to 64 discrete tokens with $ΔE<2$ fidelity. We further build a large-scale dataset, AceTone-800K, and train a vision-language model to predict LUT tokens, followed by reinforcement learning to align outputs with perceptual fidelity and aesthetics. Experiments show that AceTone achieves state-of-the-art performance on both text-guided and reference-guided grading tasks, improving LPIPS by up to 50% over existing methods. Human evaluations confirm that AceTone's results are visually pleasing and stylistically coherent, demonstrating a new pathway toward language-driven, aesthetic-aligned color grading.

  </details>



- **Learnability-Guided Diffusion for Dataset Distillation**  
  Jeffrey A. Chan-Santiago, Mubarak Shah  
  _2026-04-01_ · https://arxiv.org/abs/2604.00519v1  
  <details><summary>Abstract</summary>

  Training machine learning models on massive datasets is expensive and time-consuming. Dataset distillation addresses this by creating a small synthetic dataset that achieves the same performance as the full dataset. Recent methods use diffusion models to generate distilled data, either by promoting diversity or matching training gradients. However, existing approaches produce redundant training signals, where samples convey overlapping information. Empirically, disjoint subsets of distilled datasets capture 80-90% overlapping signals. This redundancy stems from optimizing visual diversity or average training dynamics without accounting for similarity across samples, leading to datasets where multiple samples share similar information rather than complementary knowledge. We propose learnability-driven dataset distillation, which constructs synthetic datasets incrementally through successive stages. Starting from a small set, we train a model and generate new samples guided by learnability scores that identify what the current model can learn from, creating an adaptive curriculum. We introduce Learnability-Guided Diffusion (LGD), which balances training utility for the current model with validity under a reference model to generate curriculum-aligned samples. Our approach reduces redundancy by 39.1%, promotes specialization across training stages, and achieves state-of-the-art results on ImageNet-1K (60.1%), ImageNette (87.2%), and ImageWoof (72.9%). Our code is available on our project page https://jachansantiago.github.io/learnability-guided-distillation/.

  </details>



- **MOON3.0: Reasoning-aware Multimodal Representation Learning for E-commerce Product Understanding**  
  Junxian Wu, Chenghan Fu, Zhanheng Nie, Daoze Zhang, Bowen Wan, Wanxian Guan, Chuan Yu, Jian Xu, Bo Zheng  
  _2026-04-01_ · https://arxiv.org/abs/2604.00513v1  
  <details><summary>Abstract</summary>

  With the rapid growth of e-commerce, exploring general representations rather than task-specific ones has attracted increasing attention. Although recent multimodal large language models (MLLMs) have driven significant progress in product understanding, they are typically employed as feature extractors that implicitly encode product information into global embeddings, thereby limiting their ability to capture fine-grained attributes. Therefore, we argue that leveraging the reasoning capabilities of MLLMs to explicitly model fine-grained product attributes holds significant potential. Nevertheless, achieving this goal remains non-trivial due to several key challenges: (i) long-context reasoning tends to dilute the model's attention to salient information in the raw input; (ii) supervised fine-tuning (SFT) primarily encourages rigid imitation, limiting the exploration of effective reasoning strategies; and (iii) fine-grained details are progressively attenuated during forward propagation. To address these issues, we propose MOON3.0, the first reasoning-aware MLLM-based model for product representation learning. Our method (1) employs a multi-head modality fusion module to adaptively integrate raw signals; (2) incorporates a joint contrastive and reinforcement learning framework to autonomously explore more effective reasoning strategies; and (3) introduces a fine-grained residual enhancement module to progressively preserve local details throughout the network. Additionally, we release a large-scale multimodal e-commerce benchmark MBE3.0. Experimentally, our model demonstrates state-of-the-art zero-shot performance across various downstream tasks on both our benchmark and public datasets.

  </details>



- **Certificate-Driven Closed-Loop Multi-Agent Path Finding with Inheritable Factorization**  
  Jiarui Li, Runyu Zhang, Gioele Zardini  
  _2026-04-01_ · https://arxiv.org/abs/2604.00428v1  
  <details><summary>Abstract</summary>

  Multi-agent coordination in automated warehouses and logistics is commonly modeled as the Multi-Agent Path Finding (MAPF) problem. Closed-loop MAPF algorithms improve scalability by planning only the next movement and replanning online, but this finite-horizon viewpoint can be shortsighted and makes it difficult to preserve global guarantees and exploit compositional structure. This issue is especially visible in Anytime Closed-Loop Conflict-Based Search (ACCBS), which applies Conflict-Based Search (CBS) over dynamically extended finite horizons but, under finite computational budgets, may terminate with short active prefixes in dense instances. We introduce certificate trajectories and their associated fleet budget as a general mechanism for filtering closed-loop updates. A certificate provides a conflict-free fallback plan and a monotone upper bound on the remaining cost; accepting only certificate-improving updates yields completeness. The same budget information induces a budget-limited factorization that enables global, inheritable decomposition across timesteps. Instantiating the framework on ACCBS yields Certificate-Driven Conflict-Based Search (CDCBS). Experiments on benchmark maps show that CDCBS achieves more consistent solution quality than ACCBS, particularly in dense settings, while the proposed factorization reduces effective group size.

  </details>



- **Learning Humanoid Navigation from Human Data**  
  Weizhuo Wang, Yanjie Ze, C. Karen Liu, Monroe Kennedy  
  _2026-04-01_ · https://arxiv.org/abs/2604.00416v1  
  <details><summary>Abstract</summary>

  We present EgoNav, a system that enables a humanoid robot to traverse diverse, unseen environments by learning entirely from 5 hours of human walking data, with no robot data or finetuning. A diffusion model predicts distributions of plausible future trajectories conditioned on past trajectory, a 360 deg visual memory fusing color, depth, and semantics, and video features from a frozen DINOv3 backbone that capture appearance cues invisible to depth sensors. A hybrid sampling scheme achieves real-time inference in 10 denoising steps, and a receding-horizon controller selects paths from the predicted distribution. We validate EgoNav through offline evaluations, where it outperforms baselines in collision avoidance and multi-modal coverage, and through zero-shot deployment on a Unitree G1 humanoid across unseen indoor and outdoor environments. Behaviors such as waiting for doors to open, navigating around crowds, and avoiding glass walls emerge naturally from the learned prior. We will release the dataset and trained models. Our website: https://egonav.weizhuowang.com

  </details>



- **COTTA: Context-Aware Transfer Adaptation for Trajectory Prediction in Autonomous Driving**  
  Seohyoung Park, Jaeyeol Lim, Seoyoung Ju, Kyeonghun Kim, Nam-Joon Kim, Hyuk-Jae Lee  
  _2026-04-01_ · https://arxiv.org/abs/2604.00402v1  
  <details><summary>Abstract</summary>

  Developing robust models to accurately predict the trajectories of surrounding agents is fundamental to autonomous driving safety. However, most public datasets, such as the Waymo Open Motion Dataset and Argoverse, are collected in Western road environments and do not reflect the unique traffic patterns, infrastructure, and driving behaviors of other regions, including South Korea. This domain discrepancy leads to performance degradation when state-of-the-art models trained on Western data are deployed in different geographic contexts. In this work, we investigate the adaptability of Query-Centric Trajectory Prediction (QCNet) when transferred from U.S.-based data to Korean road environments. Using a Korean autonomous driving dataset, we compare four training strategies: zero-shot transfer, training from scratch, full fine-tuning, and encoder freezing. Experimental results demonstrate that leveraging pretrained knowledge significantly improves prediction performance. Specifically, selectively fine-tuning the decoder while freezing the encoder yields the best trade-off between accuracy and training efficiency, reducing prediction error by over 66% compared to training from scratch. This study provides practical insights into effective transfer learning strategies for deploying trajectory prediction models in new geographic domains.

  </details>



- **VLM-in-the-Loop: A Plug-In Quality Assurance Module for ECG Digitization Pipelines**  
  Jiachen Li, Shihao Li, Soovadeep Bakshi, Wei Li, Dongmei Chen  
  _2026-04-01_ · https://arxiv.org/abs/2604.00396v1  
  <details><summary>Abstract</summary>

  ECG digitization could unlock billions of archived clinical records, yet existing methods collapse on real-world images despite strong benchmark numbers. We introduce \textbf{VLM-in-the-Loop}, a plug-in quality assurance module that wraps any digitization backend with closed-loop VLM feedback via a standardized interface, requiring no modification to the underlying digitizer. The core mechanism is \textbf{tool grounding}: anchoring VLM assessment in quantitative evidence from domain-specific signal analysis tools. In a controlled ablation on 200 records with paired ground truth, tool grounding raises verdict consistency from 71\% to 89\% and doubles fidelity separation ($Δ$PCC 0.03 $\rightarrow$ 0.08), with the effect replicating across three VLMs (Claude Opus~4, GPT-4o, Gemini~2.5 Pro), confirming a pattern-level rather than model-specific gain. Deployed across four backends, the module improves every one: 29.4\% of borderline leads improved on our pipeline; 41.2\% of failed limb leads recovered on ECG-Digitiser; valid leads per image doubled on Open-ECG-Digitizer (2.5 $\rightarrow$ 5.8). On 428 real clinical HCM images, the integrated system reaches 98.0\% Excellent quality. Both the plug-in architecture and tool-grounding mechanism are domain-parametric, suggesting broader applicability wherever quality criteria are objectively measurable.

  </details>



- **Neural Reconstruction of LiDAR Point Clouds under Jamming Attacks via Full-Waveform Representation and Simultaneous Laser Sensing**  
  Ryo Yoshida, Takami Sato, Wenlun Zhang, Yuki Hayakawa, Shota Nagai, Takahiro Kado, Taro Beppu, Ibuki Fujioka, Yunshan Zhong, Kentaro Yoshioka  
  _2026-04-01_ · https://arxiv.org/abs/2604.00371v1  
  <details><summary>Abstract</summary>

  LiDAR sensors are critical for autonomous driving perception, yet remain vulnerable to spoofing attacks. Jamming attacks inject high-frequency laser pulses that completely blind LiDAR sensors by overwhelming authentic returns with malicious signals. We discover that while point clouds become randomized, the underlying full-waveform data retains distinguishable signatures between attack and legitimate signals. In this work, we propose PULSAR-Net, capable of reconstructing authentic point clouds under jamming attacks by leveraging previously underutilized intermediate full-waveform representations and simultaneous laser sensing in modern LiDAR systems. PULSAR-Net adopts a novel U-Net architecture with axial spatial attention mechanisms specifically designed to identify attack-induced signals from authentic object returns in the full-waveform representation. To address the lack of full-waveform representations in existing LiDAR datasets under jamming attacks, we introduce a physics-aware dataset generation pipeline that synthesizes realistic full-waveform representations under jamming attacks. Despite being trained exclusively on synthetic data, PULSAR-Net achieves reconstruction rates of 92% and 73% for vehicles obscured by jamming attacks in real-world static and driving scenarios, respectively.

  </details>



- **VADMamba++: Efficient Video Anomaly Detection via Hybrid Modeling in Grayscale Space**  
  Jihao Lyu, Minghua Zhao, Jing Hu, Yifei Chen, Shuangli Du, Cheng Shi  
  _2026-04-01_ · https://arxiv.org/abs/2604.00360v1  
  <details><summary>Abstract</summary>

  VADMamba pioneered the introduction of Mamba to Video Anomaly Detection (VAD), achieving high accuracy and fast inference through hybrid proxy tasks. Nevertheless, its heavy reliance on optical flow as auxiliary input and inter-task fusion scoring constrains its applicability to a single proxy task. In this paper, we introduce VADMamba++, an efficient VAD method based on the Gray-to-RGB paradigm that enforces a Single-Channel to Three-Channel reconstruction mapping, designed for a single proxy task and operating without auxiliary inputs. This paradigm compels inferring color appearances from grayscale structures, allowing anomalies to be more effectively revealed through dual inconsistencies between structure and chromatic cues. Specifically, VADMamba++ reconstructs grayscale frames into the RGB space to simultaneously discriminate structural geometry and chromatic fidelity, thereby enhancing sensitivity to explicit visual anomalies. We further design a hybrid modeling backbone that integrates Mamba, CNN, and Transformer modules to capture diverse normal patterns while suppressing the appearance of anomalies. Furthermore, an intra-task fusion scoring strategy integrates explicit future-frame prediction errors with implicit quantized feature errors, further improving accuracy under a single task setting. Extensive experiments on three benchmark datasets demonstrate that VADMamba++ outperforms state-of-the-art methods while meeting performance and efficiency, especially under a strict single-task setting with only frame-level inputs.

  </details>



- **Label-efficient underwater species classification with semi-supervised learning on frozen foundation model embeddings**  
  Thomas Manuel Rost  
  _2026-03-31_ · https://arxiv.org/abs/2604.00313v1  
  <details><summary>Abstract</summary>

  Automated species classification from underwater imagery is bottlenecked by the cost of expert annotation, and supervised models trained on one dataset rarely transfer to new conditions. We investigate whether semi-supervised methods operating on frozen foundation model embeddings can close this annotation gap with minimal labeling effort. Using DINOv3 ViT-B embeddings with no fine-tuning, we propagate a small set of labeled seeds through unlabeled data via nearest-neighbor-based self-training and evaluate on the AQUA20 benchmark (20 marine species). With fewer than 5% of the training labels, self-training on frozen embeddings closes much of the gap to a fully supervised ConvNeXt baseline trained on the entire labeled dataset; at full supervision, the gap narrows to a few percentage points, with several species exceeding the supervised baseline. Class separability in the embedding space, measured by ROC-AUC, is high even at extreme label scarcity, indicating that the frozen representations capture discriminative structure well before decision boundaries can be reliably estimated. Our approach requires no training, no domain-specific data engineering, and no underwater-adapted models, establishing a practical, immediately deployable baseline for label-efficient marine species recognition. All results are reported on the held-out test set over 100 random seed initializations.

  </details>



- **OmniSch: A Multimodal PCB Schematic Benchmark For Structured Diagram Visual Reasoning**  
  Taiting Lu, Kaiyuan Lin, Yuxin Tian, Yubo Wang, Muchuan Wang, Sharique Khatri, Akshit Kartik, Yixi Wang, Amey Santosh Rane, Yida Wang, et al.  
  _2026-03-31_ · https://arxiv.org/abs/2604.00270v1  
  <details><summary>Abstract</summary>

  Recent large multimodal models (LMMs) have made rapid progress in visual grounding, document understanding, and diagram reasoning tasks. However, their ability to convert Printed Circuit Board (PCB) schematic diagrams into machine-readable spatially weighted netlist graphs, jointly capturing component attributes, connectivity, and geometry, remains largely underexplored, despite such graph representations are the backbone of practical electronic design automation (EDA) workflows. To bridge this gap, we introduce OmniSch, the first comprehensive benchmark designed to assess LMMs on schematic understanding and spatial netlist graph construction. OmniSch contains 1,854 real-world schematic diagrams and includes four tasks: (1) visual grounding for schematic entities, with 109.9K grounded instances aligning 423.4K diagram semantic labels to their visual regions; (2) diagram-to-graph reasoning, understanding topological relationship among diagram elements; (3) geometric reasoning, constructing layout-dependent weights for each connection; and (4) tool-augmented agentic reasoning for visual search, invoking external tools to accomplish (1)-(3). Our results reveal substantial gaps of current LMMs in interpreting schematic engineering artifacts, including unreliable fine-grained grounding, brittle layout-to-graph parsing, inconsistent global connectivity reasoning and inefficient visual exploration.

  </details>



- **Benchmarking Interaction, Beyond Policy: a Reproducible Benchmark for Collaborative Instance Object Navigation**  
  Edoardo Zorzi, Francesco Taioli, Yiming Wang, Marco Cristani, Alessandro Farinelli, Alberto Castellini, Loris Bazzani  
  _2026-03-31_ · https://arxiv.org/abs/2604.00265v1  
  <details><summary>Abstract</summary>

  We propose Question-Asking Navigation (QAsk-Nav), the first reproducible benchmark for Collaborative Instance Object Navigation (CoIN) that enables an explicit, separate assessment of embodied navigation and collaborative question asking. CoIN tasks an embodied agent with reaching a target specified in free-form natural language under partial observability, using only egocentric visual observations and interactive natural-language dialogue with a human, where the dialogue can help to resolve ambiguity among visually similar object instances. Existing CoIN benchmarks are primarily focused on navigation success and offer no support for consistent evaluation of collaborative interaction. To address this limitation, QAsk-Nav provides (i) a lightweight question-asking protocol scored independently of navigation, (ii) an enhanced navigation protocol with realistic, diverse, high-quality target descriptions, and (iii) an open-source dataset, that includes 28,000 quality-checked reasoning and question-asking traces for training and analysis of interactive capabilities of CoIN models. Using the proposed QAsk-Nav benchmark, we develop Light-CoNav, a lightweight unified model for collaborative navigation that is 3x smaller and 70x faster than existing modular methods, while outperforming state-of-the-art CoIN approaches in generalization to unseen objects and environments. Project page at https://benchmarking-interaction.github.io/

  </details>



- **Neural-Assisted in-Motion Self-Heading Alignment**  
  Zeev Yampolsky, Felipe O. Silva, Adriano Frutuoso, Itzik Klein  
  _2026-03-31_ · https://arxiv.org/abs/2604.00168v1  
  <details><summary>Abstract</summary>

  Autonomous platforms operating in the oceans require accurate navigation to successfully complete their mission. In this regard, the initial heading estimation accuracy and the time required to achieve it play a critical role. The initial heading is traditionally estimated by model-based approaches employing orientation decomposition. However, methods such as the dual vector decomposition and optimized attitude decomposition achieve satisfactory heading accuracy only after long alignment times. To allow rapid and accurate initial heading estimation, we propose an end-to-end, model-free, neural-assisted framework using the same inputs as the model-based approaches. Our proposed approach was trained and evaluated on real-world dataset captured by an autonomous surface vehicle. Our approach shows a significant accuracy improvement over the model-based approaches achieving an average absolute error improvement of 53%. Additionally, our proposed approach was able to reduce the alignment time by up to 67%. Thus, by employing our proposed approach, the reduction in alignment time and improved accuracy allow for a shorter deployment time of an autonomous platform and increased navigation accuracy during the mission.

  </details>


