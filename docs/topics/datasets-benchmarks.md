# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-12 07:42 UTC_

Total papers shown: **43**


---

- **GaussiAnimate: Reconstruct and Rig Animatable Categories with Level of Dynamics**  
  Jiaxin Wang, Dongxin Lyu, Zeyu Cai, Zhiyang Dou, Cheng Lin, Anpei Chen, Yuliang Xiu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08547v1  
  <details><summary>Abstract</summary>

  Free-form bones, that conform closely to the surface, can effectively capture non-rigid deformations, but lack a kinematic structure necessary for intuitive control. Thus, we propose a Scaffold-Skin Rigging System, termed "Skelebones", with three key steps: (1) Bones: compress temporally-consistent deformable Gaussians into free-form bones, approximating non-rigid surface deformations; (2) Skeleton: extract a Mean Curvature Skeleton from canonical Gaussians and refine it temporally, ensuring a category-agnostic, motion-adaptive, and topology-correct kinematic structure; (3) Binding: bind the skeleton and bones via non-parametric partwise motion matching (PartMM), synthesizing novel bone motions by matching, retrieving, and blending existing ones. Collectively, these three steps enable us to compress the Level of Dynamics of 4D shapes into compact skelebones that are both controllable and expressive. We validate our approach on both synthetic and real-world datasets, achieving significant improvements in reanimation performance across unseen poses-with 17.3% PSNR gains over Linear Blend Skinning (LBS) and 21.7% over Bag-of-Bones (BoB)-while maintaining excellent reconstruction fidelity, particularly for characters exhibiting complex non-rigid surface dynamics. Our Partwise Motion Matching algorithm demonstrates strong generalization to both Gaussian and mesh representations, especially under low-data regime (~1000 frames), achieving 48.4% RMSE improvement over robust LBS and outperforming GRU- and MLP-based learning methods by >20%. Code will be made publicly available for research purposes at cookmaker.cn/gaussianimate.

  </details>



- **AVGen-Bench: A Task-Driven Benchmark for Multi-Granular Evaluation of Text-to-Audio-Video Generation**  
  Ziwei Zhou, Zeyuan Lai, Rui Wang, Yifan Yang, Zhen Xing, Yuqing Yang, Qi Dai, Lili Qiu, Chong Luo  
  _2026-04-09_ · https://arxiv.org/abs/2604.08540v1  
  <details><summary>Abstract</summary>

  Text-to-Audio-Video (T2AV) generation is rapidly becoming a core interface for media creation, yet its evaluation remains fragmented. Existing benchmarks largely assess audio and video in isolation or rely on coarse embedding similarity, failing to capture the fine-grained joint correctness required by realistic prompts. We introduce AVGen-Bench, a task-driven benchmark for T2AV generation featuring high-quality prompts across 11 real-world categories. To support comprehensive assessment, we propose a multi-granular evaluation framework that combines lightweight specialist models with Multimodal Large Language Models (MLLMs), enabling evaluation from perceptual quality to fine-grained semantic controllability. Our evaluation reveals a pronounced gap between strong audio-visual aesthetics and weak semantic reliability, including persistent failures in text rendering, speech coherence, physical reasoning, and a universal breakdown in musical pitch control. Code and benchmark resources are available at http://aka.ms/avgenbench.

  </details>



- **ParseBench: A Document Parsing Benchmark for AI Agents**  
  Boyang Zhang, Sebastián G. Acosta, Preston Carlson, Sacha Bron, Pierre-Loïc Doulcet, Simon Suo  
  _2026-04-09_ · https://arxiv.org/abs/2604.08538v1  
  <details><summary>Abstract</summary>

  AI agents are changing the requirements for document parsing. What matters is \emph{semantic correctness}: parsed output must preserve the structure and meaning needed for autonomous decisions, including correct table structure, precise chart data, semantically meaningful formatting, and visual grounding. Existing benchmarks do not fully capture this setting for enterprise automation, relying on narrow document distributions and text-similarity metrics that miss agent-critical failures. We introduce \textbf{ParseBench}, a benchmark of ${\sim}2{,}000$ human-verified pages from enterprise documents spanning insurance, finance, and government, organized around five capability dimensions: tables, charts, content faithfulness, semantic formatting, and visual grounding. Across 14 methods spanning vision-language models, specialized document parsers, and LlamaParse, the benchmark reveals a fragmented capability landscape: no method is consistently strong across all five dimensions. LlamaParse Agentic achieves the highest overall score at \agenticoverall\%, and the benchmark highlights the remaining capability gaps across current systems. Dataset and evaluation code are available on \href{https://huggingface.co/datasets/llamaindex/ParseBench}{HuggingFace} and \href{https://github.com/run-llama/ParseBench}{GitHub}.

  </details>



- **Fail2Drive: Benchmarking Closed-Loop Driving Generalization**  
  Simon Gerstenecker, Andreas Geiger, Katrin Renz  
  _2026-04-09_ · https://arxiv.org/abs/2604.08535v1  
  <details><summary>Abstract</summary>

  Generalization under distribution shift remains a central bottleneck for closed-loop autonomous driving. Although simulators like CARLA enable safe and scalable testing, existing benchmarks rarely measure true generalization: they typically reuse training scenarios at test time. Success can therefore reflect memorization rather than robust driving behavior. We introduce Fail2Drive, the first paired-route benchmark for closed-loop generalization in CARLA, with 200 routes and 17 new scenario classes spanning appearance, layout, behavioral, and robustness shifts. Each shifted route is matched with an in-distribution counterpart, isolating the effect of the shift and turning qualitative failures into quantitative diagnostics. Evaluating multiple state-of-the-art models reveals consistent degradation, with an average success-rate drop of 22.8\%. Our analysis uncovers unexpected failure modes, such as ignoring objects clearly visible in the LiDAR and failing to learn the fundamental concepts of free and occupied space. To accelerate follow-up work, Fail2Drive includes an open-source toolbox for creating new scenarios and validating solvability via a privileged expert policy. Together, these components establish a reproducible foundation for benchmarking and improving closed-loop driving generalization. We open-source all code, data, and tools at https://github.com/autonomousvision/fail2drive .

  </details>



- **FIT: A Large-Scale Dataset for Fit-Aware Virtual Try-On**  
  Johanna Karras, Yuanhao Wang, Yingwei Li, Ira Kemelmacher-Shlizerman  
  _2026-04-09_ · https://arxiv.org/abs/2604.08526v1  
  <details><summary>Abstract</summary>

  Given a person and a garment image, virtual try-on (VTO) aims to synthesize a realistic image of the person wearing the garment, while preserving their original pose and identity. Although recent VTO methods excel at visualizing garment appearance, they largely overlook a crucial aspect of the try-on experience: the accuracy of garment fit -- for example, depicting how an extra-large shirt looks on an extra-small person. A key obstacle is the absence of datasets that provide precise garment and body size information, particularly for "ill-fit" cases, where garments are significantly too large or too small. Consequently, current VTO methods default to generating well-fitted results regardless of the garment or person size. In this paper, we take the first steps towards solving this open problem. We introduce FIT (Fit-Inclusive Try-on), a large-scale VTO dataset comprising over 1.13M try-on image triplets accompanied by precise body and garment measurements. We overcome the challenges of data collection via a scalable synthetic strategy: (1) We programmatically generate 3D garments using GarmentCode and drape them via physics simulation to capture realistic garment fit. (2) We employ a novel re-texturing framework to transform synthetic renderings into photorealistic images while strictly preserving geometry. (3) We introduce person identity preservation into our re-texturing model to generate paired person images (same person, different garments) for supervised training. Finally, we leverage our FIT dataset to train a baseline fit-aware virtual try-on model. Our data and results set the new state-of-the-art for fit-aware virtual try-on, as well as offer a robust benchmark for future research. We will make all data and code publicly available on our project page: https://johannakarras.github.io/FIT.

  </details>



- **UniversalVTG: A Universal and Lightweight Foundation Model for Video Temporal Grounding**  
  Joungbin An, Agrim Jain, Kristen Grauman  
  _2026-04-09_ · https://arxiv.org/abs/2604.08522v1  
  <details><summary>Abstract</summary>

  Video temporal grounding (VTG) is typically tackled with dataset-specific models that transfer poorly across domains and query styles. Recent efforts to overcome this limitation have adapted large multimodal language models (MLLMs) to VTG, but their high compute cost and limited video context still hinder long-video grounding. We instead scale unified supervision while keeping the model lightweight. We present UniversalVTG, a single VTG model trained with large-scale cross-dataset pretraining. An offline Query Unifier canonicalizes heterogeneous query formats into a shared declarative space, reducing linguistic mismatch and preventing the negative transfer observed under naïve joint training. Combined with an efficient grounding head, UniversalVTG scales to long, untrimmed videos. Across diverse benchmarks-GoalStep-StepGrounding, Ego4D-NLQ, TACoS, Charades-STA, and ActivityNet-Captions-one UniversalVTG checkpoint achieves state-of-the-art performance versus dedicated VTG models. Moreover, despite being $>100\times$ smaller than recent MLLM-based approaches, UniversalVTG matches or exceeds their accuracy on multiple benchmarks, offering a practical alternative to parameter-heavy MLLMs.

  </details>



- **Visually-grounded Humanoid Agents**  
  Hang Ye, Xiaoxuan Ma, Fan Lu, Wayne Wu, Kwan-Yee Lin, Yizhou Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08509v1  
  <details><summary>Abstract</summary>

  Digital human generation has been studied for decades and supports a wide range of real-world applications. However, most existing systems are passively animated, relying on privileged state or scripted control, which limits scalability to novel environments. We instead ask: how can digital humans actively behave using only visual observations and specified goals in novel scenes? Achieving this would enable populating any 3D environments with digital humans at scale that exhibit spontaneous, natural, goal-directed behaviors. To this end, we introduce Visually-grounded Humanoid Agents, a coupled two-layer (world-agent) paradigm that replicates humans at multiple levels: they look, perceive, reason, and behave like real people in real-world 3D scenes. The World Layer reconstructs semantically rich 3D Gaussian scenes from real-world videos via an occlusion-aware pipeline and accommodates animatable Gaussian-based human avatars. The Agent Layer transforms these avatars into autonomous humanoid agents, equipping them with first-person RGB-D perception and enabling them to perform accurate, embodied planning with spatial awareness and iterative reasoning, which is then executed at the low level as full-body actions to drive their behaviors in the scene. We further introduce a benchmark to evaluate humanoid-scene interaction in diverse reconstructed environments. Experiments show our agents achieve robust autonomous behavior, yielding higher task success rates and fewer collisions than ablations and state-of-the-art planning methods. This work enables active digital human population and advances human-centric embodied AI. Data, code, and models will be open-sourced.

  </details>



- **Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics**  
  Ying Shen, Jerry Xiong, Tianjiao Yu, Ismini Lourentzou  
  _2026-04-09_ · https://arxiv.org/abs/2604.08503v1  
  <details><summary>Abstract</summary>

  Recent advances in generative video modeling, driven by large-scale datasets and powerful architectures, have yielded remarkable visual realism. However, emerging evidence suggests that simply scaling data and model size does not endow these systems with an understanding of the underlying physical laws that govern real-world dynamics. Existing approaches often fail to capture or enforce such physical consistency, resulting in unrealistic motion and dynamics. In his work, we investigate whether integrating the inference of latent physical properties directly into the video generation process can equip models with the ability to produce physically plausible videos. To this end, we propose Phantom, a Physics-Infused Video Generation model that jointly models the visual content and latent physical dynamics. Conditioned on observed video frames and inferred physical states, Phantom jointly predicts latent physical dynamics and generates future video frames. Phantom leverages a physics-aware video representation that serves as an abstract yet informaive embedding of the underlying physics, facilitating the joint prediction of physical dynamics alongside video content without requiring an explicit specification of a complex set of physical dynamics and properties. By integrating the inference of physical-aware video representation directly into the video generation process, Phantom produces video sequences that are both visually realistic and physically consistent. Quantitative and qualitative results on both standard video generation and physics-aware benchmarks demonstrate that Phantom not only outperforms existing methods in terms of adherence to physical dynamics but also delivers competitive perceptual fidelity.

  </details>



- **Quantifying Explanation Consistency: The C-Score Metric for CAM-Based Explainability in Medical Image Classification**  
  Kabilan Elangovan, Daniel Ting  
  _2026-04-09_ · https://arxiv.org/abs/2604.08502v1  
  <details><summary>Abstract</summary>

  Class Activation Mapping (CAM) methods are widely used to generate visual explanations for deep learning classifiers in medical imaging. However, existing evaluation frameworks assess whether explanations are correct, measured by localisation fidelity against radiologist annotations, rather than whether they are consistent: whether the model applies the same spatial reasoning strategy across different patients with the same pathology. We propose the C-Score (Consistency Score), a confidence-weighted, annotation-free metric that quantifies intra-class explanation reproducibility via intensity-emphasised pairwise soft IoU across correctly classified instances. We evaluate six CAM techniques: GradCAM, GradCAM++, LayerCAM, EigenCAM, ScoreCAM, and MS GradCAM++ across three CNN architectures (DenseNet201, InceptionV3, ResNet50V2) over thirty training epochs on the Kermany chest X-ray dataset, covering transfer learning and fine-tuning phases. We identify three distinct mechanisms of AUC-consistency dissociation, invisible to standard classification metrics: threshold-mediated gold list collapse, technique-specific attribution collapse at peak AUC, and class-level consistency masking in global aggregation. C-Score provides an early warning signal of impending model instability. ScoreCAM deterioration on ResNet50V2 is detectable one full checkpoint before catastrophic AUC collapse and yields architecture-specific clinical deployment recommendations grounded in explanation quality rather than predictive ranking alone.

  </details>



- **CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning**  
  Rui Gan, Junyi Ma, Pei Li, Xingyou Yang, Kai Chen, Sikai Chen, Bin Ran  
  _2026-04-09_ · https://arxiv.org/abs/2604.08457v1  
  <details><summary>Abstract</summary>

  Cooperative autonomous driving requires traffic scene understanding from both vehicle and infrastructure perspectives. While vision-language models (VLMs) show strong general reasoning capabilities, their performance in safety-critical traffic scenarios remains insufficiently evaluated due to the ego-vehicle focus of existing benchmarks. To bridge this gap, we present \textbf{CrashSight}, a large-scale vision-language benchmark for roadway crash understanding using real-world roadside camera data. The dataset comprises 250 crash videos, annotated with 13K multiple-choice question-answer pairs organized under a two-tier taxonomy. Tier 1 evaluates the visual grounding of scene context and involved parties, while Tier 2 probes higher-level reasoning, including crash mechanics, causal attribution, temporal progression, and post-crash outcomes. We benchmark 8 state-of-the-art VLMs and show that, despite strong scene description capabilities, current models struggle with temporal and causal reasoning in safety-critical scenarios. We provide a detailed analysis of failure scenarios and discuss directions for improving VLM crash understanding. The benchmark provides a standardized evaluation framework for infrastructure-assisted perception in cooperative autonomous driving. The CrashSight benchmark, including the full dataset and code, is accessible at https://mcgrche.github.io/crashsight.

  </details>



- **A Soft Robotic Interface for Chick-Robot Affective Interactions**  
  Jue Chen, Alexander Mielke, Kaspar Althoefer, Elisabetta Versace  
  _2026-04-09_ · https://arxiv.org/abs/2604.08443v1  
  <details><summary>Abstract</summary>

  The potential of Animal-Robot Interaction (ARI) in welfare applications depends on how much an animal perceives a robotic agent as socially relevant, non-threatening and potentially attractive (acceptance). Here, we present an animal-centered soft robotic affective interface for newly hatched chicks (Gallus gallus). The soft interface provides safe and controllable cues, including warmth, breathing-like rhythmic deformation, and face-like visual stimuli. We evaluated chick acceptance of the interface and chick-robot interactions by measuring spontaneous approach and touch responses during video tracking. Overall, chicks approached and spent increasing time on or near the interface, demonstrating acceptance of the device. Across different layouts, chicks showed strong preference for warm thermal stimulation, which increased over time. Face-like visual cues elicited a swift and stable preference, speeding up the initial approach to the tactile interface. Although the breathing cue did not elicit any preference, neither did it trigger avoidance, paving the way for further exploration. These findings translate affective interface concepts to ARI, demonstrating that appropriate soft, thermal and visual stimuli can sustain early chick-robot interactions. This work establishes a reliable evaluation protocol and a safe baseline for designing multimodal robotic devices for animal welfare and neuroscientific research.

  </details>



- **Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems**  
  Tolga Dimlioglu, Nadine Chang, Maying Shen, Rafid Mahmood, Jose M. Alvarez  
  _2026-04-09_ · https://arxiv.org/abs/2604.08366v1  
  <details><summary>Abstract</summary>

  Large-scale deep learning models for physical AI applications depend on diverse training data collection efforts. These models and correspondingly, the training data, must address different evaluation criteria necessary for the models to be deployable in real-world environments. Data selection policies can guide the development of the training set, but current frameworks do not account for the ambiguity in how data points affect different metrics. In this work, we propose Mixture Optimization via Scaling-Aware Iterative Collection (MOSAIC), a general data selection framework that operates by: (i) partitioning the dataset into domains; (ii) fitting neural scaling laws from each data domain to the evaluation metrics; and (iii) optimizing a data mixture by iteratively adding data from domains that maximize the change in metrics. We apply MOSAIC to autonomous driving (AD), where an End-to-End (E2E) planner model is evaluated on the Extended Predictive Driver Model Score (EPDMS), an aggregate of driving rule compliance metrics. Here, MOSAIC outperforms a diverse set of baselines on EPDMS with up to 80\% less data.

  </details>



- **MegaStyle: Constructing Diverse and Scalable Style Dataset via Consistent Text-to-Image Style Mapping**  
  Junyao Gao, Sibo Liu, Jiaxing Li, Yanan Sun, Yuanpeng Tu, Fei Shen, Weidong Zhang, Cairong Zhao, Jun Zhang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08364v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce MegaStyle, a novel and scalable data curation pipeline that constructs an intra-style consistent, inter-style diverse and high-quality style dataset. We achieve this by leveraging the consistent text-to-image style mapping capability of current large generative models, which can generate images in the same style from a given style description. Building on this foundation, we curate a diverse and balanced prompt gallery with 170K style prompts and 400K content prompts, and generate a large-scale style dataset MegaStyle-1.4M via content-style prompt combinations. With MegaStyle-1.4M, we propose style-supervised contrastive learning to fine-tune a style encoder MegaStyle-Encoder for extracting expressive, style-specific representations, and we also train a FLUX-based style transfer model MegaStyle-FLUX. Extensive experiments demonstrate the importance of maintaining intra-style consistency, inter-style diversity and high-quality for style dataset, as well as the effectiveness of the proposed MegaStyle-1.4M. Moreover, when trained on MegaStyle-1.4M, MegaStyle-Encoder and MegaStyle-FLUX provide reliable style similarity measurement and generalizable style transfer, making a significant contribution to the style transfer community. More results are available at our project website https://jeoyal.github.io/MegaStyle/.

  </details>



- **PokeGym: A Visually-Driven Long-Horizon Benchmark for Vision-Language Models**  
  Ruizhi Zhang, Ye Huang, Yuangang Pan, Chuanfu Shen, Zhilin Liu, Ting Xie, Wen Li, Lixin Duan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08340v1  
  <details><summary>Abstract</summary>

  While Vision-Language Models (VLMs) have achieved remarkable progress in static visual understanding, their deployment in complex 3D embodied environments remains severely limited. Existing benchmarks suffer from four critical deficiencies: (1) passive perception tasks circumvent interactive dynamics; (2) simplified 2D environments fail to assess depth perception; (3) privileged state leakage bypasses genuine visual processing; and (4) human evaluation is prohibitively expensive and unscalable. We introduce PokeGym, a visually-driven long-horizon benchmark instantiated within Pokemon Legends: Z-A, a visually complex 3D open-world Role-Playing Game. PokeGym enforces strict code-level isolation: agents operate solely on raw RGB observations while an independent evaluator verifies success via memory scanning, ensuring pure vision-based decision-making and automated, scalable assessment. The benchmark comprises 30 tasks (30-220 steps) spanning navigation, interaction, and mixed scenarios, with three instruction granularities (Visual-Guided, Step-Guided, Goal-Only) to systematically deconstruct visual grounding, semantic reasoning, and autonomous exploration capabilities. Our evaluation reveals a key limitation of current VLMs: physical deadlock recovery, rather than high-level planning, constitutes the primary bottleneck, with deadlocks showing a strong negative correlation with task success. Furthermore, we uncover a metacognitive divergence: weaker models predominantly suffer from Unaware Deadlocks (oblivious to entrapment), whereas advanced models exhibit Aware Deadlocks (recognizing entrapment yet failing to recover). These findings highlight the need to integrate explicit spatial intuition into VLM architectures. The code and benchmark will be available on GitHub.

  </details>



- **InstAP: Instance-Aware Vision-Language Pre-Train for Spatial-Temporal Understanding**  
  Ashutosh Kumar, Rajat Saini, Jingjing Pan, Mustafa Erdogan, Mingfang Zhang, Betty Le Dem, Norimasa Kobori, Quan Kong  
  _2026-04-09_ · https://arxiv.org/abs/2604.08337v1  
  <details><summary>Abstract</summary>

  Current vision-language pre-training (VLP) paradigms excel at global scene understanding but struggle with instance-level reasoning due to global-only supervision. We introduce InstAP, an Instance-Aware Pre-training framework that jointly optimizes global vision-text alignment and fine-grained, instance-level contrastive alignment by grounding textual mentions to specific spatial-temporal regions. To support this, we present InstVL, a large-scale dataset (2 million images, 50,000 videos) with dual-granularity annotations: holistic scene captions and dense, grounded instance descriptions. On the InstVL benchmark, InstAP substantially outperforms existing VLP models on instance-level retrieval, and also surpasses a strong VLP baseline trained on the exact same data corpus, isolating the benefit of our instance-aware objective. Moreover, instance-centric pre-training improves global understanding: InstAP achieves competitive zero-shot performance on multiple video benchmarks, including MSR-VTT and DiDeMo. Qualitative visualizations further show that InstAP localizes textual mentions to the correct instances, while global-only models exhibit more diffuse, scene-level attention.

  </details>



- **HistDiT: A Structure-Aware Latent Conditional Diffusion Model for High-Fidelity Virtual Staining in Histopathology**  
  Aasim Bin Saleem, Amr Ahmed, Ardhendu Behera, Hafeezullah Amin, Iman Yi Liao, Mahmoud Khattab, Pan Jia Wern, Haslina Makmur  
  _2026-04-09_ · https://arxiv.org/abs/2604.08305v1  
  <details><summary>Abstract</summary>

  Immunohistochemistry (IHC) is essential for assessing specific immune biomarkers like Human Epidermal growth-factor Receptor 2 (HER2) in breast cancer. However, the traditional protocols of obtaining IHC stains are resource-intensive, time-consuming, and prone to structural damages. Virtual staining has emerged as a scalable alternative, but it faces significant challenges in preserving fine-grained cellular structures while accurately translating biochemical expressions. Current state-of-the-art methods still rely on Generative Adversarial Networks (GANs) or standard convolutional U-Net diffusion models that often struggle with "structure and staining trade-offs". The generated samples are either structurally relevant but blurry, or texturally realistic but have artifacts that compromise their diagnostic use. In this paper, we introduce HistDiT, a novel latent conditional Diffusion Transformer (DiT) architecture that establishes a new benchmark for visual fidelity in virtual histological staining. The novelty introduced in this work is, a) the Dual-Stream Conditioning strategy that explicitly maintains a balance between spatial constraints via VAE-encoded latents and semantic phenotype guidance via UNI embeddings; b) the multi-objective loss function that contributes to sharper images with clear morphological structure; and c) the use of the Structural Correlation Metric (SCM) to focus on the core morphological structure for precise assessment of sample quality. Consequently, our model outperforms existing baselines, as demonstrated through rigorous quantitative and qualitative evaluations.

  </details>



- **CAMotion: A High-Quality Benchmark for Camouflaged Moving Object Detection in the Wild**  
  Siyuan Yao, Hao Sun, Ruiqi Yu, Xiwei Jiang, Wenqi Ren, Xiaochun Cao  
  _2026-04-09_ · https://arxiv.org/abs/2604.08287v1  
  <details><summary>Abstract</summary>

  Discovering camouflaged objects is a challenging task in computer vision due to the high similarity between camouflaged objects and their surroundings. While the problem of camouflaged object detection over sequential video frames has received increasing attention, the scale and diversity of existing video camouflaged object detection (VCOD) datasets are greatly limited, which hinders the deeper analysis and broader evaluation of recent deep learning-based algorithms with data-hungry training strategy. To break this bottleneck, in this paper, we construct CAMotion, a high-quality benchmark covers a wide range of species for camouflaged moving object detection in the wild. CAMotion comprises various sequences with multiple challenging attributes such as uncertain edge, occlusion, motion blur, and shape complexity, etc. The sequence annotation details and statistical distribution are presented from various perspectives, allowing CAMotion to provide in-depth analyses on the camouflaged object's motion characteristics in different challenging scenarios. Additionally, we conduct a comprehensive evaluation of existing SOTA models on CAMotion, and discuss the major challenges in VCOD task. The benchmark is available at https://www.camotion.focuslab.net.cn, we hope that our CAMotion can lead to further advancements in the research community.

  </details>



- **Revisiting Radar Perception With Spectral Point Clouds**  
  Hamza Alsharif, Jing Gu, Pavol Jancura, Satish Ravindran, Gijs Dubbelman  
  _2026-04-09_ · https://arxiv.org/abs/2604.08282v1  
  <details><summary>Abstract</summary>

  Radar perception models are trained with different inputs, from range-Doppler spectra to sparse point clouds. Dense spectra are assumed to outperform sparse point clouds, yet they can vary considerably across sensors and configurations, which hinders transfer. In this paper, we provide alternatives for incorporating spectral information into radar point clouds and show that, point clouds need not underperform compared to spectra. We introduce the spectral point cloud paradigm, where point clouds are treated as sparse, compressed representations of the radar spectra, and argue that, when enriched with spectral information, they serve as strong candidates for a unified input representation that is more robust against sensor-specific differences. We develop an experimental framework that compares spectral point cloud (PC) models at varying densities against a dense range-Doppler (RD) benchmark, and report the density levels where the PC configurations meet the performance of the RD benchmark. Furthermore, we experiment with two basic spectral enrichment approaches, that inject additional target-relevant information into the point clouds. Contrary to the common belief that the dense RD approach is superior, we show that point clouds can do just as well, and can surpass the RD benchmark when enrichment is applied. Spectral point clouds can therefore serve as strong candidates for unified radar perception, paving the way for future radar foundation models.

  </details>



- **Orion-Lite: Distilling LLM Reasoning into Efficient Vision-Only Driving Models**  
  Jing Gu, Niccolò Cavagnero, Gijs Dubbelman  
  _2026-04-09_ · https://arxiv.org/abs/2604.08266v1  
  <details><summary>Abstract</summary>

  Leveraging the general world knowledge of Large Language Models (LLMs) holds significant promise for improving the ability of autonomous driving systems to handle rare and complex scenarios. While integrating LLMs into Vision-Language-Action (VLA) models has yielded state-of-the-art performance, their massive parameter counts pose severe challenges for latency-sensitive and energy-efficient deployment. Distilling LLM knowledge into a compact driving model offers a compelling solution to retain these reasoning capabilities while maintaining a manageable computational footprint. Although previous works have demonstrated the efficacy of distillation, these efforts have primarily focused on relatively simple scenarios and open-loop evaluations. Therefore, in this work, we investigate LLM distillation in more complex, interactive scenarios under closed-loop evaluation. We demonstrate that through a combination of latent feature distillation and ground-truth trajectory supervision, an efficient vision-only student model \textbf{Orion-Lite} can even surpass the performance of its massive VLA teacher, ORION. Setting a new state-of-the-art on the rigorous Bench2Drive benchmark, with a Driving Score of 80.6. Ultimately, this reveals that vision-only architectures still possess significant, untapped potential for high-performance reactive planning.

  </details>



- **EvoGymCM: Harnessing Continuous Material Stiffness for Soft Robot Co-Design**  
  Le Shen, Kangyao Huang, Wentao Zhao, Huaping Liu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08258v1  
  <details><summary>Abstract</summary>

  In the automated co-design of soft robots, precisely adapting the material stiffness field to task environments is crucial for unlocking their full physical potential. However, mainstream platforms (e.g., EvoGym) strictly discretize the material dimension, artificially restricting the design space and performance of soft robots. To address this, we propose EvoGymCM (EvoGym with Continuous Materials), a benchmark suite formally establishing continuous material stiffness as a first-class design variable alongside morphology and control. Aligning with real-world material mechanisms, EvoGymCM introduces two settings: (i) EvoGymCM-R (Reactive), motivated by programmable materials with dynamically tunable stiffness; and (ii) EvoGymCM-I (Invariant), motivated by traditional materials with invariant stiffness fields. To tackle the resulting high-dimensional coupling, we formulate two Morphology-Material-Control co-design paradigms: (i) Reactive-Material Co-Design, which learns real-time stiffness tuning policies to guide programmable materials; and (ii) Invariant-Material Co-Design, which jointly optimizes morphology and fixed material fields to guide traditional material fabrication. Systematic experiments across diverse tasks demonstrate that continuous material optimization boosts performance and unlocks synergy across morphology, material, and control.

  </details>



- **$\oslash$ Source Models Leak What They Shouldn't $\nrightarrow$: Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial Optimization**  
  Arnav Devalapally, Poornima Jain, Kartik Srinivas, Vineeth N. Balasubramanian  
  _2026-04-09_ · https://arxiv.org/abs/2604.08238v1  
  <details><summary>Abstract</summary>

  The increasing adaptation of vision models across domains, such as satellite imagery and medical scans, has raised an emerging privacy risk: models may inadvertently retain and leak sensitive source-domain specific information in the target domain. This creates a compelling use case for machine unlearning to protect the privacy of sensitive source-domain data. Among adaptation techniques, source-free domain adaptation (SFDA) calls for an urgent need for machine unlearning (MU), where the source data itself is protected, yet the source model exposed during adaptation encodes its influence. Our experiments reveal that existing SFDA methods exhibit strong zero-shot performance on source-exclusive classes in the target domain, indicating they inadvertently leak knowledge of these classes into the target domain, even when they are not represented in the target data. We identify and address this risk by proposing an MU setting called SCADA-UL: Unlearning Source-exclusive ClAsses in Domain Adaptation. Existing MU methods do not address this setting as they are not designed to handle data distribution shifts. We propose a new unlearning method, where an adversarially generated forget class sample is unlearned by the model during the domain adaptation process using a novel rescaled labeling strategy and adversarial optimization. We also extend our study to two variants: a continual version of this problem setting and to one where the specific source classes to be forgotten may be unknown. Alongside theoretical interpretations, our comprehensive empirical results show that our method consistently outperforms baselines in the proposed setting while achieving retraining-level unlearning performance on benchmark datasets. Our code is available at https://github.com/D-Arnav/SCADA

  </details>



- **EditCaption: Human-Aligned Instruction Synthesis for Image Editing via Supervised Fine-Tuning and Direct Preference Optimization**  
  Xiangyuan Wang, Honghao Cai, Yunhao Bai, Tianze Zhou, Haohua Chen, Yao Hu, Xu Tang, Yibo Chen, Wei Zhu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08213v1  
  <details><summary>Abstract</summary>

  High-quality training triplets (source-target image pairs with precise editing instructions) are a critical bottleneck for scaling instruction-guided image editing models. Vision-language models (VLMs) are widely used for automated instruction synthesis, but we identify three systematic failure modes in image-pair settings: orientation inconsistency (e.g., left/right confusion), viewpoint ambiguity, and insufficient fine-grained attribute description. Human evaluation shows that over 47% of instructions from strong baseline VLMs contain critical errors unusable for downstream training. We propose EditCaption, a scalable two-stage post-training pipeline for VLM-based instruction synthesis. Stage 1 builds a 100K supervised fine-tuning (SFT) dataset by combining GLM automatic annotation, EditScore-based filtering, and human refinement for spatial, directional, and attribute-level accuracy. Stage 2 collects 10K human preference pairs targeting the three failure modes and applies direct preference optimization (DPO) for alignment beyond SFT alone. On Eval-400, ByteMorph-Bench, and HQ-Edit, fine-tuned Qwen3-VL models outperform open-source baselines; the 235B model reaches 4.712 on Eval-400 (vs. Gemini-3-Pro 4.706, GPT-4.1 4.220, Kimi-K2.5 4.111) and 4.588 on ByteMorph-Bench (vs. Gemini-3-Pro 4.522, GPT-4.1 3.412). Human evaluation shows critical errors falling from 47.75% to 23% and correctness rising from 41.75% to 66%. The work offers a practical path to scalable, human-aligned instruction synthesis for image editing data.

  </details>



- **Vision-Language Foundation Models for Comprehensive Automated Pavement Condition Assessment**  
  Blessing Agyei Kyem, Joshua Kofi Asamoah, Anthony Dontoh, Armstrong Aboah  
  _2026-04-09_ · https://arxiv.org/abs/2604.08212v1  
  <details><summary>Abstract</summary>

  General-purpose vision-language models demonstrate strong performance in everyday domains but struggle with specialized technical fields requiring precise terminology, structured reasoning, and adherence to engineering standards. This work addresses whether domain-specific instruction tuning can enable comprehensive pavement condition assessment through vision-language models. PaveInstruct, a dataset containing 278,889 image-instruction-response pairs spanning 32 task types, was created by unifying annotations from nine heterogeneous pavement datasets. PaveGPT, a pavement foundation model trained on this dataset, was evaluated against state-of-the-art vision-language models across perception, understanding, and reasoning tasks. Instruction tuning transformed model capabilities, achieving improvements exceeding 20% in spatial grounding, reasoning, and generation tasks while producing ASTM D6433-compliant outputs. These results enable transportation agencies to deploy unified conversational assessment tools that replace multiple specialized systems, simplifying workflows and reducing technical expertise requirements. The approach establishes a pathway for developing instruction-driven AI systems across infrastructure domains including bridge inspection, railway maintenance, and building condition assessment.

  </details>



- **SciFigDetect: A Benchmark for AI-Generated Scientific Figure Detection**  
  You Hu, Chenzhuo Zhao, Changfa Mo, Haotian Liu, Xiaobai Li  
  _2026-04-09_ · https://arxiv.org/abs/2604.08211v1  
  <details><summary>Abstract</summary>

  Modern multimodal generators can now produce scientific figures at near-publishable quality, creating a new challenge for visual forensics and research integrity. Unlike conventional AI-generated natural images, scientific figures are structured, text-dense, and tightly aligned with scholarly semantics, making them a distinct and difficult detection target. However, existing AI-generated image detection benchmarks and methods are almost entirely developed for open-domain imagery, leaving this setting largely unexplored. We present the first benchmark for AI-generated scientific figure detection. To construct it, we develop an agent-based data pipeline that retrieves licensed source papers, performs multimodal understanding of paper text and figures, builds structured prompts, synthesizes candidate figures, and filters them through a review-driven refinement loop. The resulting benchmark covers multiple figure categories, multiple generation sources and aligned real--synthetic pairs. We benchmark representative detectors under zero-shot, cross-generator, and degraded-image settings. Results show that current methods fail dramatically in zero-shot transfer, exhibit strong generator-specific overfitting, and remain fragile under common post-processing corruptions. These findings reveal a substantial gap between existing AIGI detection capabilities and the emerging distribution of high-quality scientific figures. We hope this benchmark can serve as a foundation for future research on robust and generalizable scientific-figure forensics. The dataset is available at https://github.com/Joyce-yoyo/SciFigDetect.

  </details>



- **MedVR: Annotation-Free Medical Visual Reasoning via Agentic Reinforcement Learning**  
  Zheng Jiang, Heng Guo, Chengyu Fang, Changchen Xiao, Xinyang Hu, Lifeng Sun, Minfeng Xu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08203v1  
  <details><summary>Abstract</summary>

  Medical Vision-Language Models (VLMs) hold immense promise for complex clinical tasks, but their reasoning capabilities are often constrained by text-only paradigms that fail to ground inferences in visual evidence. This limitation not only curtails performance on tasks requiring fine-grained visual analysis but also introduces risks of visual hallucination in safety-critical applications. Thus, we introduce MedVR, a novel reinforcement learning framework that enables annotation-free visual reasoning for medical VLMs. Its core innovation lies in two synergistic mechanisms: Entropy-guided Visual Regrounding (EVR) uses model uncertainty to direct exploration, while Consensus-based Credit Assignment (CCA) distills pseudo-supervision from rollout agreement. Without any human annotations for intermediate steps, MedVR achieves state-of-the-art performance on diverse public medical VQA benchmarks, significantly outperforming existing models. By learning to reason directly with visual evidence, MedVR promotes the robustness and transparency essential for accelerating the clinical deployment of medical AI.

  </details>



- **OceanMAE: A Foundation Model for Ocean Remote Sensing**  
  Viola-Joanna Stamer, Panagiotis Agrafiotis, Behnood Rasti, Begüm Demir  
  _2026-04-09_ · https://arxiv.org/abs/2604.08171v1  
  <details><summary>Abstract</summary>

  Accurate ocean mapping is essential for applications such as bathymetry estimation, seabed characterization, marine litter detection, and ecosystem monitoring. However, ocean remote sensing (RS) remains constrained by limited labeled data and by the reduced transferability of models pre-trained mainly on land-dominated Earth observation imagery. In this paper, we propose OceanMAE, an ocean-specific masked autoencoder that extends standard MAE pre-training by integrating multispectral Sentinel-2 observations with physically meaningful ocean descriptors during self-supervised learning. By incorporating these auxiliary ocean features, OceanMAE is designed to learn more informative and ocean-aware latent representations from large- scale unlabeled data. To transfer these representations to downstream applications, we further employ a modified UNet-based framework for marine segmentation and bathymetry estimation. Pre-trained on the Hydro dataset, OceanMAE is evaluated on MADOS and MARIDA for marine pollutant and debris segmentation, and on MagicBathyNet for bathymetry regression. The experiments show that OceanMAE yields the strongest gains on marine segmentation, while bathymetry benefits are competitive and task-dependent. In addition, an ablation against a standard MAE on MARIDA indicates that incorporating auxiliary ocean descriptors during pre-training improves downstream segmentation quality. These findings highlight the value of physically informed and domain-aligned self-supervised pre- training for ocean RS. Code and weights are publicly available at https://git.tu-berlin.de/joanna.stamer/SSLORS2.

  </details>



- **T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation**  
  Pranjal Khadka  
  _2026-04-09_ · https://arxiv.org/abs/2604.08167v1  
  <details><summary>Abstract</summary>

  Medical image segmentation traditionally relies on fully supervised 3D architectures that demand a large amount of dense, voxel-level annotations from clinical experts which is a prohibitively expensive process. Vision Language Models (VLMs) offer a powerful alternative by leveraging broad visual semantic representations learned from billions of images. However, when applied independently to 2D slices of a 3D scan, these models often produce noisy and anatomically implausible segmentations that violate the inherent continuity of anatomical structures. We propose a temporal adapter that addresses this by injecting adjacent-slice context directly into the model's visual token representations. The adapter comprises a temporal transformer attending across a fixed context window at the token level, a spatial context block refining within-slice representations, and an adaptive gate balancing temporal and single-slice features. Training on 30 labeled volumes from the FLARE22 dataset, our method achieves a mean Dice of 0.704 across 13 abdominal organs with a gain of +0.206 over the baseline VLM trained with no temporal context. Zero-shot evaluation on BTCV and AMOS22 datasets yields consistent improvements of +0.210 and +0.230, with the average cross-domain performance drop reducing from 38.0% to 24.9%. Furthermore, in a cross-modality evaluation on AMOS22 MRI with neither model receiving any MRI supervision, our method achieves a mean Dice of 0.366, outperforming a fully supervised 3D baseline (DynUNet, 0.224) trained exclusively on CT, suggesting that CLIP's visual semantic representations generalize more gracefully across imaging modalities than convolutional features.

  </details>



- **EPIR: An Efficient Patch Tokenization, Integration and Representation Framework for Micro-expression Recognition**  
  Junbo Wang, Liangyu Fu, Yuke Li, Yining Zhu, Xuecheng Wu, Kun Hu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08106v1  
  <details><summary>Abstract</summary>

  Micro-expression recognition can obtain the real emotion of the individual at the current moment. Although deep learning-based methods, especially Transformer-based methods, have achieved impressive results, these methods have high computational complexity due to the large number of tokens in the multi-head self-attention. In addition, the existing micro-expression datasets are small-scale, which makes it difficult for Transformer-based models to learn effective micro-expression representations. Therefore, we propose a novel Efficient Patch tokenization, Integration and Representation framework (EPIR), which can balance high recognition performance and low computational complexity. Specifically, we first propose a dual norm shifted tokenization (DNSPT) module to learn the spatial relationship between neighboring pixels in the face region, which is implemented by a refined spatial transformation and dual norm projection. Then, we propose a token integration module to integrate partial tokens among multiple cascaded Transformer blocks, thereby reducing the number of tokens without information loss. Furthermore, we design a discriminative token extractor, which first improves the attention in the Transformer block to reduce the unnecessary focus of the attention calculation on self-tokens, and uses the dynamic token selection module (DTSM) to select key tokens, thereby capturing more discriminative micro-expression representations. We conduct extensive experiments on four popular public datasets (i.e., CASME II, SAMM, SMIC, and CAS(ME)3. The experimental results show that our method achieves significant performance gains over the state-of-the-art methods, such as 9.6% improvement on the CAS(ME)$^3$ dataset in terms of UF1 and 4.58% improvement on the SMIC dataset in terms of UAR metric.

  </details>



- **DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather**  
  Christof Leitgeb, Thomas Puchleitner, Max Peter Ronecker, Daniel Watzenig  
  _2026-04-09_ · https://arxiv.org/abs/2604.08074v1  
  <details><summary>Abstract</summary>

  Reliable and weather-robust perception systems are essential for safe autonomous driving and typically employ multi-modal sensor configurations to achieve comprehensive environmental awareness. While recent automotive FMCW Radar-based approaches achieved remarkable performance on detection tasks in adverse weather conditions, they exhibited limitations in resolving fine-grained spatial details particularly critical for detecting smaller and vulnerable road users (VRUs). Furthermore, existing research has not adequately addressed VRU detection in adverse weather datasets such as K-Radar. We present DinoRADE, a Radar-centered detection pipeline that processes dense Radar tensors and aggregates vision features around transformed reference points in the camera perspective via deformable cross-attention. Vision features are provided by a DINOv3 Vision Foundation Model. We present a comprehensive performance evaluation on the K-Radar dataset in all weather conditions and are among the first to report detection performance individually for five object classes. Additionally, we compare our method with existing single-class detection approaches and outperform recent Radar-camera approaches by 12.1%. The code is available under https://github.com/chr-is-tof/RADE-Net.

  </details>



- **Tensor-Augmented Convolutional Neural Networks: Enhancing Expressivity with Generic Tensor Kernels**  
  Chia-Wei Hsing, Wei-Lin Tu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08072v1  
  <details><summary>Abstract</summary>

  Convolutional Neural Networks (CNNs) excel at extracting local features hierarchically, but their performance in capturing complex correlations hinges heavily on deep architectures, which are usually computationally demanding and difficult to interpret. To address these issues, we propose a physically-guided shallow model: tensor-augmented CNN (TACNN), which replaces conventional convolution kernels with generic tensors to enhance representational capacity. This choice is motivated by the fact that an order-$N$ tensor naturally encodes an arbitrary quantum superposition state in the Hilbert space of dimension $d^N$, where $d$ is the local physical dimension, thus offering substantially richer expressivity. Furthermore, in our design the convolution output of each layer becomes a multilinear form capable of capturing high-order feature correlations, thereby equipping a shallow multilayer architecture with an expressive power competitive to that of deep CNNs. On the Fashion-MNIST benchmark, TACNN demonstrates clear advantages over conventional CNNs, achieving remarkable accuracies with only a few layers. In particular, a TACNN with only two convolution layers attains a test accuracy of 93.7$\%$, surpassing or matching considerably deeper models such as VGG-16 (93.5$\%$) and GoogLeNet (93.7$\%$). These findings highlight TACNN as a promising framework that strengthens model expressivity while preserving architectural simplicity, paving the way towards more interpretable and efficient deep learning models.

  </details>



- **AtlasOCR: Building the First Open-Source Darija OCR Model with Vision Language Models**  
  Imane Momayiz, Soufiane Ait Elaouad, Abdeljalil Elmajjodi, Haitame Bouanane  
  _2026-04-09_ · https://arxiv.org/abs/2604.08070v1  
  <details><summary>Abstract</summary>

  Darija, the Moroccan Arabic dialect, is rich in visual content yet lacks specialized Optical Character Recognition (OCR) tools. This paper introduces AtlasOCR, the first open-source Darija OCR model built by fine-tuning a 3B parameter Vision Language Model (VLM). We detail our comprehensive approach, from curating a unique Darija-specific dataset leveraging both synthetic generation with our OCRSmith library and carefully sourced real-world data, to implementing efficient fine-tuning strategies. We utilize QLoRA and Unsloth for parameter-efficient training of Qwen2.5-VL 3B and present comprehensive ablation studies optimizing key hyperparameters. Our evaluation on the newly curated AtlasOCRBench and the established KITAB-Bench demonstrates state-of-the-art performance, challenging larger models and highlighting AtlasOCR's robustness and generalization capabilities for both Darija and standard Arabic OCR tasks.

  </details>



- **Adapting Foundation Models for Annotation-Efficient Adnexal Mass Segmentation in Cine Images**  
  Francesca Fati, Alberto Rota, Adriana V. Gregory, Anna Catozzo, Maria C. Giuliano, Mrinal Dhar, Luigi De Vitis, Annie T. Packard, Francesco Multinu, Elena De Momi, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.08045v1  
  <details><summary>Abstract</summary>

  Adnexal mass evaluation via ultrasound is a challenging clinical task, often hindered by subjective interpretation and significant inter-observer variability. While automated segmentation is a foundational step for quantitative risk assessment, traditional fully supervised convolutional architectures frequently require large amounts of pixel-level annotations and struggle with domain shifts common in medical imaging. In this work, we propose a label-efficient segmentation framework that leverages the robust semantic priors of a pretrained DINOv3 foundational vision transformer backbone. By integrating this backbone with a Dense Prediction Transformer (DPT)-style decoder, our model hierarchically reassembles multi-scale features to combine global semantic representations with fine-grained spatial details. Evaluated on a clinical dataset of 7,777 annotated frames from 112 patients, our method achieves state-of-the-art performance compared to established fully supervised baselines, including U-Net, U-Net++, DeepLabV3, and MAnet. Specifically, we obtain a Dice score of 0.945 and improved boundary adherence, reducing the 95th-percentile Hausdorff Distance by 11.4% relative to the strongest convolutional baseline. Furthermore, we conduct an extensive efficiency analysis demonstrating that our DINOv3-based approach retains significantly higher performance under data starvation regimes, maintaining strong results even when trained on only 25% of the data. These results suggest that leveraging large-scale self-supervised foundations provides a promising and data-efficient solution for medical image segmentation in data-constrained clinical environments. Project Repository: https://github.com/FrancescaFati/MESA

  </details>



- **Beyond Mamba: Enhancing State-space Models with Deformable Dilated Convolutions for Multi-scale Traffic Object Detection**  
  Jun Li, Yingying Shi, Zhixuan Ruan, Nan Guo, Jianhua Xu  
  _2026-04-09_ · https://arxiv.org/abs/2604.08038v1  
  <details><summary>Abstract</summary>

  In a real-world traffic scenario, varying-scale objects are usually distributed in a cluttered background, which poses great challenges to accurate detection. Although current Mamba-based methods can efficiently model long-range dependencies, they still struggle to capture small objects with abundant local details, which hinders joint modeling of local structures and global semantics. Moreover, state-space models exhibit limited hierarchical feature representation and weak cross-scale interaction due to flat sequential modeling and insufficient spatial inductive biases, leading to sub-optimal performance in complex scenes. To address these issues, we propose a Mamba with Deformable Dilated Convolutions Network (MDDCNet) for accurate traffic object detection in this study. In MDDCNet, a well-designed hybrid backbone with successive Multi-Scale Deformable Dilated Convolution (MSDDC) blocks and Mamba blocks enables hierarchical feature representation from local details to global semantics. Meanwhile, a Channel-Enhanced Feed-Forward Network (CE-FFN) is further devised to overcome the limited channel interaction capability of conventional feed-forward networks, whilst a Mamba-based Attention-Aggregating Feature Pyramid Network (A^2FPN) is constructed to achieve enhanced multi-scale feature fusion and interaction. Extensive experimental results on public benchmark and real-world datasets demonstrate the superiority of our method over various advanced detectors. The code is available at https://github.com/Bettermea/MDDCNet.

  </details>



- **Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles**  
  Jiawei Liu, Xun Gong, Fen Fang, Muli Yang, Bohao Qu, Yunfeng Hu, Hong Chen, Xulei Yang, Qing Guo  
  _2026-04-09_ · https://arxiv.org/abs/2604.08031v1  
  <details><summary>Abstract</summary>

  Most Human-Machine Interaction (HMI) research overlooks the maneuvering needs of passengers in autonomous driving (AD). Natural language offers an intuitive interface, yet translating passenger open-ended instructions into control signals, without sacrificing interpretability and traceability, remains a challenge. This study proposes an instruction-realization framework that leverages a large language model (LLM) to interpret instructions, generates executable scripts that schedule multiple model predictive control (MPC)-based motion planners based on real-time feedback, and converts planned trajectories into control signals. This scheduling-centric design decouples semantic reasoning from vehicle control at different timescales, establishing a transparent, traceable decision-making chain from high-level instructions to low-level actions. Due to the absence of high-fidelity evaluation tools, this study introduces a benchmark for open-ended instruction realization in a closed-loop setting. Comprehensive experiments reveal that the framework significantly improves task-completion rates over instruction-realization baselines, reduces LLM query costs, achieves safety and compliance on par with specialized AD approaches, and exhibits considerable tolerance to LLM inference latency. For more qualitative illustrations and a clearer understanding.

  </details>



- **Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI**  
  Minh Sao Khue Luu, Evgeniy N. Pavlovskiy, Bair N. Tuchinov  
  _2026-04-09_ · https://arxiv.org/abs/2604.08015v1  
  <details><summary>Abstract</summary>

  We propose a unified objective function, termed CATMIL, that augments the base segmentation loss with two auxiliary supervision terms operating at different levels. The first term, Component-Adaptive Tversky, reweights voxel contributions based on connected components to balance the influence of lesions of different sizes. The second term, based on Multiple Instance Learning, introduces lesion-level supervision by encouraging the detection of each lesion instance. These terms are combined with the standard nnU-Net loss to jointly optimize voxel-level segmentation accuracy and lesion-level detection. We evaluate the proposed objective on the MSLesSeg dataset using a consistent nnU-Net framework and 5-fold cross-validation. The results show that CATMIL achieves the most balanced performance across segmentation accuracy, lesion detection, and error control. It improves Dice score (0.7834) and reduces boundary error compared to standard losses. More importantly, it substantially increases small lesion recall and reduces false negatives, while maintaining the lowest false positive volume among compared methods. These findings demonstrate that integrating component-level and lesion-level supervision within a unified objective provides an effective and practical approach for improving small lesion segmentation in highly imbalanced settings. All code and pretrained models are available at \href{https://github.com/luumsk/SmallLesionMRI}{this url}.

  </details>



- **SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving**  
  Felix Embacher, Jonas Uhrig, Marius Cordts, Markus Enzweiler  
  _2026-04-09_ · https://arxiv.org/abs/2604.08008v1  
  <details><summary>Abstract</summary>

  Retrieving rare and safety-critical driving scenarios from large-scale datasets is essential for building robust autonomous driving (AD) systems. As dataset sizes continue to grow, the key challenge shifts from collecting more data to efficiently identifying the most relevant samples. We introduce SearchAD, a large-scale rare image retrieval dataset for AD containing over 423k frames drawn from 11 established datasets. SearchAD provides high-quality manual annotations of more than 513k bounding boxes covering 90 rare categories. It specifically targets the needle-in-a-haystack problem of locating extremely rare classes, with some appearing fewer than 50 times across the entire dataset. Unlike existing benchmarks, which focused on instance-level retrieval, SearchAD emphasizes semantic image retrieval with a well-defined data split, enabling text-to-image and image-to-image retrieval, few-shot learning, and fine-tuning of multi-modal retrieval models. Comprehensive evaluations show that text-based methods outperform image-based ones due to stronger inherent semantic grounding. While models directly aligning spatial visual features with language achieve the best zero-shot results, and our fine-tuning baseline significantly improves performance, absolute retrieval capabilities remain unsatisfactory. With a held-out test set on a public benchmark server, SearchAD establishes the first large-scale dataset for retrieval-driven data curation and long-tail perception research in AD: https://iis-esslingen.github.io/searchad/

  </details>



- **PASK: Toward Intent-Aware Proactive Agents with Long-Term Memory**  
  Zhifei Xie, Zongzheng Hu, Fangda Ye, Xin Zhang, Haobo Chai, Zihang Liu, Pengcheng Wu, Guibin Zhang, Yue Liao, Xiaobin Hu, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.08000v1  
  <details><summary>Abstract</summary>

  Proactivity is a core expectation for AGI. Prior work remains largely confined to laboratory settings, leaving a clear gap in real-world proactive agent: depth, complexity, ambiguity, precision and real-time constraints. We study this setting, where useful intervention requires inferring latent needs from ongoing context and grounding actions in evolving user memory under latency and long-horizon constraints. We first propose DD-MM-PAS (Demand Detection, Memory Modeling, Proactive Agent System) as a general paradigm for streaming proactive AI agent. We instantiate this paradigm in Pask, with streaming IntentFlow model for DD, a hybrid memory (workspace, user, global) for long-term MM, PAS infra framework and introduce how these components form a closed loop. We also introduce LatentNeeds-Bench, a real-world benchmark built from user-consented data and refined through thousands of rounds of human editing. Experiments show that IntentFlow matches leading Gemini3-Flash models under latency constraints, while identifying deeper user intent.

  </details>



- **MotionScape: A Large-Scale Real-World Highly Dynamic UAV Video Dataset for World Models**  
  Zile Guo, Zhan Chen, Enze Zhu, Kan Wei, Yongkang Zou, Xiaoxuan Liu, Lei Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.07991v1  
  <details><summary>Abstract</summary>

  Recent advances in world models have demonstrated strong capabilities in simulating physical reality, making them an increasingly important foundation for embodied intelligence. For UAV agents in particular, accurate prediction of complex 3D dynamics is essential for autonomous navigation and robust decision-making in unconstrained environments. However, under the highly dynamic camera trajectories typical of UAV views, existing world models often struggle to maintain spatiotemporal physical consistency. A key reason lies in the distribution bias of current training data: most existing datasets exhibit restricted 2.5D motion patterns, such as ground-constrained autonomous driving scenes or relatively smooth human-centric egocentric videos, and therefore lack realistic high-dynamic 6-DoF UAV motion priors. To address this gap, we present MotionScape, a large-scale real-world UAV-view video dataset with highly dynamic motion for world modeling. MotionScape contains over 30 hours of 4K UAV-view videos, totaling more than 4.5M frames. This novel dataset features semantically and geometrically aligned training samples, where diverse real-world UAV videos are tightly coupled with accurate 6-DoF camera trajectories and fine-grained natural language descriptions. To build the dataset, we develop an automated multi-stage processing pipeline that integrates CLIP-based relevance filtering, temporal segmentation, robust visual SLAM for trajectory recovery, and large-language-model-driven semantic annotation. Extensive experiments show that incorporating such semantically and geometrically aligned annotations effectively improves the ability of existing world models to simulate complex 3D dynamics and handle large viewpoint shifts, thereby benefiting decision-making and planning for UAV agents in complex environments. The dataset is publicly available at https://github.com/Thelegendzz/MotionScape

  </details>



- **SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations**  
  Yunnan Wang, Kecheng Zheng, Jianyuan Wang, Minghao Chen, David Novotny, Christian Rupprecht, Yinghao Xu, Xing Zhu, Wenjun Zeng, Xin Jin, et al.  
  _2026-04-09_ · https://arxiv.org/abs/2604.07990v1  
  <details><summary>Abstract</summary>

  The convergence of 3D geometric perception and video synthesis has created an unprecedented demand for large-scale video data that is rich in both semantic and spatio-temporal information. While existing datasets have advanced either 3D understanding or video generation, a significant gap remains in providing a unified resource that supports both domains at scale. To bridge this chasm, we introduce SceneScribe-1M, a new large-scale, multi-modal video dataset. It comprises one million in-the-wild videos, each meticulously annotated with detailed textual descriptions, precise camera parameters, dense depth maps, and consistent 3D point tracks. We demonstrate the versatility and value of SceneScribe-1M by establishing benchmarks across a wide array of downstream tasks, including monocular depth estimation, scene reconstruction, and dynamic point tracking, as well as generative tasks such as text-to-video synthesis, with or without camera control. By open-sourcing SceneScribe-1M, we aim to provide a comprehensive benchmark and a catalyst for research, fostering the development of models that can both perceive the dynamic 3D world and generate controllable, realistic video content.

  </details>



- **Lighting-grounded Video Generation with Renderer-based Agent Reasoning**  
  Ziqi Cai, Taoyu Yang, Zheng Chang, Si Li, Han Jiang, Shuchen Weng, Boxin Shi  
  _2026-04-09_ · https://arxiv.org/abs/2604.07966v1  
  <details><summary>Abstract</summary>

  Diffusion models have achieved remarkable progress in video generation, but their controllability remains a major limitation. Key scene factors such as layout, lighting, and camera trajectory are often entangled or only weakly modeled, restricting their applicability in domains like filmmaking and virtual production where explicit scene control is essential. We present LiVER, a diffusion-based framework for scene-controllable video generation. To achieve this, we introduce a novel framework that conditions video synthesis on explicit 3D scene properties, supported by a new large-scale dataset with dense annotations of object layout, lighting, and camera parameters. Our method disentangles these properties by rendering control signals from a unified 3D representation. We propose a lightweight conditioning module and a progressive training strategy to integrate these signals into a foundational video diffusion model, ensuring stable convergence and high fidelity. Our framework enables a wide range of applications, including image-to-video and video-to-video synthesis where the underlying 3D scene is fully editable. To further enhance usability, we develop a scene agent that automatically translates high-level user instructions into the required 3D control signals. Experiments show that LiVER achieves state-of-the-art photorealism and temporal consistency while enabling precise, disentangled control over scene factors, setting a new standard for controllable video generation.

  </details>



- **On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning**  
  Amirhossein Afsharrad, Amirhesam Abedsoltan, Ahmadreza Moradipari, Sanjay Lall  
  _2026-04-09_ · https://arxiv.org/abs/2604.07944v1  
  <details><summary>Abstract</summary>

  Large language models (LLMs) have recently demonstrated strong potential for autonomous vehicle motion planning by reformulating trajectory prediction as a language generation problem. However, deploying capable LLMs in resource-constrained onboard systems remains a fundamental challenge. In this paper, we study how to effectively transfer motion planning knowledge from a large teacher LLM to a smaller, more deployable student model. We build on the GPT-Driver framework, which represents driving scenes as language prompts and generates waypoint trajectories with chain-of-thought reasoning, and investigate two student training paradigms: (i) on-policy generalized knowledge distillation (GKD), which trains the student on its own self-generated outputs using dense token-level feedback from the teacher, and (ii) a dense-feedback reinforcement learning (RL) baseline that uses the teacher's log-probabilities as per-token reward signals in a policy gradient framework. Experiments on the nuScenes benchmark show that GKD substantially outperforms the RL baseline and closely approaches teacher-level performance despite a 5$\times$ reduction in model size. These results highlight the practical value of on-policy distillation as a principled and effective approach to deploying LLM-based planners in autonomous driving systems.

  </details>



- **Shortcut Learning in Glomerular AI: Adversarial Penalties Hurt, Entropy Helps**  
  Mohammad Daouk, Jan Ulrich Becker, Neeraja Kambham, Anthony Chang, Hien Nguyen, Chandra Mohan  
  _2026-04-09_ · https://arxiv.org/abs/2604.07936v1  
  <details><summary>Abstract</summary>

  Stain variability is a pervasive source of distribution shift and potential shortcut learning in renal pathology AI. We ask whether lupus nephritis glomerular lesion classifiers exploit stain as a shortcut, and how to mitigate such bias without stain or site labels. We curate a multi-center, multi-stain dataset of 9{,}674 glomerular patches (224$\times$224) from 365 WSIs across three centers and four stains (PAS, H\&E, Jones, Trichrome), labeled as proliferative vs.\ non-proliferative. We evaluate Bayesian CNN and ViT backbones with Monte Carlo dropout in three settings: (1) stain-only classification; (2) a dual-head model jointly predicting lesion and stain with supervised stain loss; and (3) a dual-head model with label-free stain regularization via entropy maximization on the stain head. In (1), stain identity is trivially learnable, confirming a strong candidate shortcut. In (2), varying the strength and sign of stain supervision strongly modulates stain performance but leaves lesion metrics essentially unchanged, indicating no measurable stain-driven shortcut learning on this multi-stain, multi-center dataset, while overly adversarial stain penalties inflate predictive uncertainty. In (3), entropy-based regularization holds stain predictions near chance without degrading lesion accuracy or calibration. Overall, a carefully curated multi-stain dataset can be inherently robust to stain shortcuts, and a Bayesian dual-head architecture with label-free entropy regularization offers a simple, deployment-friendly safeguard against potential stain-related drift in glomerular AI.

  </details>



- **Stitch4D: Sparse Multi-Location 4D Urban Reconstruction via Spatio-Temporal Interpolation**  
  Hina Kogure, Kei Katsumata, Taiki Miyanishi, Komei Sugiura  
  _2026-04-09_ · https://arxiv.org/abs/2604.07923v1  
  <details><summary>Abstract</summary>

  Dynamic urban environments are often captured by cameras placed at spatially separated locations with little or no view overlap. However, most existing 4D reconstruction methods assume densely overlapping views. When applied to such sparse observations, these methods fail to reconstruct intermediate regions and often introduce temporal artifacts. To address this practical yet underexplored sparse multi-location setting, we propose Stitch4D, a unified 4D reconstruction framework that explicitly compensates for missing spatial coverage in sparse observations. Stitch4D (i) synthesizes intermediate bridge views to densify spatial constraints and improve spatial coverage, and (ii) jointly optimizes real and synthesized observations within a unified coordinate frame under explicit inter-location consistency constraints. By restoring intermediate coverage before optimization, Stitch4D prevents geometric collapse and reconstructs coherent geometry and smooth scene dynamics even in sparsely observed environments. To evaluate this setting, we introduce Urban Sparse 4D (U-S4D), a CARLA-based benchmark designed to assess spatiotemporal alignment under sparse multi-location configurations. Experimental results on U-S4D show that Stitch4D surpasses representative 4D reconstruction baselines and achieves superior visual quality. These results indicate that recovering intermediate spatial coverage is essential for stable 4D reconstruction in sparse urban environments.

  </details>


