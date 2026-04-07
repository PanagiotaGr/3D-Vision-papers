# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **50**


---

- **LoMa: Local Feature Matching Revisited**  
  David Nordström, Johan Edstedt, Georg Bökman, Jonathan Astermark, Anders Heyden, Viktor Larsson, Mårten Wadenbäck, Michael Felsberg, Fredrik Kahl  
  _2026-04-06_ · https://arxiv.org/abs/2604.04931v1  
  <details><summary>Abstract</summary>

  Local feature matching has long been a fundamental component of 3D vision systems such as Structure-from-Motion (SfM), yet progress has lagged behind the rapid advances of modern data-driven approaches. The newer approaches, such as feed-forward reconstruction models, have benefited extensively from scaling dataset sizes, whereas local feature matching models are still only trained on a few mid-sized datasets. In this paper, we revisit local feature matching from a data-driven perspective. In our approach, which we call LoMa, we combine large and diverse data mixtures, modern training recipes, scaled model capacity, and scaled compute, resulting in remarkable gains in performance. Since current standard benchmarks mainly rely on collecting sparse views from successful 3D reconstructions, the evaluation of progress in feature matching has been limited to relatively easy image pairs. To address the resulting saturation of benchmarks, we collect 1000 highly challenging image pairs from internet data into a new dataset called HardMatch. Ground truth correspondences for HardMatch are obtained via manual annotation by the authors. In our extensive benchmarking suite, we find that LoMa makes outstanding progress across the board, outperforming the state-of-the-art method ALIKED+LightGlue by +18.6 mAA on HardMatch, +29.5 mAA on WxBS, +21.4 (1m, 10$^\circ$) on InLoc, +24.2 AUC on RUBIK, and +12.4 mAA on IMC 2022. We release our code and models publicly at https://github.com/davnords/LoMa.

  </details>



- **Rethinking Model Efficiency: Multi-Agent Inference with Large Models**  
  Sixun Dong, Juhua Hu, Steven Li, Wei Wen, Qi Qian  
  _2026-04-06_ · https://arxiv.org/abs/2604.04929v1  
  <details><summary>Abstract</summary>

  Most vision-language models (VLMs) apply a large language model (LLM) as the decoder, where the response tokens are generated sequentially through autoregression. Therefore, the number of output tokens can be the bottleneck of the end-to-end latency. However, different models may require vastly different numbers of output tokens to achieve comparable performance. In this work, we conduct a comprehensive analysis of the latency across different components of VLMs on simulated data. The experiment shows that a large model with fewer output tokens can be more efficient than a small model with a long output sequence. The empirical study on diverse real-world benchmarks confirms the observation that a large model can achieve better or comparable performance as a small model with significantly fewer output tokens. To leverage the efficiency of large models, we propose a multi-agent inference framework that keeps large models with short responses but transfers the key reasoning tokens from the small model when necessary. The comparison on benchmark tasks demonstrates that by reusing the reasoning tokens from small models, it can help approach the performance of a large model with its own reasoning, which confirms the effectiveness of our proposal.

  </details>



- **Vero: An Open RL Recipe for General Visual Reasoning**  
  Gabriel Sarch, Linrong Cai, Qunzhong Wang, Haoyang Wu, Danqi Chen, Zhuang Liu  
  _2026-04-06_ · https://arxiv.org/abs/2604.04917v1  
  <details><summary>Abstract</summary>

  What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) show such broad visual reasoning is within reach, but the recipe behind them remains unclear, locked behind proprietary reinforcement learning (RL) pipelines with non-public data. We introduce Vero, a family of fully open VLMs that matches or exceeds existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answer formats. Vero achieves state-of-the-art performance, improving over four base models by 3.7-5.5 points on average across VeroEval, our suite of 30 challenging benchmarks. Starting from Qwen3-VL-8B-Instruct, Vero outperforms Qwen3-VL-8B-Thinking on 23 of 30 benchmarks without additional proprietary thinking data. When trained from the same base model, Vero-600K exceeds existing RL datasets across task categories. Systematic ablations reveal that different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation, suggesting that broad data coverage is the primary driver of strong RL scaling. All data, code, and models are released.

  </details>



- **SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing**  
  Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, et al.  
  _2026-04-06_ · https://arxiv.org/abs/2604.04911v1  
  <details><summary>Abstract</summary>

  Image spatial editing performs geometry-driven transformations, allowing precise control over object layout and camera viewpoints. Current models are insufficient for fine-grained spatial manipulations, motivating a dedicated assessment suite. Our contributions are listed: (i) We introduce SpatialEdit-Bench, a complete benchmark that evaluates spatial editing by jointly measuring perceptual plausibility and geometric fidelity via viewpoint reconstruction and framing analysis. (ii) To address the data bottleneck for scalable training, we construct SpatialEdit-500k, a synthetic dataset generated with a controllable Blender pipeline that renders objects across diverse backgrounds and systematic camera trajectories, providing precise ground-truth transformations for both object- and camera-centric operations. (iii) Building on this data, we develop SpatialEdit-16B, a baseline model for fine-grained spatial editing. Our method achieves competitive performance on general editing while substantially outperforming prior methods on spatial manipulation tasks. All resources will be made public at https://github.com/EasonXiao-888/SpatialEdit.

  </details>



- **FileGram: Grounding Agent Personalization in File-System Behavioral Traces**  
  Shuai Liu, Shulin Tian, Kairui Hu, Yuhao Dong, Zhe Yang, Bo Li, Jingkang Yang, Chen Change Loy, Ziwei Liu  
  _2026-04-06_ · https://arxiv.org/abs/2604.04901v1  
  <details><summary>Abstract</summary>

  Coworking AI agents operating within local file systems are rapidly emerging as a paradigm in human-AI interaction; however, effective personalization remains limited by severe data constraints, as strict privacy barriers and the difficulty of jointly collecting multimodal real-world traces prevent scalable training and evaluation, and existing methods remain interaction-centric while overlooking dense behavioral traces in file-system operations; to address this gap, we propose FileGram, a comprehensive framework that grounds agent memory and personalization in file-system behavioral traces, comprising three core components: (1) FileGramEngine, a scalable persona-driven data engine that simulates realistic workflows and generates fine-grained multimodal action sequences at scale; (2) FileGramBench, a diagnostic benchmark grounded in file-system behavioral traces for evaluating memory systems on profile reconstruction, trace disentanglement, persona drift detection, and multimodal grounding; and (3) FileGramOS, a bottom-up memory architecture that builds user profiles directly from atomic actions and content deltas rather than dialogue summaries, encoding these traces into procedural, semantic, and episodic channels with query-time abstraction; extensive experiments show that FileGramBench remains challenging for state-of-the-art memory systems and that FileGramEngine and FileGramOS are effective, and by open-sourcing the framework, we hope to support future research on personalized memory-centric file-system agents.

  </details>



- **HorizonWeaver: Generalizable Multi-Level Semantic Editing for Driving Scenes**  
  Mauricio Soroco, Francesco Pittaluga, Zaid Tasneem, Abhishek Aich, Bingbing Zhuang, Wuyang Chen, Manmohan Chandraker, Ziyu Jiang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04887v1  
  <details><summary>Abstract</summary>

  Ensuring safety in autonomous driving requires scalable generation of realistic, controllable driving scenes beyond what real-world testing provides. Yet existing instruction guided image editors, trained on object-centric or artistic data, struggle with dense, safety-critical driving layouts. We propose HorizonWeaver, which tackles three fundamental challenges in driving scene editing: (1) multi-level granularity, requiring coherent object- and scene-level edits in dense environments; (2) rich high-level semantics, preserving diverse objects while following detailed instructions; and (3) ubiquitous domain shifts, handling changes in climate, layout, and traffic across unseen environments. The core of HorizonWeaver is a set of complementary contributions across data, model, and training: (1) Data: Large-scale dataset generation, where we build a paired real/synthetic dataset from Boreas, nuScenes, and Argoverse2 to improve generalization; (2) Model: Language-Guided Masks for fine-grained editing, where semantics-enriched masks and prompts enable precise, language-guided edits; and (3) Training: Content preservation and instruction alignment, where joint losses enforce scene consistency and instruction fidelity. Together, HorizonWeaver provides a scalable framework for photorealistic, instruction-driven editing of complex driving scenes, collecting 255K images across 13 editing categories and outperforming prior methods in L1, CLIP, and DINO metrics, achieving +46.4% user preference and improving BEV segmentation IoU by +33%. Project page: https://msoroco.github.io/horizonweaver/

  </details>



- **DIRECT: Video Mashup Creation via Hierarchical Multi-Agent Planning and Intent-Guided Editing**  
  Ke Li, Maoliang Li, Jialiang Chen, Jiayu Chen, Zihao Zheng, Shaoqi Wang, Xiang Chen  
  _2026-04-06_ · https://arxiv.org/abs/2604.04875v1  
  <details><summary>Abstract</summary>

  Video mashup creation represents a complex video editing paradigm that recomposes existing footage to craft engaging audio-visual experiences, demanding intricate orchestration across semantic, visual, and auditory dimensions and multiple levels. However, existing automated editing frameworks often overlook the cross-level multimodal orchestration to achieve professional-grade fluidity, resulting in disjointed sequences with abrupt visual transitions and musical misalignment. To address this, we formulate video mashup creation as a Multimodal Coherency Satisfaction Problem (MMCSP) and propose the DIRECT framework. Simulating a professional production pipeline, our hierarchical multi-agent framework decomposes the challenge into three cascade levels: the Screenwriter for source-aware global structural anchoring, the Director for instantiating adaptive editing intent and guidance, and the Editor for intent-guided shot sequence editing with fine-grained optimization. We further introduce Mashup-Bench, a comprehensive benchmark with tailored metrics for visual continuity and auditory alignment. Extensive experiments demonstrate that DIRECT significantly outperforms state-of-the-art baselines in both objective metrics and human subjective evaluation. Project page and code: https://github.com/AK-DREAM/DIRECT

  </details>



- **Unified Vector Floorplan Generation via Markup Representation**  
  Kaede Shiohara, Toshihiko Yamasaki  
  _2026-04-06_ · https://arxiv.org/abs/2604.04859v1  
  <details><summary>Abstract</summary>

  Automatic residential floorplan generation has long been a central challenge bridging architecture and computer graphics, aiming to make spatial design more efficient and accessible. While early methods based on constraint satisfaction or combinatorial optimization ensure feasibility, they lack diversity and flexibility. Recent generative models achieve promising results but struggle to generalize across heterogeneous conditional tasks, such as generation from site boundaries, room adjacency graphs, or partial layouts, due to their suboptimal representations. To address this gap, we introduce Floorplan Markup Language (FML), a general representation that encodes floorplan information within a single structured grammar, which casts the entire floorplan generation problem into a next token prediction task. Leveraging FML, we develop a transformer-based generative model, FMLM, capable of producing high-fidelity and functional floorplans under diverse conditions. Comprehensive experiments on the RPLAN dataset demonstrate that FMLM, despite being a single model, surpasses the previous task-specific state-of-the-art methods.

  </details>



- **The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models**  
  Runhao Mao, Hanshi Wang, Yixiang Yang, Qianli Ma, Jingmeng Zhou, Zhipeng Zhang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04857v1  
  <details><summary>Abstract</summary>

  The integration of Vision-Language Models (VLMs) into autonomous driving promises to solve long-tail scenarios, but this paradigm faces the critical and unaddressed challenge of catastrophic forgetting. The very fine-tuning process used to adapt these models to driving-specific data simultaneously erodes their invaluable pre-trained world knowledge, creating a self-defeating paradox that undermines the core reason for their use. This paper provides the first systematic investigation into this phenomenon. We introduce a new large-scale dataset of 180K scenes, which enables the first-ever benchmark specifically designed to quantify catastrophic forgetting in autonomous driving. Our analysis reveals that existing methods suffer from significant knowledge degradation. To address this, we propose the Drive Expert Adapter (DEA), a novel framework that circumvents this trade-off by shifting adaptation from the weight space to the prompt space. DEA dynamically routes inference through different knowledge experts based on scene-specific cues, enhancing driving-task performance without corrupting the model's foundational parameters. Extensive experiments demonstrate that our approach not only achieves state-of-the-art results on driving tasks but also effectively mitigates catastrophic forgetting, preserving the essential generalization capabilities that make VLMs a transformative force for autonomous systems. Data and model are released at FidelityDrivingBench.

  </details>



- **E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes**  
  Jiajun Zhai, Hao Shi, Shangwei Guo, Kailun Yang, Kaiwei Wang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04834v1  
  <details><summary>Abstract</summary>

  Robotic Vision-Language-Action (VLA) models generalize well for open-ended manipulation, but their perception is fragile under sensing-stage degradations such as extreme low light, motion blur, and black clipping. We present E-VLA, an event-augmented VLA framework that improves manipulation robustness when conventional frame-based vision becomes unreliable. Instead of reconstructing images from events, E-VLA directly leverages motion and structural cues in event streams to preserve semantic perception and perception-action consistency under adverse conditions. We build an open-source teleoperation platform with a DAVIS346 event camera and collect a real-world synchronized RGB-event-action manipulation dataset across diverse tasks and illumination settings. We also propose lightweight, pretrained-compatible event integration strategies and study event windowing and fusion for stable deployment. Experiments show that even a simple parameter-free fusion, i.e., overlaying accumulated event maps onto RGB images, could substantially improve robustness in dark and blur-heavy scenes: on Pick-Place at 20 lux, success increases from 0% (image-only) to 60% with overlay fusion and to 90% with our event adapter; under severe motion blur (1000 ms exposure), Pick-Place improves from 0% to 20-25%, and Sorting from 5% to 32.5%. Overall, E-VLA provides systematic evidence that event-driven perception can be effectively integrated into VLA models, pointing toward robust embodied intelligence beyond conventional frame-based imaging. Code and dataset will be available at https://github.com/JJayzee/E-VLA.

  </details>



- **AnyUser: Translating Sketched User Intent into Domestic Robots**  
  Songyuan Yang, Huibin Tan, Kailun Yang, Wenjing Yang, Shaowu Yang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04811v1  
  <details><summary>Abstract</summary>

  We introduce AnyUser, a unified robotic instruction system for intuitive domestic task instruction via free-form sketches on camera images, optionally with language. AnyUser interprets multimodal inputs (sketch, vision, language) as spatial-semantic primitives to generate executable robot actions requiring no prior maps or models. Novel components include multimodal fusion for understanding and a hierarchical policy for robust action generation. Efficacy is shown via extensive evaluations: (1) Quantitative benchmarks on the large-scale dataset showing high accuracy in interpreting diverse sketch-based commands across various simulated domestic scenes. (2) Real-world validation on two distinct robotic platforms, a statically mounted 7-DoF assistive arm (KUKA LBR iiwa) and a dual-arm mobile manipulator (Realman RMC-AIDAL), performing representative tasks like targeted wiping and area cleaning, confirming the system's ability to ground instructions and execute them reliably in physical environments. (3) A comprehensive user study involving diverse demographics (elderly, simulated non-verbal, low technical literacy) demonstrating significant improvements in usability and task specification efficiency, achieving high task completion rates (85.7%-96.4%) and user satisfaction. AnyUser bridges the gap between advanced robotic capabilities and the need for accessible non-expert interaction, laying the foundation for practical assistive robots adaptable to real-world human environments.

  </details>



- **Multi-Modal Sensor Fusion using Hybrid Attention for Autonomous Driving**  
  Mayank Mayank, Bharanidhar Duraisamy, Florian Geiß, Abhinav Valada  
  _2026-04-06_ · https://arxiv.org/abs/2604.04797v1  
  <details><summary>Abstract</summary>

  Accurate 3D object detection for autonomous driving requires complementary sensors. Cameras provide dense semantics but unreliable depth, while millimeter-wave radar offers precise range and velocity measurements with sparse geometry. We propose MMF-BEV, a radar-camera BEV fusion framework that leverages deformable attention for cross-modal feature alignment on the View-of-Delft (VoD) 4D radar dataset [1]. MMF-BEV builds a BEVDepth [2] camera branch and a RadarBEVNet [3] radar branch, each enhanced with Deformable Self-Attention, and fuses them via a Deformable Cross-Attention module. We evaluate three configurations: camera-only, radar-only, and hybrid fusion. A sensor contribution analysis quantifies per-distance modality weighting, providing interpretable evidence of sensor complementarity. A two-stage training strategy - pre-training the camera branch with depth supervision, then jointly training radar and fusion modules stabilizes learning. Experiments on VoD show that MMF-BEV consistently outperforms unimodal baselines and achieves competitive results against prior fusion methods across all object classes in both the full annotated area and near-range Region of Interest.

  </details>



- **CLEAR: Unlocking Generative Potential for Degraded Image Understanding in Unified Multimodal Models**  
  Xiangzhao Hao, Zefeng Zhang, Zhenyu Zhang, Linhao Yu, Yao Chen, Yiqian Zhang, Haiyun Guo, Shuohuan Wang, Yu Sun  
  _2026-04-06_ · https://arxiv.org/abs/2604.04780v1  
  <details><summary>Abstract</summary>

  Image degradation from blur, noise, compression, and poor illumination severely undermines multimodal understanding in real-world settings. Unified multimodal models that combine understanding and generation within a single architecture are a natural fit for this challenge, as their generative pathway can model the fine-grained visual structure that degradation destroys. Yet these models fail to leverage their own generative capacity on degraded inputs. We trace this disconnect to two compounding factors: existing training regimes never ask the model to invoke generation during reasoning, and the standard decode-reencode pathway does not support effective joint optimization. We present CLEAR, a framework that connects the two capabilities through three progressive steps: (1) supervised fine-tuning on a degradation-aware dataset to establish the generate-then-answer reasoning pattern; (2) a Latent Representation Bridge that replaces the decode-reencode detour with a direct, optimizable connection between generation and reasoning; (3) Interleaved GRPO, a reinforcement learning method that jointly optimizes text reasoning and visual generation under answer-correctness rewards. We construct MMD-Bench, covering three degradation severity levels across six standard multimodal benchmarks. Experiments show that CLEAR substantially improves robustness on degraded inputs while preserving clean-image performance. Our analysis further reveals that removing pixel-level reconstruction supervision leads to intermediate visual states with higher perceptual quality, suggesting that task-driven optimization and visual quality are naturally aligned.

  </details>



- **MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale**  
  Bin Wang, Tianyao He, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Tao Chu, Yuan Qu, Zhenjiang Jin, Weijun Zeng, Ziyang Miao, et al.  
  _2026-04-06_ · https://arxiv.org/abs/2604.04771v1  
  <details><summary>Abstract</summary>

  Current document parsing methods compete primarily on model architecture innovation, while systematic engineering of training data remains underexplored. Yet SOTA models of different architectures and parameter scales exhibit highly consistent failure patterns on the same set of hard samples, suggesting that the performance bottleneck stems from shared deficiencies in training data rather than architecture itself. Building on this finding, we present \minerupro, which advances the state of the art solely through data engineering and training strategy optimization while keeping the 1.2B-parameter architecture of \mineru completely fixed. At its core is a Data Engine co-designed around coverage, informativeness, and annotation accuracy: Diversity-and-Difficulty-Aware Sampling expands training data from under 10M to 65.5M samples while correcting distribution shift; Cross-Model Consistency Verification leverages output agreement among heterogeneous models to assess sample difficulty and generate reliable annotations; the Judge-and-Refine pipeline improves annotation quality for hard samples through render-then-verify iterative correction. A three-stage progressive training strategy -- large-scale pre-training, hard sample fine-tuning, and GRPO alignment -- sequentially exploits these data at different quality tiers. On the evaluation front, we fix element-matching biases in OmniDocBench~v1.5 and introduce a Hard subset, establishing the more discriminative OmniDocBench~v1.6 protocol. Without any architectural modification, \minerupro achieves 95.69 on OmniDocBench~v1.6, improving over the same-architecture baseline by 2.71 points and surpassing all existing methods including models with over 200$\times$ more parameters.

  </details>



- **Explainable Machine Learning for Sepsis Outcome Prediction Using a Novel Romanian Electronic Health Record Dataset**  
  Andrei-Alexandru Bunea, Ovidiu Ghibea, Dan-Matei Popovici, Ion Daniel, Octavian Andronic  
  _2026-04-06_ · https://arxiv.org/abs/2604.04698v1  
  <details><summary>Abstract</summary>

  We develop and analyze explainable machine learning (ML) models for sepsis outcome prediction using a novel Electronic Health Record (EHR) dataset from 12,286 hospitalizations at a large emergency hospital in Romania. The dataset includes demographics, International Classification of Diseases (ICD-10) diagnostics, and 600 types of laboratory tests. This study aims to identify clinically strong predictors while achieving state-of-the-art results across three classification tasks: (1)deceased vs. discharged, (2)deceased vs. recovered, and (3)recovered vs. ameliorated. We trained five ML models to capture complex distributions while preserving clinical interpretability. Experiments explored the trade-off between feature richness and patient coverage, using subsets of the 10--50 most frequent laboratory tests. Model performance was evaluated using accuracy and area under the curve (AUC), and explainability was assessed using SHapley Additive exPlanations (SHAP). The highest performance was obtained for the deceased vs. recovered case study (AUC=0.983, accuracy=0.93). SHAP analysis identified several strong predictors such as cardiovascular comorbidities, urea levels, aspartate aminotransferase, platelet count, and eosinophil percentage. Eosinopenia emerged as a top predictor, highlighting its value as an underutilized marker that is not included in current assessment standards, while the high performance suggests the applicability of these models in clinical settings.

  </details>



- **Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity**  
  Jaeyoon Jung, Yejun Yoon, Kunwoo Park  
  _2026-04-06_ · https://arxiv.org/abs/2604.04692v1  
  <details><summary>Abstract</summary>

  Automated fact-checking is a crucial task not only in journalism but also across web platforms, where it supports a responsible information ecosystem and mitigates the harms of misinformation. While recent research has progressed from text-only to multimodal fact-checking, a prevailing assumption is that incorporating visual evidence universally improves performance. In this work, we challenge this assumption and show that indiscriminate use of multimodal evidence can reduce accuracy. To address this challenge, we propose AMuFC, a multimodal fact-checking framework that employs two collaborative agents with distinct roles for the adaptive use of visual evidence: An Analyzer determines whether visual evidence is necessary for claim verification, and a Verifier predicts claim veracity conditioned on both the retrieved evidence and the Analyzer's assessment. Experimental results on three datasets show that incorporating the Analyzer's assessment of visual evidence necessity into the Verifier's prediction yields substantial improvements in verification performance. In addition to all code, we release WebFC, a newly constructed dataset for evaluating fact-checking modules in a more realistic scenario, available at https://github.com/ssu-humane/AMuFC.

  </details>



- **Unsharp Measurement with Adaptive Gaussian POVMs for Quantum-Inspired Image Processing**  
  Debashis Saikia, Bikash K. Behera, Mayukha Pal, Prasanta K. Panigrahi  
  _2026-04-06_ · https://arxiv.org/abs/2604.04685v1  
  <details><summary>Abstract</summary>

  We propose a quantum measurement-based framework for probabilistic transformation of grayscale images using adaptive positive operator-valued measures (POVMs). In contrast, to existing approaches that are largely centered around segmentation or thresholding, the transformation is formulated here as a measurement-induced process acting directly on pixel intensities. The intensity values are embedded in a finite-dimensional Hilbert space, which allows the construction of data-adaptive measurement operators derived from Gaussian models of the image histogram. These operators naturally define an unsharp measurement of the intensity observable, with the reconstructed image obtained through expectation values of the measurement outcomes. To control the degree of measurement localization, we introduce a nonlinear sharpening transformation with a sharpening parameter, $γ$, that induces a continuous transition from unsharp measurements to projective measurements. This transition reflects an inherent trade-off between probabilistic smoothing and localization of intensity structures. In addition to the nonlinear sharpening parameter, we introduce another parameter $k$ (number of gaussian centers) which controls the resolution of the image during the transformation. Experimental results on standard benchmark images show that the proposed method gives effective data-adaptive transformations while preserving structural information.

  </details>



- **ZeD-MAP: Bundle Adjustment Guided Zero-Shot Depth Maps for Real-Time Aerial Imaging**  
  Selim Ahmet Iz, Francesco Nex, Norman Kerle, Henry Meissner, Ralf Berger  
  _2026-04-06_ · https://arxiv.org/abs/2604.04667v1  
  <details><summary>Abstract</summary>

  Real-time depth reconstruction from ultra-high-resolution UAV imagery is essential for time-critical geospatial tasks such as disaster response, yet remains challenging due to wide-baseline parallax, large image sizes, low-texture or specular surfaces, occlusions, and strict computational constraints. Recent zero-shot diffusion models offer fast per-image dense predictions without task-specific retraining, and require fewer labelled datasets than transformer-based predictors while avoiding the rigid capture geometry requirement of classical multi-view stereo. However, their probabilistic inference prevents reliable metric accuracy and temporal consistency across sequential frames and overlapping tiles. We present ZeD-MAP, a cluster-level framework that converts a test-time diffusion depth model into a metrically consistent, SLAM-like mapping pipeline by integrating incremental cluster-based bundle adjustment (BA). Streamed UAV frames are grouped into overlapping clusters; periodic BA produces metrically consistent poses and sparse 3D tie-points, which are reprojected into selected frames and used as metric guidance for diffusion-based depth estimation. Validation on ground-marker flights captured at approximately 50 m altitude (GSD is approximately 0.85 cm/px, corresponding to 2,650 square meters ground coverage per frame) with the DLR Modular Aerial Camera System (MACS) shows that our method achieves sub-meter accuracy, with approximately 0.87 m error in the horizontal (XY) plane and 0.12 m in the vertical (Z) direction, while maintaining per-image runtimes between 1.47 and 4.91 seconds. Results are subject to minor noise from manual point-cloud annotation. These findings show that BA-based metric guidance provides consistency comparable to classical photogrammetric methods while significantly accelerating processing, enabling real-time 3D map generation.

  </details>



- **Synthesis4AD: Synthetic Anomalies are All You Need for 3D Anomaly Detection**  
  Yihan Sun, Yuqi Cheng, Junjie Zu, Yuxiang Tan, Guoyang Xie, Yucheng Wang, Yunkang Cao, Weiming Shen  
  _2026-04-06_ · https://arxiv.org/abs/2604.04658v1  
  <details><summary>Abstract</summary>

  Industrial 3D anomaly detection performance is fundamentally constrained by the scarcity and long-tailed distribution of abnormal samples. To address this challenge, we propose Synthesis4AD, an end-to-end paradigm that leverages large-scale, high-fidelity synthetic anomalies to learn more discriminative representations for 3D anomaly detection. At the core of Synthesis4AD is 3D-DefectStudio, a software platform built upon the controllable synthesis engine MPAS, which injects geometrically realistic defects guided by higher-dimensional support primitives while simultaneously generating accurate point-wise anomaly masks. Furthermore, Synthesis4AD incorporates a multimodal large language model (MLLM) to interpret product design information and automatically translate it into executable anomaly synthesis instructions, enabling scalable and knowledge-driven anomalous data generation. To improve the robustness and generalization of the downstream detector on unstructured point clouds, Synthesis4AD further introduces a training pipeline based on spatial-distribution normalization and geometry-faithful data augmentations, which alleviates the sensitivity of Point Transformer architectures to absolute coordinates and improves feature learning under realistic data variations. Extensive experiments demonstrate state-of-the-art performance on Real3D-AD, MulSen-AD, and a real-world industrial parts dataset. The proposed synthesis method MPAS and the interactive system 3D-DefectStudio will be publicly released at https://github.com/hustCYQ/Synthesis4AD.

  </details>



- **Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale**  
  Zhengcen Li, Chenyang Jiang, Hang Zhao, Shiyang Zhou, Yunyang Mo, Feng Gao, Fan Yang, Qiben Shan, Shaocong Wu, Jingyong Su  
  _2026-04-06_ · https://arxiv.org/abs/2604.04634v1  
  <details><summary>Abstract</summary>

  The rapid advancement of video generation models has enabled the creation of highly realistic synthetic media, raising significant societal concerns regarding the spread of misinformation. However, current detection methods suffer from critical limitations. They rely on preprocessing operations like fixed-resolution resizing and cropping. These operations not only discard subtle, high-frequency forgery traces but also cause spatial distortion and significant information loss. Furthermore, existing methods are often trained and evaluated on outdated datasets that fail to capture the sophistication of modern generative models. To address these challenges, we introduce a comprehensive dataset and a novel detection framework. First, we curate a large-scale dataset of over 140K videos from 15 state-of-the-art open-source and commercial generators, along with Magic Videos benchmark designed specifically for evaluating ultra-realistic synthetic content. In addition, we propose a novel detection framework built on the Qwen2.5-VL Vision Transformer, which operates natively at variable spatial resolutions and temporal durations. This native-scale approach effectively preserves the high-frequency artifacts and spatiotemporal inconsistencies typically lost during conventional preprocessing. Extensive experiments demonstrate that our method achieves superior performance across multiple benchmarks, underscoring the critical importance of native-scale processing and establishing a robust new baseline for AI-generated video detection.

  </details>



- **InCTRLv2: Generalist Residual Models for Few-Shot Anomaly Detection and Segmentation**  
  Jiawen Zhu, Mengjia Niu, Guansong Pang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04632v1  
  <details><summary>Abstract</summary>

  While recent anomaly detection (AD) methods have made substantial progress in recognizing abnormal patterns within specific domains, most of them are specialist models that are trained on large training samples from a specific target dataset, struggling to generalize to unseen datasets. To address this limitation, the paradigm of Generalist Anomaly Detection (GAD) has emerged in recent years, aiming to learn a single generalist model to detect anomalies across diverse domains without retraining. To this end, this work introduces InCTRLv2, a novel few-shot Generalist Anomaly Detection and Segmentation (GADS) framework that significantly extends our previously proposed GAD model, InCTRL. Building on the idea of learning in-context residuals with few-shot normal examples to detect anomalies as in InCTRL, InCTRLv2 introduces two new, complementary perspectives of anomaly perception under a dual-branch framework. This is accomplished by two novel modules upon InCTRL: i) Discriminative Anomaly Score Learning (DASL) with both normal and abnormal data in the main branch, which learns a semantic-guided abnormality and normality space that supports the classification of query samples from both the abnormality and normality perspectives; and ii) One-class Anomaly Score Learning (OASL) using only the normal data, which learns generalized normality patterns in a semantic space via an auxiliary branch, focusing on detecting anomalies through the lens of normality solely. Both branches are guided by rich visual-text semantic priors encoded by large-scale vision-language models. Together, they offer a dual semantic perspective for AD: one emphasizes normal-abnormal discriminations, while the other emphasizes normality-deviated semantics. Extensive experiments on ten AD datasets demonstrate that InCTRLv2 achieves SotA performance in both anomaly detection and segmentation tasks across various settings.

  </details>



- **Temporal Inversion for Learning Interval Change in Chest X-Rays**  
  Hanbin Ko, Kyeongmin Jeon, Doowoong Choi, Chang Min Park  
  _2026-04-06_ · https://arxiv.org/abs/2604.04563v1  
  <details><summary>Abstract</summary>

  Recent advances in vision--language pretraining have enabled strong medical foundation models, yet most analyze radiographs in isolation, overlooking the key clinical task of comparing prior and current images to assess interval change. For chest radiographs (CXRs), capturing interval change is essential, as radiologists must evaluate not only the static appearance of findings but also how they evolve over time. We introduce TILA (Temporal Inversion-aware Learning and Alignment), a simple yet effective framework that uses temporal inversion, reversing image pairs, as a supervisory signal to enhance the sensitivity of existing temporal vision-language models to directional change. TILA integrates inversion-aware objectives across pretraining, fine-tuning, and inference, complementing conventional appearance modeling with explicit learning of temporal order. We also propose a unified evaluation protocol to assess order sensitivity and consistency under temporal inversion, and introduce MS-CXR-Tretrieval, a retrieval evaluation set constructed through a general protocol that can be applied to any temporal CXR dataset. Experiments on public datasets and real-world hospital cohorts demonstrate that TILA consistently improves progression classification and temporal embedding alignment when applied to multiple existing architectures.

  </details>



- **G-EDF-Loc: 3D Continuous Gaussian Distance Field for Robust Gradient-Based 6DoF Localization**  
  José E. Maese, Lucía Coto-Elena, Luis Merino, Fernando Caballero  
  _2026-04-06_ · https://arxiv.org/abs/2604.04525v1  
  <details><summary>Abstract</summary>

  This paper presents a robust 6-DoF localization framework based on a direct, CPU-based scan-to-map registration pipeline. The system leverages G-EDF, a novel continuous and memory-efficient 3D distance field representation. The approach models the Euclidean Distance Field (EDF) using a Block-Sparse Gaussian Mixture Model with adaptive spatial partitioning, ensuring $C^1$ continuity across block transitions and mitigating boundary artifacts. By leveraging the analytical gradients of this continuous map, which maintain Eikonal consistency, the proposed method achieves high-fidelity spatial reconstruction and real-time localization. Experimental results on large-scale datasets demonstrate that G-EDF-Loc performs competitively against state-of-the-art methods, exhibiting exceptional resilience even under severe odometry degradation or in the complete absence of IMU priors.

  </details>



- **Reproducibility study on how to find Spurious Correlations, Shortcut Learning, Clever Hans or Group-Distributional non-robustness and how to fix them**  
  Ole Delzer, Sidney Bender  
  _2026-04-06_ · https://arxiv.org/abs/2604.04518v1  
  <details><summary>Abstract</summary>

  Deep Neural Networks (DNNs) are increasingly utilized in high-stakes domains like medical diagnostics and autonomous driving where model reliability is critical. However, the research landscape for ensuring this reliability is terminologically fractured across communities that pursue the same goal of ensuring models rely on causally relevant features rather than confounding signals. While frameworks such as distributionally robust optimization (DRO), invariant risk minimization (IRM), shortcut learning, simplicity bias, and the Clever Hans effect all address model failure due to spurious correlations, researchers typically only reference work within their own domains. This reproducibility study unifies these perspectives through a comparative analysis of correction methods under challenging constraints like limited data availability and severe subgroup imbalance. We evaluate recently proposed correction methods based on explainable artificial intelligence (XAI) techniques alongside popular non-XAI baselines using both synthetic and real-world datasets. Findings show that XAI-based methods generally outperform non-XAI approaches, with Counterfactual Knowledge Distillation (CFKD) proving most consistently effective at improving generalization. Our experiments also reveal that the practical application of many methods is hindered by a dependency on group labels, as manual annotation is often infeasible and automated tools like Spectral Relevance Analysis (SpRAy) struggle with complex features and severe imbalance. Furthermore, the scarcity of minority group samples in validation sets renders model selection and hyperparameter tuning unreliable, posing a significant obstacle to the deployment of robust and trustworthy models in safety-critical areas.

  </details>



- **Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection**  
  Weihao Cao, Runqi Wang, Xiaoyue Duan, Jinchao Zhang, Ang Yang, Liping Jing  
  _2026-04-06_ · https://arxiv.org/abs/2604.04444v1  
  <details><summary>Abstract</summary>

  Open-vocabulary object detection (OVOD) enables models to detect any object category, including unseen ones. Benefiting from large-scale pre-training, existing OVOD methods achieve strong detection performance on general scenarios (e.g., OV-COCO) but suffer severe performance drops when transferred to downstream tasks with substantial domain shifts. This degradation stems from the scarcity and weak semantics of category labels in domain-specific task, as well as the inability of existing models to capture auxiliary semantics beyond coarse-grained category label. To address these issues, we propose HSA-DINO, a parameter-efficient semantic augmentation framework for enhancing open-vocabulary object detection. Specifically, we propose a multi-scale prompt bank that leverages image feature pyramids to capture hierarchical semantics and select domain-specific local semantic prompts, progressively enriching textual representations from coarse to fine-grained levels. Furthermore, we introduce a semantic-aware router that dynamically selects the appropriate semantic augmentation strategy during inference, thereby preventing parameter updates from degrading the generalization ability of the pre-trained OVOD model. We evaluate HSA-DINO on OV-COCO, several vertical domain datasets, and modified benchmark settings. The results show that HSA-DINO performs favorably against previous state-of-the-art methods, achieving a superior trade-off between domain adaptability and open-vocabulary generalization.

  </details>



- **Estimating Central, Peripheral, and Temporal Visual Contributions to Human Decision Making in Atari Games**  
  Henrik Krauss, Takehisa Yairi  
  _2026-04-06_ · https://arxiv.org/abs/2604.04439v1  
  <details><summary>Abstract</summary>

  We study how different visual information sources contribute to human decision making in dynamic visual environments. Using Atari-HEAD, a large-scale Atari gameplay dataset with synchronized eye-tracking, we introduce a controlled ablation framework as a means to reverse-engineer the contribution of peripheral visual information, explicit gaze information in form of gaze maps, and past-state information from human behavior. We train action-prediction networks under six settings that selectively include or exclude these information sources. Across 20 games, peripheral information shows by far the strongest contribution, with median prediction-accuracy drops in the range of 35.27-43.90% when removed. Gaze information yields smaller drops of 2.11-2.76%, while past-state information shows a broader range of 1.52-15.51%, with the upper end likely more informative due to reduced peripheral-information leakage. To complement aggregate accuracies, we cluster states by true-action probabilities assigned by the different model configurations. This analysis identifies coarse behavioral regimes, including focus-dominated, periphery-dominated, and more contextual decision situations. These results suggest that human decision making in Atari depends strongly on information beyond the current focus of gaze, while the proposed framework provides a way to estimate such information-source contributions from behavior.

  </details>



- **BoxComm: Benchmarking Category-Aware Commentary Generation and Narration Rhythm in Boxing**  
  Kaiwen Wang, Kaili Zheng, Rongrong Deng, Yiming Shi, Chenyi Guo, Ji Wu  
  _2026-04-06_ · https://arxiv.org/abs/2604.04419v1  
  <details><summary>Abstract</summary>

  Recent multimodal large language models (MLLMs) have shown strong capabilities in general video understanding, driving growing interest in automatic sports commentary generation. However, existing benchmarks for this task focus exclusively on team sports such as soccer and basketball, leaving combat sports entirely unexplored. Notably, combat sports present distinct challenges: critical actions unfold within milliseconds with visually subtle yet semantically decisive differences, and professional commentary contains a substantially higher proportion of tactical analysis compared to team sports. In this paper, we present BoxComm, a large-scale dataset comprising 445 World Boxing Championship match videos with over 52K commentary sentences from professional broadcasts. We propose a structured commentary taxonomy that categorizes each sentence into play-by-play, tactical, or contextual, providing the first category-level annotation for sports commentary benchmarks. Building on this taxonomy, we introduce two novel and complementary evaluations tailored to sports commentary generation: (1) category-conditioned generation, which evaluates whether models can produce accurate commentary of a specified type given video context; and (2) commentary rhythm assessment, which measures whether freely generated commentary exhibits appropriate temporal pacing and type distribution over continuous video segments, capturing a dimension of commentary competence that prior benchmarks have not addressed. Experiments on multiple state-of-the-art MLLMs reveal that current models struggle on both evaluations. We further propose EIC-Gen, an improved baseline incorporating detected punch events to supply structured action cues, yielding consistent gains and highlighting the importance of perceiving fleeting and subtle events for combat sports commentary.

  </details>



- **3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image**  
  Ze-Xin Yin, Liu Liu, Xinjie Wang, Wei Sui, Zhizhong Su, Jian Yang, Jin Xie  
  _2026-04-06_ · https://arxiv.org/abs/2604.04406v1  
  <details><summary>Abstract</summary>

  Compositional 3D scene generation from a single view requires the simultaneous recovery of scene layout and 3D assets. Existing approaches mainly fall into two categories: feed-forward generation methods and per-instance generation methods. The former directly predict 3D assets with explicit 6DoF poses through efficient network inference, but they generalize poorly to complex scenes. The latter improve generalization through a divide-and-conquer strategy, but suffer from time-consuming pose optimization. To bridge this gap, we introduce 3D-Fixer, a novel in-place completion paradigm. Specifically, 3D-Fixer extends 3D object generative priors to generate complete 3D assets conditioned on the partially visible point cloud at the original locations, which are cropped from the fragmented geometry obtained from the geometry estimation methods. Unlike prior works that require explicit pose alignment, 3D-Fixer uses fragmented geometry as a spatial anchor to preserve layout fidelity. At its core, we propose a coarse-to-fine generation scheme to resolve boundary ambiguity under occlusion, supported by a dual-branch conditioning network and an Occlusion-Robust Feature Alignment (ORFA) strategy for stable training. Furthermore, to address the data scarcity bottleneck, we present ARSG-110K, the largest scene-level dataset to date, comprising over 110K diverse scenes and 3M annotated images with high-fidelity 3D ground truth. Extensive experiments show that 3D-Fixer achieves state-of-the-art geometric accuracy, which significantly outperforms baselines such as MIDI and Gen3DSR, while maintaining the efficiency of the diffusion process. Code and data will be publicly available at https://zx-yin.github.io/3dfixer.

  </details>



- **UENR-600K: A Large-Scale Physically Grounded Dataset for Nighttime Video Deraining**  
  Pei Yang, Hai Ci, Beibei Lin, Yiren Song, Mike Zheng Shou  
  _2026-04-06_ · https://arxiv.org/abs/2604.04402v1  
  <details><summary>Abstract</summary>

  Nighttime video deraining is uniquely challenging because raindrops interact with artificial lighting. Unlike daytime white rain, nighttime rain takes on various colors and appears locally illuminated. Existing small-scale synthetic datasets rely on 2D rain overlays and fail to capture these physical properties, causing models to generalize poorly to real-world night rain. Meanwhile, capturing real paired nighttime videos remains impractical because rain effects cannot be isolated from other degradations like sensor noise. To bridge this gap, we introduce UENR-600K, a large-scale, physically grounded dataset containing 600,000 1080p frame pairs. We utilize Unreal Engine to simulate rain as 3D particles within virtual environments. This approach guarantees photorealism and physically real raindrops, capturing correct details like color refractions, scene occlusions, rain curtains. Leveraging this high-quality data, we establish a new state-of-the-art baseline by adapting the Wan 2.2 video generation model. Our baseline treat deraining as a video-to-video generation task, exploiting strong generative priors to almost entirely bridge the sim-to-real gap. Extensive benchmarking demonstrates that models trained on our dataset generalize significantly better to real-world videos. Project page: https://showlab.github.io/UENR-600K/.

  </details>



- **BiTDiff: Fine-Grained 3D Conducting Motion Generation via BiMamba-Transformer Diffusion**  
  Tianzhi Jia, Kaixing Yang, Xiaole Yang, Xulong Tang, Ke Qiu, Shikui Wei, Yao Zhao  
  _2026-04-06_ · https://arxiv.org/abs/2604.04395v1  
  <details><summary>Abstract</summary>

  3D conducting motion generation aims to synthesize fine-grained conductor motions from music, with broad potential in music education, virtual performance, digital human animation, and human-AI co-creation. However, this task remains underexplored due to two major challenges: (1) the lack of large-scale fine-grained 3D conducting datasets and (2) the absence of effective methods that can jointly support long-sequence generation with high quality and efficiency. To address the data limitation, we develop a quality-oriented 3D conducting motion collection pipeline and construct CM-Data, a fine-grained SMPL-X dataset with about 10 hours of conducting motion data. To the best of our knowledge, CM-Data is the first and largest public dataset for 3D conducting motion generation. To address the methodological limitation, we propose BiTDiff, a novel framework for 3D conducting motion generation, built upon a BiMamba-Transformer hybrid model architecture for efficient long-sequence modeling and a Diffusion-based generative strategy with human-kinematic decomposition for high-quality motion synthesis. Specifically, BiTDiff introduces auxiliary physical-consistency losses and a hand-/body-specific forward-kinematics design for better fine-grained motion modeling, while leveraging BiMamba for memory-efficient long-sequence temporal modeling and Transformer for cross-modal semantic alignment. In addition, BiTDiff supports training-free joint-level motion editing, enabling downstream human-AI interaction design. Extensive quantitative and qualitative experiments demonstrate that BiTDiff achieves state-of-the-art (SOTA) performance for 3D conducting motion generation on the CM-Data dataset. Code will be available upon acceptance.

  </details>



- **Spatially-Weighted CLIP for Street-View Geo-localization**  
  Ting Han, Fengjiao Li, Chunsong Chen, Haoling Huang, Yiping Chen, Meiliu Wu  
  _2026-04-06_ · https://arxiv.org/abs/2604.04357v1  
  <details><summary>Abstract</summary>

  This paper proposes Spatially-Weighted CLIP (SW-CLIP), a novel framework for street-view geo-localization that explicitly incorporates spatial autocorrelation into vision-language contrastive learning. Unlike conventional CLIP-based methods that treat all non-matching samples as equally negative, SW-CLIP leverages Tobler's First Law of Geography to model geographic relationships through distance-aware soft supervision. Specifically, we introduce a location-as-text representation to encode geographic positions and replace one-hot InfoNCE targets with spatially weighted soft labels derived from geodesic distance. Additionally, a neighborhood-consistency regularization is employed to preserve local spatial structure in the embedding space. Experiments on a multi-city dataset demonstrate that SW-CLIP significantly improves geo-localization accuracy, reduces long-tail errors, and enhances spatial coherence compared to standard CLIP. The results highlight the importance of shifting from semantic alignment to geographic alignment for robust geo-localization and provide a general paradigm for integrating spatial principles into multimodal representation learning.

  </details>



- **OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text**  
  Weiguo Pian, Saksham Singh Kushwaha, Zhimin Chen, Shijian Deng, Kai Wang, Yunhui Guo, Yapeng Tian  
  _2026-04-06_ · https://arxiv.org/abs/2604.04348v1  
  <details><summary>Abstract</summary>

  In this paper, we propose Universal Holistic Audio Generation (UniHAGen), a task for synthesizing comprehensive auditory scenes that include both on-screen and off-screen sounds across diverse domains (e.g., ambient events, musical instruments, and human speech). Prior video-conditioned audio generation models typically focus on producing on-screen environmental sounds that correspond to visible sounding events, neglecting off-screen auditory events. While recent holistic joint text-video-to-audio generation models aim to produce auditory scenes with both on- and off-screen sound but they are limited to non-speech sounds, lacking the ability to generate or integrate human speech. To overcome these limitations, we introduce OmniSonic, a flow-matching-based diffusion framework jointly conditioned on video and text. It features a TriAttn-DiT architecture that performs three cross-attention operations to process on-screen environmental sound, off-screen environmental sound, and speech conditions simultaneously, with a Mixture-of-Experts (MoE) gating mechanism that adaptively balances their contributions during generation. Furthermore, we construct UniHAGen-Bench, a new benchmark with over one thousand samples covering three representative on/off-screen speech-environment scenarios. Extensive experiments show that OmniSonic consistently outperforms state-of-the-art approaches on both objective metrics and human evaluations, establishing a strong baseline for universal and holistic audio generation. Project page: https://weiguopian.github.io/OmniSonic_webpage/

  </details>



- **GA-GS: Generation-Assisted Gaussian Splatting for Static Scene Reconstruction**  
  Yedong Shen, Shiqi Zhang, Sha Zhang, Yifan Duan, Xinran Zhang, Wenhao Yu, Lu Zhang, Jiajun Deng, Yanyong Zhang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04331v1  
  <details><summary>Abstract</summary>

  Reconstructing static 3D scene from monocular video with dynamic objects is important for numerous applications such as virtual reality and autonomous driving. Current approaches typically rely on background for static scene reconstruction, limiting the ability to recover regions occluded by dynamic objects. In this paper, we propose GA-GS, a Generation-Assisted Gaussian Splatting method for Static Scene Reconstruction. The key innovation of our work lies in leveraging generation to assist in reconstructing occluded regions. We employ a motion-aware module to segment and remove dynamic regions, and thenuse a diffusion model to inpaint the occluded areas, providing pseudo-ground-truth supervision. To balance contributions from real background and generated region, we introduce a learnable authenticity scalar for each Gaussian primitive, which dynamically modulates opacity during splatting for authenticity-aware rendering and supervision. Since no existing dataset provides ground-truth static scene of video with dynamic objects, we construct a dataset named Trajectory-Match, using a fixed-path robot to record each scene with/without dynamic objects, enabling quantitative evaluation in reconstruction of occluded regions. Extensive experiments on both the DAVIS and our dataset show that GA-GS achieves state-of-the-art performance in static scene reconstruction, especially in challenging scenarios with large-scale, persistent occlusions.

  </details>



- **HighFM: Towards a Foundation Model for Learning Representations from High-Frequency Earth Observation Data**  
  Stella Girtsou, Konstantinos Alexis, Giorgos Giannopoulos, Harris Kontoes  
  _2026-04-05_ · https://arxiv.org/abs/2604.04306v1  
  <details><summary>Abstract</summary>

  The increasing frequency and severity of climate related disasters have intensified the need for real time monitoring, early warning, and informed decision-making. Earth Observation (EO), powered by satellite data and Machine Learning (ML), offers powerful tools to meet these challenges. Foundation Models (FMs) have revolutionized EO ML by enabling general-purpose pretraining on large scale remote sensing datasets. However most existing models rely on high-resolution satellite imagery with low revisit rates limiting their suitability for fast-evolving phenomena and time critical emergency response. In this work, we present HighFM, a first cut approach towards a FM for high temporal resolution, multispectral EO data. Leveraging over 2 TB of SEVIRI imagery from the Meteosat Second Generation (MSG) platform, we adapt the SatMAE masked autoencoding framework to learn robust spatiotemporal representations. To support real time monitoring, we enhance the original architecture with fine grained temporal encodings to capture short term variability. The pretrained models are then finetuned on cloud masking and active fire detection tasks. We benchmark our SEVIRI pretrained Vision Transformers against traditional baselines and recent geospatial FMs, demonstrating consistent gains across both balanced accuracy and IoU metrics. Our results highlight the potential of temporally dense geostationary data for real-time EO, offering a scalable path toward foundation models for disaster detection and tracking.

  </details>



- **Precise Robot Command Understanding Using Grammar-Constrained Large Language Models**  
  Xinyun Huo, Raghav Gnanasambandam, Xinyao Zhang  
  _2026-04-05_ · https://arxiv.org/abs/2604.04233v1  
  <details><summary>Abstract</summary>

  Human-robot collaboration in industrial settings requires precise and reliable communication to enhance operational efficiency. While Large Language Models (LLMs) understand general language, they often lack the domain-specific rigidity needed for safe and executable industrial commands. To address this gap, this paper introduces a novel grammar-constrained LLM that integrates a grammar-driven Natural Language Understanding (NLU) system with a fine-tuned LLM, which enables both conversational flexibility and the deterministic precision required in robotics. Our method employs a two-stage process. First, a fine-tuned LLM performs high-level contextual reasoning and parameter inference on natural language inputs. Second, a Structured Language Model (SLM) and a grammar-based canonicalizer constrain the LLM's output, forcing it into a standardized symbolic format composed of valid action frames and command elements. This process guarantees that generated commands are valid and structured in a robot-readable JSON format. A key feature of the proposed model is a validation and feedback loop. A grammar parser validates the output against a predefined list of executable robotic actions. If a command is invalid, the system automatically generates corrective prompts and re-engages the LLM. This iterative self-correction mechanism allows the model to recover from initial interpretation errors to improve system robustness. We evaluate our grammar-constrained hybrid model against two baselines: a fine-tuned API-based LLM and a standalone grammar-driven NLU model. Using the Human Robot Interaction Corpus (HuRIC) dataset, we demonstrate that the hybrid approach achieves superior command validity, which promotes safer and more effective industrial human-robot collaboration.

  </details>



- **Learning from Imperfect Demonstrations via Temporal Behavior Tree-Guided Trajectory Repair**  
  Aniruddh G. Puranic, Sebastian Schirmer, John S. Baras, Calin Belta  
  _2026-04-05_ · https://arxiv.org/abs/2604.04225v1  
  <details><summary>Abstract</summary>

  Learning robot control policies from demonstrations is a powerful paradigm, yet real-world data is often suboptimal, noisy, or otherwise imperfect, posing significant challenges for imitation and reinforcement learning. In this work, we present a formal framework that leverages Temporal Behavior Trees (TBT), an extension of Signal Temporal Logic (STL) with Behavior Tree semantics, to repair suboptimal trajectories prior to their use in downstream policy learning. Given demonstrations that violate a TBT specification, a model-based repair algorithm corrects trajectory segments to satisfy the formal constraints, yielding a dataset that is both logically consistent and interpretable. The repaired trajectories are then used to extract potential functions that shape the reward signal for reinforcement learning, guiding the agent toward task-consistent regions of the state space without requiring knowledge of the agent's kinematic model. We demonstrate the effectiveness of this framework on discrete grid-world navigation and continuous single and multi-agent reach-avoid tasks, highlighting its potential for data-efficient robot learning in settings where high-quality demonstrations cannot be assumed.

  </details>



- **Graphic-Design-Bench: A Comprehensive Benchmark for Evaluating AI on Graphic Design Tasks**  
  Adrienne Deganutti, Elad Hirsch, Haonan Zhu, Jaejung Seol, Purvanshi Mehta  
  _2026-04-05_ · https://arxiv.org/abs/2604.04192v1  
  <details><summary>Abstract</summary>

  We introduce GraphicDesignBench (GDB), the first comprehensive benchmark suite designed specifically to evaluate AI models on the full breadth of professional graphic design tasks. Unlike existing benchmarks that focus on natural-image understanding or generic text-to-image synthesis, GDB targets the unique challenges of professional design work: translating communicative intent into structured layouts, rendering typographically faithful text, manipulating layered compositions, producing valid vector graphics, and reasoning about animation. The suite comprises 50 tasks organized along five axes: layout, typography, infographics, template & design semantics and animation, each evaluated under both understanding and generation settings, and grounded in real-world design templates drawn from the LICA layered-composition dataset. We evaluate a set of frontier closed-source models using a standardized metric taxonomy covering spatial accuracy, perceptual quality, text fidelity, semantic alignment, and structural validity. Our results reveal that current models fall short on the core challenges of professional design: spatial reasoning over complex layouts, faithful vector code generation, fine-grained typographic perception, and temporal decomposition of animations remain largely unsolved. While high-level semantic understanding is within reach, the gap widens sharply as tasks demand precision, structure, and compositional awareness. GDB provides a rigorous, reproducible testbed for tracking progress toward AI systems that can function as capable design collaborators. The full evaluation framework is publicly available.

  </details>



- **Scale-Aware Vision-Language Adaptation for Extreme Far-Distance Video Person Re-identification**  
  Ashwat Rajbhandari, Bharatesh Chakravarthi  
  _2026-04-05_ · https://arxiv.org/abs/2604.04183v1  
  <details><summary>Abstract</summary>

  Extreme far-distance video person re-identification (ReID) is particularly challenging due to scale compression, resolution degradation, motion blur, and aerial-ground viewpoint mismatch. As camera altitude and subject distance increase, models trained on close-range imagery degrade significantly. In this work, we investigate how large-scale vision-language models can be adapted to operate reliably under these conditions. Starting from a CLIP-based baseline, we upgrade the visual backbone from ViT-B/16 to ViT-L/14 and introduce backbone-aware selective fine-tuning to stabilize adaptation of the larger transformer. To address noisy and low-resolution tracklets, we incorporate a lightweight temporal attention pooling mechanism that suppresses degraded frames and emphasizes informative observations. We retain adapter-based and prompt-conditioned cross-view learning to mitigate aerial-ground domain shifts, and further refine retrieval using improved optimization and k-reciprocal re-ranking. Experiments on the DetReIDX stress-test benchmark show that our approach achieves mAP scores of 46.69 (A2G), 41.23 (G2A), and 22.98 (A2A), corresponding to an overall mAP of 35.73. These results show that large-scale vision-language backbones, when combined with stability-focused adaptation, significantly enhance robustness in extreme far-distance video person ReID.

  </details>



- **GENFIG1: Visual Summaries of Scholarly Work as a Challenge for Vision-Language Models**  
  Yaohan Guan, Pristina Wang, Najim Dehak, Alan Yuille, Jieneng Chen, Daniel Khashabi  
  _2026-04-05_ · https://arxiv.org/abs/2604.04172v1  
  <details><summary>Abstract</summary>

  In many science papers, "Figure 1" serves as the primary visual summary of the core research idea. These figures are visually simple yet conceptually rich, often requiring significant effort and iteration by human authors to get right, highlighting the difficulty of science visual communication. With this intuition, we introduce GENFIG1, a benchmark for generative AI models (e.g., Vision-Language Models). GENFIG1 evaluates models for their ability to produce figures that clearly express and motivate the central idea of a paper (title, abstract, introduction, and figure caption) as input. Solving GENFIG1 requires more than producing visually appealing graphics: the task entails reasoning for text-to-image generation that couples scientific understanding with visual synthesis. Specifically, models must (i) comprehend and grasp the technical concepts of the paper, (ii) identify the most salient ones, and (iii) design a coherent and aesthetically effective graphic that conveys those concepts visually and is faithful to the input. We curate the benchmark from papers published at top deep-learning conferences, apply stringent quality control, and introduce an automatic evaluation metric that correlates well with expert human judgments. We evaluate a suite of representative models on GENFIG1 and demonstrate that the task presents significant challenges, even for the best-performing systems. We hope this benchmark serves as a foundation for future progress in multimodal AI.

  </details>



- **Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation**  
  Xu Yan, Jun Yin, Shiliang Sun, Minghua Wan  
  _2026-04-05_ · https://arxiv.org/abs/2604.04170v1  
  <details><summary>Abstract</summary>

  Although multi-view multi-label learning has been extensively studied, research on the dual-missing scenario, where both views and labels are incomplete, remains largely unexplored. Existing methods mainly rely on contrastive learning or information bottleneck theory to learn consistent representations under missing-view conditions, but loss-based alignment without explicit structural constraints limits the ability to capture stable and discriminative shared semantics. To address this issue, we introduce a more structured mechanism for consistent representation learning: we learn discrete consistent representations through a multi-view shared codebook and cross-view reconstruction, which naturally align different views within the limited shared codebook embeddings and reduce feature redundancy. At the decision level, we design a weight estimation method that evaluates the ability of each view to preserve label correlation structures, assigning weights accordingly to enhance the quality of the fused prediction. In addition, we introduce a fused-teacher self-distillation framework, where the fused prediction guides the training of view-specific classifiers and feeds the global knowledge back into the single-view branches, thereby enhancing the generalization ability of the model under missing-label conditions. The effectiveness of our proposed method is thoroughly demonstrated through extensive comparative experiments with advanced methods on five benchmark datasets. Code is available at https://github.com/xuy11/SCSD.

  </details>



- **Hierarchical Co-Embedding of Font Shapes and Impression Tags**  
  Yugo Kubota, Kaito Shiku, Seiichi Uchida  
  _2026-04-05_ · https://arxiv.org/abs/2604.04158v1  
  <details><summary>Abstract</summary>

  Font shapes can evoke a wide range of impressions, but the correspondence between fonts and impression descriptions is not one-to-one: some impressions are broadly compatible with diverse styles, whereas others strongly constrain the set of plausible fonts. We refer to this graded constraint strength as style specificity. In this paper, we propose a hyperbolic co-embedding framework that models font--impression correspondence through entailment rather than simple paired alignment. Font images and impression descriptions, represented as single tags or tag sets, are embedded in a shared hyperbolic space with two complementary entailment constraints: impression-to-font entailment and low-to-high style-specificity entailment among impressions. This formulation induces a radial structure in which low style-specificity impressions lie near the origin and high style-specificity impressions lie farther away, yielding an interpretable geometric measure of how strongly an impression constrains font style. Experiments on the MyFonts dataset demonstrate improved bidirectional retrieval over strong one-to-one baselines. In addition, traversal and tag-level analyses show that the learned space captures a coherent progression from ambiguous to more style-specific impressions and provides a meaningful, data-driven quantification of style specificity.

  </details>



- **NTIRE 2026 3D Restoration and Reconstruction in Real-world Adverse Conditions: RealX3D Challenge Results**  
  Shuhong Liu, Chenyu Bao, Ziteng Cui, Xuangeng Chu, Bin Ren, Lin Gu, Xiang Chen, Mingrui Li, Long Ma, Marcos V. Conde, et al.  
  _2026-04-05_ · https://arxiv.org/abs/2604.04135v1  
  <details><summary>Abstract</summary>

  This paper presents a comprehensive review of the NTIRE 2026 3D Restoration and Reconstruction (3DRR) Challenge, detailing the proposed methods and results. The challenge seeks to identify robust reconstruction pipelines that are robust under real-world adverse conditions, specifically extreme low-light and smoke-degraded environments, as captured by our RealX3D benchmark. A total of 279 participants registered for the competition, of whom 33 teams submitted valid results. We thoroughly evaluate the submitted approaches against state-of-the-art baselines, revealing significant progress in 3D reconstruction under adverse conditions. Our analysis highlights shared design principles among top-performing methods and provides insights into effective strategies for handling 3D scene degradation.

  </details>



- **SARES-DEIM: Sparse Mixture-of-Experts Meets DETR for Robust SAR Ship Detection**  
  Fenghao Song, Shaojing Yang, Xi Zhou  
  _2026-04-05_ · https://arxiv.org/abs/2604.04127v1  
  <details><summary>Abstract</summary>

  Ship detection in Synthetic Aperture Radar (SAR) imagery is fundamentally challenged by inherent coherent speckle noise, complex coastal clutter, and the prevalence of small-scale targets. Conventional detectors, primarily designed for optical imagery, often exhibit limited robustness against SAR-specific degradation and suffer from the loss of fine-grained ship signatures during spatial downsampling. To address these limitations, we propose SARES-DEIM, a domain-aware detection framework grounded in the DEtection TRansformer (DETR) paradigm. Central to our approach is SARESMoE (SAR-aware Expert Selection Mixture-of-Experts), a module leveraging a sparse gating mechanism to selectively route features toward specialized frequency and wavelet experts. This sparsely-activated architecture effectively filters speckle noise and semantic clutter while maintaining high computational efficiency. Furthermore, we introduce the Space-to-Depth Enhancement Pyramid (SDEP) neck to preserve high-resolution spatial cues from shallow stages, significantly improving the localization of small targets. Extensive experiments on two benchmark datasets demonstrate the superiority of SARES-DEIM. Notably, on the challenging HRSID dataset, our model achieves a mAP50:95 of 76.4% and a mAP50 of 93.8%, outperforming state-of-the-art YOLO-series and specialized SAR detectors.

  </details>



- **Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware**  
  Arunkumar Rathinam, Jules Lecomte, Jost Reelsen, Gregor Lenz, Axel von Arnim, Djamila Aouada  
  _2026-04-05_ · https://arxiv.org/abs/2604.04117v1  
  <details><summary>Abstract</summary>

  Reliable relative pose estimation is a key enabler for autonomous rendezvous and proximity operations, yet space imagery is notoriously challenging due to extreme illumination, high contrast, and fast target motion. Event cameras provide asynchronous, change-driven measurements that can remain informative when frame-based imagery saturates or blurs, while neuromorphic processors can exploit sparse activations for low-latency, energy-efficient inferences. This paper presents a spacecraft 6-DoF pose-estimation pipeline that couples event-based vision with the BrainChip Akida neuromorphic processor. Using the SPADES dataset, we train compact MobileNet-style keypoint regression networks on lightweight event-frame representations, apply quantization-aware training (8/4-bit), and convert the models to Akida-compatible spiking neural networks. We benchmark three event representations and demonstrate real-time, low-power inference on Akida V1 hardware. We additionally design a heatmap-based model targeting Akida V2 and evaluate it on Akida Cloud, yielding improved pose accuracy. To our knowledge, this is the first end-to-end demonstration of spacecraft pose estimation running on Akida hardware, highlighting a practical route to low-latency, low-power perception for future autonomous space missions.

  </details>



- **A Physics-Informed, Behavior-Aware Digital Twin for Robust Multimodal Forecasting of Core Body Temperature in Precision Livestock Farming**  
  Riasad Alvi, Mohaimenul Azam Khan Raiaan, Sadia Sultana Chowa, Arefin Ittesafun Abian, Reem E Mohamed, Md Rafiqul Islam, Yakub Sebastian, Sheikh Izzal Azid, Sami Azam  
  _2026-04-05_ · https://arxiv.org/abs/2604.04098v1  
  <details><summary>Abstract</summary>

  Precision livestock farming requires accurate and timely heat stress prediction to ensure animal welfare and optimize farm management. This study presents a physics-informed digital twin (DT) framework combined with an uncertainty-aware, expert-weighted stacked ensemble for multimodal forecasting of Core Body Temperature (CBT) in dairy cattle. Using the high-frequency, heterogeneous MmCows dataset, the DT integrates an ordinary differential equation (ODE)-based thermoregulation model that simulates metabolic heat production and dissipation, a Gaussian process for capturing cow-specific deviations, a Kalman filter for aligning predictions with real-time sensor data, and a behavioral Markov chain that models activity-state transitions under varying environmental conditions. The DT outputs key physiological indicators, such as predicted CBT, heat stress probability, and behavioral state distributions are fused with raw sensor data and enriched through multi-scale temporal analysis and cross-modal feature engineering to form a comprehensive feature set. The predictive methodology is designed in a three-stage stacked ensemble, where stage 1 trains modality-specific LightGBM 'expert' models on distinct feature groups, stage 2 collects their predictions as meta-features, and at stage 3 Optuna-tuned LightGBM meta-model yields the final CBT forecast. Predictive uncertainty is quantified via bootstrapping and validated using Prediction Interval Coverage Probability (PICP). Ablation analysis confirms that incorporating DT-derived features and multimodal fusion substantially enhances performance. The proposed framework achieves a cross-validated R2 of 0.783, F1 score of 84.25% and PICP of 92.38% for 2-hour ahead forecasting, providing a robust, uncertainty-aware, and physically principled system for early heat stress detection and precision livestock management.

  </details>



- **TORA: Topological Representation Alignment for 3D Shape Assembly**  
  Nahyuk Lee, Zhiang Chen, Marc Pollefeys, Sunghwan Hong  
  _2026-04-05_ · https://arxiv.org/abs/2604.04050v1  
  <details><summary>Abstract</summary>

  Flow-matching methods for 3D shape assembly learn point-wise velocity fields that transport parts toward assembled configurations, yet they receive no explicit guidance about which cross-part interactions should drive the motion. We introduce TORA, a topology-first representation alignment framework that distills relational structure from a frozen pretrained 3D encoder into the flow-matching backbone during training. We first realize this via simple instantiation, token-wise cosine matching, which injects the learned geometric descriptors from the teacher representation. We then extend to employ a Centered Kernel Alignment (CKA) loss to match the similarity structure between student and teacher representations for enhanced topological alignment. Through systematic probing of diverse 3D encoders, we show that geometry- and contact-centric teacher properties, not semantic classification ability, govern alignment effectiveness, and that alignment is most beneficial at later transformer layers where spatial structure naturally emerges. TORA introduces zero inference overhead while yielding two consistent benefits: faster convergence (up to 6.9$\times$) and improved accuracy in-distribution, along with greater robustness under domain shift. Experiments on five benchmarks spanning geometric, semantic, and inter-object assembly demonstrate state-of-the-art performance, with particularly pronounced gains in zero-shot transfer to unseen real-world and synthetic datasets. Project page: https://nahyuklee.github.io/tora.

  </details>



- **High-Fidelity Mural Restoration via a Unified Hybrid Mask-Aware Transformer**  
  Jincheng Jiang, Qianhao Han, Chi Zhang, Zheng Zheng  
  _2026-04-05_ · https://arxiv.org/abs/2604.03984v1  
  <details><summary>Abstract</summary>

  Ancient murals are valuable cultural artifacts, but many have suffered severe degradation due to environmental exposure, material aging, and human activity. Restoring these artworks is challenging because it requires both reconstructing large missing structures and strictly preserving authentic, undamaged regions. This paper presents the Hybrid Mask-Aware Transformer (HMAT), a unified framework for high-fidelity mural restoration. HMAT integrates Mask-Aware Dynamic Filtering for robust local texture modeling with a Transformer bottleneck for long-range structural inference. To further address the diverse morphology of degradation, we introduce a mask-conditional style fusion module that dynamically guides the generative process. In addition, a Teacher-Forcing Decoder with hard-gated skip connections is designed to enforce fidelity in valid regions and focus reconstruction on missing areas. We evaluate HMAT on the DHMural dataset and a curated Nine-Colored Deer dataset under varying degradation levels. Experimental results demonstrate that the proposed method achieves competitive performance compared to state-of-the-art approaches, while producing more structurally coherent and visually faithful restorations. These findings suggest that HMAT provides an effective solution for the digital restoration of cultural heritage murals.

  </details>



- **Beyond Task-Driven Features for Object Detection**  
  Meilun Zhou, Alina Zare  
  _2026-04-04_ · https://arxiv.org/abs/2604.03839v1  
  <details><summary>Abstract</summary>

  Task-driven features learned by modern object detectors optimize end task loss yet often capture shortcut correlations that fail to reflect underlying annotation structure. Such representations limit transfer, interpretability, and robustness when task definitions change or supervision becomes sparse. This paper introduces an annotation-guided feature augmentation framework that injects embeddings into an object detection backbone. The method constructs dense spatial feature grids from annotation-guided latent spaces and fuses them with feature pyramid representations to influence region proposal and detection heads. Experiments across wildlife and remote sensing datasets evaluate classification, localization, and data efficiency under multiple supervision regimes. Results show consistent improvements in object focus, reduced background sensitivity, and stronger generalization to unseen or weakly supervised tasks. The findings demonstrate that aligning features with annotation geometry yields more meaningful representations than purely task optimized features.

  </details>



- **Task-Guided Multi-Annotation Triplet Learning for Remote Sensing Representations**  
  Meilun Zhou, Alina Zare  
  _2026-04-04_ · https://arxiv.org/abs/2604.03837v1  
  <details><summary>Abstract</summary>

  Prior multi-task triplet loss methods relied on static weights to balance supervision between various types of annotation. However, static weighting requires tuning and does not account for how tasks interact when shaping a shared representation. To address this, the proposed task-guided multi-annotation triplet loss removes this dependency by selecting triplets through a mutual-information criteria that identifies triplets most informative across tasks. This strategy modifies which samples influence the representation rather than adjusting loss magnitudes. Experiments on an aerial wildlife dataset compare the proposed task-guided selection against several triplet loss setups for shaping a representation in an effective multi-task manner. The results show improved classification and regression performance and demonstrate that task-aware triplet selection produces a more effective shared representation for downstream tasks.

  </details>



- **SPARK-IL: Spectral Retrieval-Augmented RAG for Knowledge-driven Deepfake Detection via Incremental Learning**  
  Hessen Bougueffa Eutamene, Abdellah Zakaria Sellam, Abdelmalik Taleb-Ahmed, Abdenour Hadid  
  _2026-04-04_ · https://arxiv.org/abs/2604.03833v1  
  <details><summary>Abstract</summary>

  Detecting AI-generated images remains a significant challenge because detectors trained on specific generators often fail to generalize to unseen models; however, while pixel-level artifacts vary across models, frequency-domain signatures exhibit greater consistency, providing a promising foundation for cross-generator detection. To address this, we propose SPARK-IL, a retrieval-augmented framework that combines dual-path spectral analysis with incremental learning by utilizing a partially frozen ViT-L/14 encoder for semantic representations alongside a parallel path for raw RGB pixel embeddings. Both paths undergo multi-band Fourier decomposition into four frequency bands, which are individually processed by Kolmogorov-Arnold Networks (KAN) with mixture-of-experts for band-specific transformations before the resulting spectral embeddings are fused via cross-attention with residual connections. During inference, this fused embedding retrieves the $k$ nearest labeled signatures from a Milvus database using cosine similarity to facilitate predictions via majority voting, while an incremental learning strategy expands the database and employs elastic weight consolidation to preserve previously learned transformations. Evaluated on the UniversalFakeDetect benchmark across 19 generative models -- including GANs, face-swapping, and diffusion methods -- SPARK-IL achieves a 94.6\% mean accuracy, with the code to be publicly released at https://github.com/HessenUPHF/SPARK-IL.

  </details>


