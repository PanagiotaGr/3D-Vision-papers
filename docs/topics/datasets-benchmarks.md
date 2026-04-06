# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-04-06 08:00 UTC_

Total papers shown: **23**


---

- **HyperCT: Low-Rank Hypernet for Unified Chest CT Analysis**  
  Fengbei Liu, Sunwoo Kwak, Hao Phung, Nusrat Binta Nizam, Ilan Richter, Nir Uriel, Hadar Averbuch-Elor, Daborah Estrin, Mert R. Sabuncu  
  _2026-04-03_ · https://arxiv.org/abs/2604.03224v1  
  <details><summary>Abstract</summary>

  Non-contrast chest CTs offer a rich opportunity for both conventional pulmonary and opportunistic extra-pulmonary screening. While Multi-Task Learning (MTL) can unify these diverse tasks, standard hard-parameter sharing approaches are often suboptimal for modeling distinct pathologies. We propose HyperCT, a framework that dynamically adapts a Vision Transformer backbone via a Hypernetwork. To ensure computational efficiency, we integrate Low-Rank Adaptation (LoRA), allowing the model to regress task-specific low-rank weight updates rather than full parameters. Validated on a large-scale dataset of radiological and cardiological tasks, \method{} outperforms various strong baselines, offering a unified, parameter-efficient solution for holistic patient assessment. Our code is available at https://github.com/lfb-1/HyperCT.

  </details>



- **The Eleventh NTIRE 2026 Efficient Super-Resolution Challenge Report**  
  Bin Ren, Hang Guo, Yan Shu, Jiaqi Ma, Ziteng Cui, Shuhong Liu, Guofeng Mei, Lei Sun, Zongwei Wu, Fahad Shahbaz Khan, et al.  
  _2026-04-03_ · https://arxiv.org/abs/2604.03198v1  
  <details><summary>Abstract</summary>

  This paper reviews the NTIRE 2026 challenge on efficient single-image super-resolution with a focus on the proposed solutions and results. The aim of this challenge is to devise a network that reduces one or several aspects, such as runtime, parameters, and FLOPs, while maintaining PSNR of around 26.90 dB on the DIV2K_LSDIR_valid dataset, and 26.99 dB on the DIV2K_LSDIR_test dataset. The challenge had 95 registered participants, and 15 teams made valid submissions. They gauge the state-of-the-art results for efficient single-image super-resolution.

  </details>



- **The Compression Gap: Why Discrete Tokenization Limits Vision-Language-Action Model Scaling**  
  Takuya Shiba  
  _2026-04-03_ · https://arxiv.org/abs/2604.03191v1  
  <details><summary>Abstract</summary>

  Scaling Vision-Language-Action (VLA) models by upgrading the vision encoder is expected to improve downstream manipulation performance--as it does in vision-language modeling. We show that this expectation fails when actions are represented as discrete tokens, and explain why through an information-theoretic principle we call the Compression Gap: in any visuomotor pipeline, scaling behavior is governed by the location of the tightest information bottleneck. When actions are continuous (e.g., Diffusion Policy), the vision encoder is the binding constraint, and upgrading it directly improves performance. When actions are discretized through a fixed-capacity codebook (e.g., OAT), the codebook becomes the binding constraint, and encoder improvements cannot propagate past it--regardless of how rich the upstream representation is. We validate this principle on the LIBERO benchmark with three lines of evidence: a factorial experiment showing that encoder upgrades improve Diffusion Policy by over 21 percentage points while OAT gains are substantially attenuated across model scales; an encoder quality gradient across four encoders confirming that Diffusion Policy tracks encoder quality monotonically while OAT remains flat; and a codebook size experiment demonstrating that relaxing codebook capacity partially recovers encoder sensitivity, providing causal evidence for the bottleneck hypothesis. Our findings reveal that scaling in Physical AI requires identifying where information bottlenecks lie in the pipeline, rather than uniformly increasing model or data size.

  </details>



- **EffiMiniVLM: A Compact Dual-Encoder Regression Framework**  
  Yin-Loon Khor, Yi-Jie Wong, Yan Chai Hum  
  _2026-04-03_ · https://arxiv.org/abs/2604.03172v1  
  <details><summary>Abstract</summary>

  Predicting product quality from multimodal item information is critical in cold-start scenarios, where user interaction history is unavailable and predictions must rely on images and textual metadata. However, existing vision-language models typically depend on large architectures and/or extensive external datasets, resulting in high computational cost. To address this, we propose EffiMiniVLM, a compact dual-encoder vision-language regression framework that integrates an EfficientNet-B0 image encoder and a MiniLM-based text encoder with a lightweight regression head. To improve training sample efficiency, we introduce a weighted Huber loss that leverages rating counts to emphasize more reliable samples, yielding consistent performance gains. Trained using only 20% of the Amazon Reviews 2023 dataset, the proposed model contains 27.7M parameters and requires 6.8 GFLOPs, yet achieves a CES score of 0.40 with the lowest resource cost in the benchmark. Despite its small size, it remains competitive with significantly larger models, achieving comparable performance while being approximately 4x to 8x more resource-efficient than other top-5 methods and being the only approach that does not use external datasets. Further analysis shows that scaling the data to 40% alone allows our model to overtake other methods, which use larger models and datasets, highlighting strong scalability despite the model's compact design.

  </details>



- **SCC-Loc: A Unified Semantic Cascade Consensus Framework for UAV Thermal Geo-Localization**  
  Xiaoran Zhang, Yu Liu, Jinyu Liang, Kangqiushi Li, Zhiwei Huang, Huaxin Xiao  
  _2026-04-03_ · https://arxiv.org/abs/2604.03120v1  
  <details><summary>Abstract</summary>

  Cross-modal Thermal Geo-localization (TG) provides a robust, all-weather solution for Unmanned Aerial Vehicles (UAVs) in Global Navigation Satellite System (GNSS)-denied environments. However, profound thermal-visible modality gaps introduce severe feature ambiguity, systematically corrupting conventional coarse-to-fine registration. To dismantle this bottleneck, we propose SCC-Loc, a unified Semantic-Cascade-Consensus localization framework. By sharing a single DINOv2 backbone across global retrieval and MINIMA$_{\text{RoMa}}$ matching, it minimizes memory footprint and achieves zero-shot, highly accurate absolute position estimation. Specifically, we tackle modality ambiguity by introducing three cohesive components. First, we design the Semantic-Guided Viewport Alignment (SGVA) module to adaptively optimize satellite crop regions, effectively correcting initial spatial deviations. Second, we develop the Cascaded Spatial-Adaptive Texture-Structure Filtering (C-SATSF) mechanism to explicitly enforce geometric consistency, thereby eradicating dense cross-modal outliers. Finally, we propose the Consensus-Driven Reliability-Aware Position Selection (CD-RAPS) strategy to derive the optimal solution through a synergy of physically constrained pose optimization. To address data scarcity, we construct Thermal-UAV, a comprehensive dataset providing 11,890 diverse thermal queries referenced against a large-scale satellite ortho-photo and corresponding spatially aligned Digital Surface Model (DSM). Extensive experiments demonstrate that SCC-Loc establishes a new state-of-the-art, suppressing the mean localization error to 9.37 m and providing a 7.6-fold accuracy improvement within a strict 5-m threshold over the strongest baseline. Code and dataset are available at https://github.com/FloralHercules/SCC-Loc.

  </details>



- **Revealing Physical-World Semantic Vulnerabilities: Universal Adversarial Patches for Infrared Vision-Language Models**  
  Chengyin Hu, Yuxian Dong, Yikun Guo, Xiang Chen, Junqi Wu, Jiahuan Long, Yiwei Wei, Tingsong Jiang, Wen Yao  
  _2026-04-03_ · https://arxiv.org/abs/2604.03117v1  
  <details><summary>Abstract</summary>

  Infrared vision-language models (IR-VLMs) have emerged as a promising paradigm for multimodal perception in low-visibility environments, yet their robustness to adversarial attacks remains largely unexplored. Existing adversarial patch methods are mainly designed for RGB-based models in closed-set settings and are not readily applicable to the open-ended semantic understanding and physical deployment requirements of infrared VLMs. To bridge this gap, we propose Universal Curved-Grid Patch (UCGP), a universal physical adversarial patch framework for IR-VLMs. UCGP integrates Curved-Grid Mesh (CGM) parameterization for continuous, low-frequency, and deployable patch generation with a unified representation-driven objective that promotes subspace departure, topology disruption, and stealth. To improve robustness under real-world deployment and domain shift, we further incorporate Meta Differential Evolution and EOT-augmented TPS deformation modeling. Rather than manipulating labels or prompts, UCGP directly disrupts the visual representation space, weakening cross-modal semantic alignment. Extensive experiments demonstrate that UCGP consistently compromises semantic understanding across diverse IR-VLM architectures while maintaining cross-model transferability, cross-dataset generalization, real-world physical effectiveness, and robustness against defenses. These findings reveal a previously overlooked robustness vulnerability in current infrared multimodal systems.

  </details>



- **Can VLMs Truly Forget? Benchmarking Training-Free Visual Concept Unlearning**  
  Zhangyun Tan, Zeliang Zhang, Susan Liang, Yolo Yunlong Tang, Lisha Chen, Chenliang Xu  
  _2026-04-03_ · https://arxiv.org/abs/2604.03114v1  
  <details><summary>Abstract</summary>

  VLMs trained on web-scale data retain sensitive and copyrighted visual concepts that deployment may require removing. Training-based unlearning methods share a structural flaw: fine-tuning on a narrow forget set degrades general capabilities before unlearning begins, making it impossible to attribute subsequent performance drops to the unlearning procedure itself. Training-free approaches sidestep this by suppressing concepts through prompts or system instructions, but no rigorous benchmark exists for evaluating them on visual tasks. We introduce VLM-UnBench, the first benchmark for training-free visual concept unlearning in VLMs. It covers four forgetting levels, 7 source datasets, and 11 concept axes, and pairs a three-level probe taxonomy with five evaluation conditions to separate genuine forgetting from instruction compliance. Across 8 evaluation settings and 13 VLM configurations, realistic unlearning prompts leave forget accuracy near the no-instruction baseline; meaningful reductions appear only under oracle conditions that disclose the target concept to the model. Object and scene concepts are the most resistant to suppression, and stronger instruction-tuned models remain capable despite explicit forget instructions. These results expose a clear gap between prompt-level suppression and true visual concept erasure.

  </details>



- **ARIQA-3DS: A Stereoscopic Image Quality Assessment Dataset for Realistic Augmented Reality**  
  Aymen Sekhri, Seyed Ali Amirshahi, Mohamed-Chaker Larabi  
  _2026-04-03_ · https://arxiv.org/abs/2604.03112v1  
  <details><summary>Abstract</summary>

  As Augmented Reality (AR) technologies advance towards immersive consumer adoption, the need for rigorous Quality of Experience (QoE) assessment becomes critical. However, existing datasets often lack ecological validity, relying on monocular viewing or simplified backgrounds that fail to capture the complex perceptual interplay, termed visual confusion, between real and virtual layers. To address this gap, we present ARIQA-3DS, the first large stereoscopic AR Image Quality Assessment dataset. Comprising 1,200 AR viewports, the dataset fuses high-resolution stereoscopic omnidirectional captures of real-world scenes with diverse augmented foregrounds under controlled transparency and degradation conditions. We conducted a comprehensive subjective study with 36 participants using a video see-through head-mounted display, collecting both quality ratings and simulator-sickness indicators. Our analysis reveals that perceived quality is primarily driven by foreground degradations and modulated by transparency levels, while oculomotor and disorientation symptoms show a progressive but manageable increase during viewing. ARIQA-3DS will be publicly released to serve as a comprehensive benchmark for developing next-generation AR quality assessment models.

  </details>



- **An Open-Source LiDAR and Monocular Off-Road Autonomous Navigation Stack**  
  Rémi Marsal, Quentin Picard, Adrien Poiré, Sébastien Kerbourc'h, Thibault Toralba, Clément Yver, Alexandre Chapoutot, David Filliat  
  _2026-04-03_ · https://arxiv.org/abs/2604.03096v1  
  <details><summary>Abstract</summary>

  Off-road autonomous navigation demands reliable 3D perception for robust obstacle detection in challenging unstructured terrain. While LiDAR is accurate, it is costly and power-intensive. Monocular depth estimation using foundation models offers a lightweight alternative, but its integration into outdoor navigation stacks remains underexplored. We present an open-source off-road navigation stack supporting both LiDAR and monocular 3D perception without task-specific training. For the monocular setup, we combine zero-shot depth prediction (Depth Anything V2) with metric depth rescaling using sparse SLAM measurements (VINS-Mono). Two key enhancements improve robustness: edge-masking to reduce obstacle hallucination and temporal smoothing to mitigate the impact of SLAM instability. The resulting point cloud is used to generate a robot-centric 2.5D elevation map for costmap-based planning. Evaluated in photorealistic simulations (Isaac Sim) and real-world unstructured environments, the monocular configuration matches high-resolution LiDAR performance in most scenarios, demonstrating that foundation-model-based monocular depth estimation is a viable LiDAR alternative for robust off-road navigation. By open-sourcing the navigation stack and the simulation environment, we provide a complete pipeline for off-road navigation as well as a reproducible benchmark. Code available at https://github.com/LARIAD/Offroad-Nav.

  </details>



- **A Data-Centric Vision Transformer Baseline for SAR Sea Ice Classification**  
  David Mike-Ewewie, Panhapiseth Lim, Priyanka Kumar  
  _2026-04-03_ · https://arxiv.org/abs/2604.03094v1  
  <details><summary>Abstract</summary>

  Accurate and automated sea ice classification is important for climate monitoring and maritime safety in the Arctic. While Synthetic Aperture Radar (SAR) is the operational standard because of its all-weather capability, it remains challenging to distinguish morphologically similar ice classes under severe class imbalance. Rather than claiming a fully validated multimodal system, this paper establishes a trustworthy SAR only baseline that future fusion work can build upon. Using the AI4Arctic/ASIP Sea Ice Dataset (v2), which contains 461 Sentinel-1 scenes matched with expert ice charts, we combine full-resolution Sentinel-1 Extra Wide inputs, leakage-aware stratified patch splitting, SIGRID-3 stage-of-development labels, and training-set normalization to evaluate Vision Transformer baselines. We compare ViT-Base models trained with cross entropy and weighted cross-entropy against a ViT-Large model trained with focal loss. Among the tested configurations, ViT-Large with focal loss achieves 69.6% held-out accuracy, 68.8% weighted F1, and 83.9% precision on the minority Multi-Year Ice class. These results show that focal-loss training offers a more useful precision-recall trade-off than weighted cross-entropy for rare ice classes and establishes a cleaner baseline for future multimodal fusion with optical, thermal, or meteorological data.

  </details>



- **Gram-MMD: A Texture-Aware Metric for Image Realism Assessment**  
  Joé Napolitano, Pascal Nguyen  
  _2026-04-03_ · https://arxiv.org/abs/2604.03064v1  
  <details><summary>Abstract</summary>

  Evaluating the realism of generated images remains a fundamental challenge in generative modeling. Existing distributional metrics such as the Frechet Inception Distance (FID) and CLIP-MMD (CMMD) compare feature distributions at a semantic level but may overlook fine-grained textural information that can be relevant for distinguishing real from generated images. We introduce Gram-MMD (GMMD), a realism metric that leverages Gram matrices computed from intermediate activations of pretrained backbone networks to capture correlations between feature maps. By extracting the upper-triangular part of these symmetric Gram matrices and measuring the Maximum Mean Discrepancy (MMD) between an anchor distribution of real images and an evaluation distribution, GMMD produces a representation that encodes textural and structural characteristics at a finer granularity than global embeddings. To select the hyperparameters of the metric, we employ a meta-metric protocol based on controlled degradations applied to MS-COCO images, measuring monotonicity via Spearman's rank correlation and Kendall's tau. We conduct experiments on both the KADID-10k database and the RAISE realness assessment dataset using various backbone architectures, including DINOv2, DC-AE, Stable Diffusion's VAE encoder, VGG19, and the AlexNet backbone from LPIPS, among others. We also demonstrate on a cross-domain driving scenario (KITTI / Virtual KITTI / Stanford Cars) that CMMD can incorrectly rank real images as less realistic than synthetic ones due to its semantic bias, while GMMD preserves the correct ordering. Our results suggest that GMMD captures complementary information to existing semantic-level metrics.

  </details>



- **QVAD: A Question-Centric Agentic Framework for Efficient and Training-Free Video Anomaly Detection**  
  Lokman Bekit, Hamza Karim, Nghia T Nguyen, Yasin Yilmaz  
  _2026-04-03_ · https://arxiv.org/abs/2604.03040v1  
  <details><summary>Abstract</summary>

  Video Anomaly Detection (VAD) is a fundamental challenge in computer vision, particularly due to the open-set nature of anomalies. While recent training-free approaches utilizing Vision-Language Models (VLMs) have shown promise, they typically rely on massive, resource-intensive foundation models to compensate for the ambiguity of static prompts. We argue that the bottleneck in VAD is not necessarily model capacity, but rather the static nature of inquiry. We propose QVAD, a question-centric agentic framework that treats VLM-LLM interaction as a dynamic dialogue. By iteratively refining queries based on visual context, our LLM agent guides smaller VLMs to produce high-fidelity captions and precise semantic reasoning without parameter updates. This ``prompt-updating" mechanism effectively unlocks the latent capabilities of lightweight models, enabling state-of-the-art performance on UCF-Crime, XD-Violence, and UBNormal using a fraction of the parameters required by competing methods. We further demonstrate exceptional generalizability on the single-scene ComplexVAD dataset. Crucially, QVAD achieves high inference speeds with minimal memory footprints, making advanced VAD capabilities deployable on resource-constrained edge devices.

  </details>



- **GenSmoke-GS: A Multi-Stage Method for Novel View Synthesis from Smoke-Degraded Images Using a Generative Model**  
  Qida Cao, Xinyuan Hu, Changyue Shi, Jiajun Ding, Zhou Yu, Jun Yu  
  _2026-04-03_ · https://arxiv.org/abs/2604.03039v1  
  <details><summary>Abstract</summary>

  This paper describes our method for Track 2 of the NTIRE 2026 3D Restoration and Reconstruction (3DRR) Challenge on smoke-degraded images. In this task, smoke reduces image visibility and weakens the cross-view consistency required by scene optimization and rendering. We address this problem with a multi-stage pipeline consisting of image restoration, dehazing, MLLM-based enhancement, 3DGS-MCMC optimization, and averaging over repeated runs. The main purpose of the pipeline is to improve visibility before rendering while limiting scene-content changes across input views. Experimental results on the challenge benchmark show improved quantitative performance and better visual quality than the provided baselines. The code is available at https://github.com/plbbl/GenSmoke-GS. Our method achieved a ranking of 1 out of 14 participants in Track 2 of the NTIRE 3DRR Challenge, as reported on the official competition website: https://www.codabench.org/competitions/13993/#/results-tab.

  </details>



- **ARM: Advantage Reward Modeling for Long-Horizon Manipulation**  
  Yiming Mao, Zixi Yu, Weixin Mao, Yinhao Li, Qirui Hu, Zihan Lan, Minzhao Zhu, Hua Chen  
  _2026-04-03_ · https://arxiv.org/abs/2604.03037v1  
  <details><summary>Abstract</summary>

  Long-horizon robotic manipulation remains challenging for reinforcement learning (RL) because sparse rewards provide limited guidance for credit assignment. Practical policy improvement thus relies on richer intermediate supervision, such as dense progress rewards, which are costly to obtain and ill-suited to non-monotonic behaviors such as backtracking and recovery. To address this, we propose Advantage Reward Modeling (ARM), a framework that shifts from hard-to-quantify absolute progress to estimating relative advantage. We introduce a cost-effective tri-state labeling strategy -- Progressive, Regressive, and Stagnant -- that reduces human cognitive overhead while ensuring high cross-annotator consistency. By training on these intuitive signals, ARM enables automated progress annotation for both complete demonstrations and fragmented DAgger-style data. Integrating ARM into an offline RL pipeline allows for adaptive action-reward reweighting, effectively filtering suboptimal samples. Our approach achieves a 99.4% success rate on a challenging long-horizon towel-folding task, demonstrating improved stability and data efficiency over current VLA baselines with near-zero human intervention during policy training.

  </details>



- **PolyReal: A Benchmark for Real-World Polymer Science Workflows**  
  Wanhao Liu, Weida Wang, Jiaqing Xie, Suorong Yang, Jue Wang, Benteng Chen, Guangtao Mei, Zonglin Yang, Shufei Zhang, Yuchun Mo, et al.  
  _2026-04-03_ · https://arxiv.org/abs/2604.02934v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) excel in general domains but struggle with complex, real-world science. We posit that polymer science, an interdisciplinary field spanning chemistry, physics, biology, and engineering, is an ideal high-stakes testbed due to its diverse multimodal data. Yet, existing benchmarks related to polymer science largely overlook real-world workflows, limiting their practical utility and failing to systematically evaluate MLLMs across the full, practice-grounded lifecycle of experimentation. We introduce PolyReal, a novel multimodal benchmark grounded in real-world scientific practices to evaluate MLLMs on the full lifecycle of polymer experimentation. It covers five critical capabilities: (1) foundational knowledge application; (2) lab safety analysis; (3) experiment mechanism reasoning; (4) raw data extraction; and (5) performance & application exploration. Our evaluation of leading MLLMs on PolyReal reveals a capability imbalance. While models perform well on knowledge-intensive reasoning (e.g., Experiment Mechanism Reasoning), they drop sharply on practice-based tasks (e.g., Lab Safety Analysis and Raw Data Extraction). This exposes a severe gap between abstract scientific knowledge and its practical, context-dependent application, showing that these real-world tasks remain challenging for MLLMs. Thus, PolyReal helps address this evaluation gap and provides a practical benchmark for assessing AI systems in real-world scientific workflows.

  </details>



- **BEVPredFormer: Spatio-temporal Attention for BEV Instance Prediction in Autonomous Driving**  
  Miguel Antunes-García, Santiago Montiel-Marín, Fabio Sánchez-García, Rodrigo Gutiérrez-Moreno, Rafael Barea, Luis M. Bergasa  
  _2026-04-03_ · https://arxiv.org/abs/2604.02930v1  
  <details><summary>Abstract</summary>

  A robust awareness of how dynamic scenes evolve is essential for Autonomous Driving systems, as they must accurately detect, track, and predict the behaviour of surrounding obstacles. Traditional perception pipelines that rely on modular architectures tend to suffer from cumulative errors and latency. Instance Prediction models provide a unified solution, performing Bird's-Eye-View segmentation and motion estimation across current and future frames using information directly obtained from different sensors. However, a key challenge in these models lies in the effective processing of the dense spatial and temporal information inherent in dynamic driving environments. This level of complexity demands architectures capable of capturing fine-grained motion patterns and long-range dependencies without compromising real-time performance. We introduce BEVPredFormer, a novel camera-only architecture for BEV instance prediction that uses attention-based temporal processing to improve temporal and spatial comprehension of the scene and relies on an attention-based 3D projection of the camera information. BEVPredFormer employs a recurrent-free design that incorporates gated transformer layers, divided spatio-temporal attention mechanisms, and multi-scale head tasks. Additionally, we incorporate a difference-guided feature extraction module that enhances temporal representations. Extensive ablation studies validate the effectiveness of each architectural component. When evaluated on the nuScenes dataset, BEVPredFormer was on par or surpassed State-Of-The-Art methods, highlighting its potential for robust and efficient Autonomous Driving perception.

  </details>



- **SentiAvatar: Towards Expressive and Interactive Digital Humans**  
  Chuhao Jin, Rui Zhang, Qingzhe Gao, Haoyu Shi, Dayu Wu, Yichen Jiang, Yihan Wu, Ruihua Song  
  _2026-04-03_ · https://arxiv.org/abs/2604.02908v1  
  <details><summary>Abstract</summary>

  We present SentiAvatar, a framework for building expressive interactive 3D digital humans, and use it to create SuSu, a virtual character that speaks, gestures, and emotes in real time. Achieving such a system remains challenging, as it requires jointly addressing three key problems: the lack of large-scale, high-quality multimodal data, robust semantic-to-motion mapping, and fine-grained frame-level motion-prosody synchronization. To solve these problems, first, we build SuSuInterActs (21K clips, 37 hours), a dialogue corpus captured via optical motion capture around a single character with synchronized speech, full-body motion, and facial expressions. Second, we pre-train a Motion Foundation Model on 200K+ motion sequences, equipping it with rich action priors that go well beyond the conversation. We then propose an audio-aware plan-then-infill architecture that decouples sentence-level semantic planning from frame-level prosody-driven interpolation, so that generated motions are both semantically appropriate and rhythmically aligned with speech. Experiments show that SentiAvatar achieves state-of-the-art on both SuSuInterActs (R@1 43.64%, nearly 2 times the best baseline) and BEATv2 (FGD 4.941, BC 8.078), producing 6s of output in 0.3s with unlimited multi-turn streaming. The source code, model, and dataset are available at https://sentiavatar.github.io.

  </details>



- **UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting**  
  Geonuk Kim, Minhoi Kim, Kangil Lee, Minsu Kim, Hyeonseong Jeon, Jeonghoon Han, Hyoungjoon Lim, Junho Yim  
  _2026-04-03_ · https://arxiv.org/abs/2604.02905v1  
  <details><summary>Abstract</summary>

  Although industrial inspection systems should be capable of recognizing unprecedented defects, most existing approaches operate under a closed-set assumption, which prevents them from detecting novel anomalies. While visual prompting offers a scalable alternative for industrial inspection, existing methods often suffer from prompt embedding collapse due to high intra-class variance and subtle inter-class differences. To resolve this, we propose UniSpector, which shifts the focus from naive prompt-to-region matching to the principled design of a semantically structured and transferable prompt topology. UniSpector employs the Spatial-Spectral Prompt Encoder to extract orientation-invariant, fine-grained representations; these serve as a solid basis for the Contrastive Prompt Encoder to explicitly regularize the prompt space into a semantically organized angular manifold. Additionally, Prompt-guided Query Selection generates adaptive object queries aligned with the prompt. We introduce Inspect Anything, the first benchmark for visual-prompt-based open-set defect localization, where UniSpector significantly outperforms baselines by at least 19.7% and 15.8% in AP50b and AP50m, respectively. These results show that our method enable a scalable, retraining-free inspection paradigm for continuously evolving industrial environments, while offering critical insights into the design of generic visual prompting.

  </details>



- **Toward an Artificial General Teacher: Procedural Geometry Data Generation and Visual Grounding with Vision-Language Models**  
  Hai Nguyen-Truong, Alper Balbay, Tunga Bayrak  
  _2026-04-03_ · https://arxiv.org/abs/2604.02893v1  
  <details><summary>Abstract</summary>

  We study visual explanation in geometry education as a Referring Image Segmentation (RIS) problem: given a diagram and a natural language description, the task is to produce a pixel-level mask for the referred geometric element. However, existing RIS models trained on natural image benchmarks such as RefCOCO fail catastrophically on geometric diagrams due to the fundamental domain shift between photographic scenes and abstract, textureless schematics. To address the absence of suitable training data, we present a fully automated procedural data engine that generates over 200,000 synthetic geometry diagrams with pixel-perfect segmentation masks and linguistically diverse referring expressions, requiring zero manual annotation. We further propose domain-specific fine-tuning of vision-language models (VLMs), demonstrating that a fine-tuned Florence-2 achieves 49% IoU and 85% Buffered IoU (BIoU), compared to <1% IoU in zero-shot settings. We introduce Buffered IoU, a geometry-aware evaluation metric that accounts for thin-structure localization, and show that it better reflects true segmentation quality than standard IoU. Our results establish a foundation for building Artificial General Teachers (AGTs) capable of providing visually grounded, step-by-step explanations of geometry problems.

  </details>



- **InstructTable: Improving Table Structure Recognition Through Instructions**  
  Boming Chen, Zining Wang, Zhentao Guo, Jianqiang Liu, Chen Duan, Yu Gu, Kai zhou, Pengfei Yan  
  _2026-04-03_ · https://arxiv.org/abs/2604.02880v1  
  <details><summary>Abstract</summary>

  Table structure recognition (TSR) holds widespread practical importance by parsing tabular images into structured representations, yet encounters significant challenges when processing complex layouts involving merged or empty cells. Traditional visual-centric models rely exclusively on visual information while lacking crucial semantic support, thereby impeding accurate structural recognition in complex scenarios. Vision-language models leverage contextual semantics to enhance comprehension; however, these approaches underemphasize the modeling of visual structural information. To address these limitations, this paper introduces InstructTable, an instruction-guided multi-stage training TSR framework. Meticulously designed table instruction pre-training directs attention toward fine-grained structural patterns, enhancing comprehension of complex tables. Complementary TSR fine-tuning preserves robust visual information modeling, maintaining high-precision table parsing across diverse scenarios. Furthermore, we introduce Table Mix Expand (TME), an innovative template-free method for synthesizing large-scale authentic tabular data. Leveraging TME, we construct the Balanced Complex Dense Synthetic Tables (BCDSTab) benchmark, comprising 900 complex table images synthesized through our method to serve as a rigorous benchmark. Extensive experiments on multiple public datasets (FinTabNet, PubTabNet, MUSTARD) and BCDSTab demonstrate that InstructTable achieves state-of-the-art performance in TSR tasks. Ablation studies further confirm the positive impact of the proposed tabular-data-specific instructions and synthetic data.

  </details>



- **SPG: Sparse-Projected Guides with Sparse Autoencoders for Zero-Shot Anomaly Detection**  
  Tomoyasu Nanaumi, Yukino Tsuzuki, Junichi Okubo, Junichiro Fujii, Takayoshi Yamashita  
  _2026-04-03_ · https://arxiv.org/abs/2604.02871v1  
  <details><summary>Abstract</summary>

  We study zero-shot anomaly detection and segmentation using frozen foundation model features, where all learnable parameters are trained only on a labeled auxiliary dataset and deployed to unseen target categories without any target-domain adaptation. Existing prompt-based approaches use handcrafted or learned prompt embeddings as reference vectors for normal/anomalous states. We propose Sparse-Projected Guides (SPG), a prompt-free framework that learns sparse guide coefficients in the Sparse Autoencoder (SAE) latent space, which generate normal/anomaly guide vectors via the SAE dictionary. SPG employs a two stage learning strategy on the labeled auxiliary dataset: (i) train an SAE on patch-token features, and (ii) optimize only guide coefficients using auxiliary pixel-level masks while freezing the backbone and SAE. On MVTec AD and VisA under cross-dataset zero-shot settings, SPG achieves competitive image-level detection and strong pixel-level segmentation; with DINOv3, SPG attains the highest pixellevel AUROC among the compared methods. We also report SPG instantiated with OpenCLIP (ViT-L/14@336px) to align the backbone with CLIP-based baselines. Moreover, the learned guide coefficients trace decisions back to a small set of dictionary atoms, revealing category-general and category-specific factors.

  </details>



- **Token Warping Helps MLLMs Look from Nearby Viewpoints**  
  Phillip Y. Lee, Chanho Park, Mingue Park, Seungwoo Yoo, Juil Koo, Minhyuk Sung  
  _2026-04-03_ · https://arxiv.org/abs/2604.02870v1  
  <details><summary>Abstract</summary>

  Can warping tokens, rather than pixels, help multimodal large language models (MLLMs) understand how a scene appears from a nearby viewpoint? While MLLMs perform well on visual reasoning, they remain fragile to viewpoint changes, as pixel-wise warping is highly sensitive to small depth errors and often introduces geometric distortions. Drawing on theories of mental imagery that posit part-level structural representations as the basis for human perspective transformation, we examine whether image tokens in ViT-based MLLMs serve as an effective substrate for viewpoint changes. We compare forward and backward warping, finding that backward token warping, which defines a dense grid on the target view and retrieves a corresponding source-view token for each grid point, achieves greater stability and better preserves semantic coherence under viewpoint shifts. Experiments on our proposed ViewBench benchmark demonstrate that token-level warping enables MLLMs to reason reliably from nearby viewpoints, consistently outperforming all baselines including pixel-wise warping approaches, spatially fine-tuned MLLMs, and a generative warping method.

  </details>



- **Deformation-based In-Context Learning for Point Cloud Understanding**  
  Chengxing Lin, Jinhong Deng, Yinjie Lei, Wen Li  
  _2026-04-03_ · https://arxiv.org/abs/2604.02845v1  
  <details><summary>Abstract</summary>

  Recent advances in point cloud In-Context Learning (ICL) have demonstrated strong multitask capabilities. Existing approaches typically adopt a Masked Point Modeling (MPM)-based paradigm for point cloud ICL. However, MPM-based methods directly predict the target point cloud from masked tokens without leveraging geometric priors, requiring the model to infer spatial structure and geometric details solely from token-level correlations via transformers. Additionally, these methods suffer from a training-inference objective mismatch, as the model learns to predict the target point cloud using target-side information that is unavailable at inference time. To address these challenges, we propose DeformPIC, a deformation-based framework for point cloud ICL. Unlike existing approaches that rely on masked reconstruction, DeformPIC learns to deform the query point cloud under task-specific guidance from prompts, enabling explicit geometric reasoning and consistent objectives. Extensive experiments demonstrate that DeformPIC consistently outperforms previous state-of-the-art methods, achieving reductions of 1.6, 1.8, and 4.7 points in average Chamfer Distance on reconstruction, denoising, and registration tasks, respectively. Furthermore, we introduce a new out-of-domain benchmark to evaluate generalization across unseen data distributions, where DeformPIC achieves state-of-the-art performance.

  </details>


