# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-03 07:08 UTC_

Total papers shown: **50**


---

- **HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images**  
  Yichen Liu, Donghao Zhou, Jie Wang, Xin Gao, Guisheng Liu, Jiatong Li, Quanwei Zhang, Qiang Lyu, Lanqing Guo, Shilei Wen, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.02210v1  
  <details><summary>Abstract</summary>

  Human-product images, which showcase the integration of humans and products, play a vital role in advertising, e-commerce, and digital marketing. The essential challenge of generating such images lies in ensuring the high-fidelity preservation of product details. Among existing paradigms, reference-based inpainting offers a targeted solution by leveraging product reference images to guide the inpainting process. However, limitations remain in three key aspects: the lack of diverse large-scale training data, the struggle of current models to focus on product detail preservation, and the inability of coarse supervision for achieving precise guidance. To address these issues, we propose HiFi-Inpaint, a novel high-fidelity reference-based inpainting framework tailored for generating human-product images. HiFi-Inpaint introduces Shared Enhancement Attention (SEA) to refine fine-grained product features and Detail-Aware Loss (DAL) to enforce precise pixel-level supervision using high-frequency maps. Additionally, we construct a new dataset, HP-Image-40K, with samples curated from self-synthesis data and processed with automatic filtering. Experimental results show that HiFi-Inpaint achieves state-of-the-art performance, delivering detail-preserving human-product images.

  </details>



- **From Leaderboard to Deployment: Code Quality Challenges in AV Perception Repositories**  
  Mateus Karvat, Bram Adams, Sidney Givigi  
  _2026-03-02_ · https://arxiv.org/abs/2603.02194v1  
  <details><summary>Abstract</summary>

  Autonomous vehicle (AV) perception models are typically evaluated solely on benchmark performance metrics, with limited attention to code quality, production readiness and long-term maintainability. This creates a significant gap between research excellence and real-world deployment in safety-critical systems subject to international safety standards. To address this gap, we present the first large-scale empirical study of software quality in AV perception repositories, systematically analyzing 178 unique models from the KITTI and NuScenes 3D Object Detection leaderboards. Using static analysis tools (Pylint, Bandit, and Radon), we evaluated code errors, security vulnerabilities, maintainability, and development practices. Our findings revealed that only 7.3% of the studied repositories meet basic production-readiness criteria, defined as having zero critical errors and no high-severity security vulnerabilities. Security issues are highly concentrated, with the top five issues responsible for almost 80% of occurrences, which prompted us to develop a set of actionable guidelines to prevent them. Additionally, the adoption of Continuous Integration/Continuous Deployment pipelines was correlated with better code maintainability. Our findings highlight that leaderboard performance does not reflect production readiness and that targeted interventions could substantially improve the quality and safety of AV perception code.

  </details>



- **Leveraging Model Soups to Classify Intangible Cultural Heritage Images from the Mekong Delta**  
  Quoc-Khang Tran, Minh-Thien Nguyen, Nguyen-Khang Pham  
  _2026-03-02_ · https://arxiv.org/abs/2603.02181v1  
  <details><summary>Abstract</summary>

  The classification of Intangible Cultural Heritage (ICH) images in the Mekong Delta poses unique challenges due to limited annotated data, high visual similarity among classes, and domain heterogeneity. In such low-resource settings, conventional deep learning models often suffer from high variance or overfit to spurious correlations, leading to poor generalization. To address these limitations, we propose a robust framework that integrates the hybrid CoAtNet architecture with model soups, a lightweight weight-space ensembling technique that averages checkpoints from a single training trajectory without increasing inference cost. CoAtNet captures both local and global patterns through stage-wise fusion of convolution and self-attention. We apply two ensembling strategies - greedy and uniform soup - to selectively combine diverse checkpoints into a final model. Beyond performance improvements, we analyze the ensembling effect through the lens of bias-variance decomposition. Our findings show that model soups reduces variance by stabilizing predictions across diverse model snapshots, while introducing minimal additional bias. Furthermore, using cross-entropy-based distance metrics and Multidimensional Scaling (MDS), we show that model soups selects geometrically diverse checkpoints, unlike Soft Voting, which blends redundant models centered in output space. Evaluated on the ICH-17 dataset (7,406 images across 17 classes), our approach achieves state-of-the-art results with 72.36% top-1 accuracy and 69.28% macro F1-score, outperforming strong baselines including ResNet-50, DenseNet-121, and ViT. These results underscore that diversity-aware checkpoint averaging provides a principled and efficient way to reduce variance and enhance generalization in culturally rich, data-scarce classification tasks.

  </details>



- **Kiwi-Edit: Versatile Video Editing via Instruction and Reference Guidance**  
  Yiqi Lin, Guoqiang Liang, Ziyun Zeng, Zechen Bai, Yanzhe Chen, Mike Zheng Shou  
  _2026-03-02_ · https://arxiv.org/abs/2603.02175v1  
  <details><summary>Abstract</summary>

  Instruction-based video editing has witnessed rapid progress, yet current methods often struggle with precise visual control, as natural language is inherently limited in describing complex visual nuances. Although reference-guided editing offers a robust solution, its potential is currently bottlenecked by the scarcity of high-quality paired training data. To bridge this gap, we introduce a scalable data generation pipeline that transforms existing video editing pairs into high-fidelity training quadruplets, leveraging image generative models to create synthesized reference scaffolds. Using this pipeline, we construct RefVIE, a large-scale dataset tailored for instruction-reference-following tasks, and establish RefVIE-Bench for comprehensive evaluation. Furthermore, we propose a unified editing architecture, Kiwi-Edit, that synergizes learnable queries and latent visual features for reference semantic guidance. Our model achieves significant gains in instruction following and reference fidelity via a progressive multi-stage training curriculum. Extensive experiments demonstrate that our data and architecture establish a new state-of-the-art in controllable video editing. All datasets, models, and code is released at https://github.com/showlab/Kiwi-Edit.

  </details>



- **GeoDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis**  
  Srikumar Sastry, Dan Cher, Brian Wei, Aayush Dhakal, Subash Khanal, Dev Gupta, Nathan Jacobs  
  _2026-03-02_ · https://arxiv.org/abs/2603.02172v1  
  <details><summary>Abstract</summary>

  We introduce GeoDiT, a diffusion transformer designed for text-to-satellite image generation with point-based control. Existing controlled satellite image generative models often require pixel-level maps that are time-consuming to acquire, yet semantically limited. To address this limitation, we introduce a novel point-based conditioning framework that controls the generation process through the spatial location of the points and the textual description associated with each point, providing semantically rich control signals. This approach enables flexible, annotation-friendly, and computationally simple inference for satellite image generation. To this end, we introduce an adaptive local attention mechanism that effectively regularizes the attention scores based on the input point queries. We systematically evaluate various domain-specific design choices for training GeoDiT, including the selection of satellite image representation for alignment and geolocation representation for conditioning. Our experiments demonstrate that GeoDiT achieves impressive generation performance, surpassing the state-of-the-art remote sensing generative models.

  </details>



- **Is Bigger Always Better? Efficiency Analysis in Resource-Constrained Small Object Detection**  
  Kwame Mbobda-Kuate, Gabriel Kasmi  
  _2026-03-02_ · https://arxiv.org/abs/2603.02142v1  
  <details><summary>Abstract</summary>

  Scaling laws assume larger models trained on more data consistently outperform smaller ones -- an assumption that drives model selection in computer vision but remains untested in resource-constrained Earth observation (EO). We conduct a systematic efficiency analysis across three scaling dimensions: model size, dataset size, and input resolution, on rooftop PV detection in Madagascar. Optimizing for model efficiency (mAP$_{50}$ per unit of model size), we find a consistent efficiency inversion: YOLO11N achieves both the highest efficiency ($24\times$ higher than YOLO11X) and the highest absolute mAP$_{50}$ (0.617). Resolution is the dominant resource allocation lever ($+$120% efficiency gain), while additional data yields negligible returns at low resolution. These findings are robust to the deployment objective: small high-resolution configurations are Pareto-dominant across all 44 setups in the joint accuracy-throughput space, leaving no tradeoff to resolve. In data-scarce EO, bigger is not just unnecessary: it can be worse.

  </details>



- **OmniLottie: Generating Vector Animations via Parameterized Lottie Tokens**  
  Yiying Yang, Wei Cheng, Sijin Chen, Honghao Fu, Xianfang Zeng, Yujun Cai, Gang Yu, Xingjun Ma  
  _2026-03-02_ · https://arxiv.org/abs/2603.02138v1  
  <details><summary>Abstract</summary>

  OmniLottie is a versatile framework that generates high quality vector animations from multi-modal instructions. For flexible motion and visual content control, we focus on Lottie, a light weight JSON formatting for both shapes and animation behaviors representation. However, the raw Lottie JSON files contain extensive invariant structural metadata and formatting tokens, posing significant challenges for learning vector animation generation. Therefore, we introduce a well designed Lottie tokenizer that transforms JSON files into structured sequences of commands and parameters representing shapes, animation functions and control parameters. Such tokenizer enables us to build OmniLottie upon pretrained vision language models to follow multi-modal interleaved instructions and generate high quality vector animations. To further advance research in vector animation generation, we curate MMLottie-2M, a large scale dataset of professionally designed vector animations paired with textual and visual annotations. With extensive experiments, we validate that OmniLottie can produce vivid and semantically aligned vector animations that adhere closely to multi modal human instructions.

  </details>



- **SimRecon: SimReady Compositional Scene Reconstruction from Real Videos**  
  Chong Xia, Kai Zhu, Zizhuo Wang, Fangfu Liu, Zhizheng Zhang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02133v1  
  <details><summary>Abstract</summary>

  Compositional scene reconstruction seeks to create object-centric representations rather than holistic scenes from real-world videos, which is natively applicable for simulation and interaction. Conventional compositional reconstruction approaches primarily emphasize on visual appearance and show limited generalization ability to real-world scenarios. In this paper, we propose SimRecon, a framework that realizes a "Perception-Generation-Simulation" pipeline towards cluttered scene reconstruction, which first conducts scene-level semantic reconstruction from video input, then performs single-object generation, and finally assembles these assets in the simulator. However, naively combining these three stages leads to visual infidelity of generated assets and physical implausibility of the final scene, a problem particularly severe for complex scenes. Thus, we further propose two bridging modules between the three stages to address this problem. To be specific, for the transition from Perception to Generation, critical for visual fidelity, we introduce Active Viewpoint Optimization, which actively searches in 3D space to acquire optimal projected images as conditions for single-object completion. Moreover, for the transition from Generation to Simulation, essential for physical plausibility, we propose a Scene Graph Synthesizer, which guides the construction from scratch in 3D simulators, mirroring the native, constructive principle of the real world. Extensive experiments on the ScanNet dataset validate our method's superior performance over previous state-of-the-art approaches.

  </details>



- **Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons**  
  Anthony Liang, Yigit Korkmaz, Jiahui Zhang, Minyoung Hwang, Abrar Anwar, Sidhant Kaushik, Aditya Shah, Alex S. Huang, Luke Zettlemoyer, Dieter Fox, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.02115v1  
  <details><summary>Abstract</summary>

  General-purpose robot reward models are typically trained to predict absolute task progress from expert demonstrations, providing only local, frame-level supervision. While effective for expert demonstrations, this paradigm scales poorly to large-scale robotics datasets where failed and suboptimal trajectories are abundant and assigning dense progress labels is ambiguous. We introduce Robometer, a scalable reward modeling framework that combines intra-trajectory progress supervision with inter-trajectory preference supervision. Robometer is trained with a dual objective: a frame-level progress loss that anchors reward magnitude on expert data, and a trajectory-comparison preference loss that imposes global ordering constraints across trajectories of the same task, enabling effective learning from both real and augmented failed trajectories. To support this formulation at scale, we curate RBM-1M, a reward-learning dataset comprising over one million trajectories spanning diverse robot embodiments and tasks, including substantial suboptimal and failure data. Across benchmarks and real-world evaluations, Robometer learns more generalizable reward functions than prior methods and improves robot learning performance across a diverse set of downstream applications. Code, model weights, and videos at https://robometer.github.io/.

  </details>



- **OmniRet: Efficient and High-Fidelity Omni Modality Retrieval**  
  Chuong Huynh, Manh Luong, Abhinav Shrivastava  
  _2026-03-02_ · https://arxiv.org/abs/2603.02098v1  
  <details><summary>Abstract</summary>

  Multimodal retrieval is the task of aggregating information from queries across heterogeneous modalities to retrieve desired targets. State-of-the-art multimodal retrieval models can understand complex queries, yet they are typically limited to two modalities: text and vision. This limitation impedes the development of universal retrieval systems capable of comprehending queries that combine more than two modalities. To advance toward this goal, we present OmniRet, the first retrieval model capable of handling complex, composed queries spanning three key modalities: text, vision, and audio. Our OmniRet model addresses two critical challenges for universal retrieval: computational efficiency and representation fidelity. First, feeding massive token sequences from modality-specific encoders to Large Language Models (LLMs) is computationally inefficient. We therefore introduce an attention-based resampling mechanism to generate compact, fixed-size representations from these sequences. Second, compressing rich omni-modal data into a single embedding vector inevitably causes information loss and discards fine-grained details. We propose Attention Sliced Wasserstein Pooling to preserve these fine-grained details, leading to improved omni-modal representations. OmniRet is trained on an aggregation of approximately 6 million query-target pairs spanning 30 datasets. We benchmark our model on 13 retrieval tasks and a MMEBv2 subset. Our model demonstrates significant improvements on composed query, audio and video retrieval tasks, while achieving on-par performance with state-of-the-art models on others. Furthermore, we curate a new Audio-Centric Multimodal Benchmark (ACM). This new benchmark introduces two critical, previously missing tasks-composed audio retrieval and audio-visual retrieval to more comprehensively evaluate a model's omni-modal embedding capacity.

  </details>



- **Detection-Gated Glottal Segmentation with Zero-Shot Cross-Dataset Transfer and Clinical Feature Extraction**  
  Harikrishnan Unnikrishnan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02087v1  
  <details><summary>Abstract</summary>

  Background: Accurate glottal segmentation in high-speed videoendoscopy (HSV) is essential for extracting kinematic biomarkers of laryngeal function. However, existing deep learning models often produce spurious artifacts in non-glottal frames and fail to generalize across different clinical settings. Methods: We propose a detection-gated pipeline that integrates a YOLOv8-based detector with a U-Net segmenter. A temporal consistency wrapper ensures robustness by suppressing false positives during glottal closure and instrument occlusion. The model was trained on a limited subset of the GIRAFE dataset (600 frames) and evaluated via zero-shot transfer on the large-scale BAGLS dataset. Results: The pipeline achieved state-of-the-art performance on the GIRAFE benchmark (DSC 0.81) and demonstrated superior generalizability on BAGLS (DSC 0.85, in-distribution) without institutional fine-tuning. Downstream validation on a 65-subject clinical cohort confirmed that automated kinematic features (Open Quotient, coefficient of variation) remained consistent with established clinical benchmarks. The coefficient of variation (CV) of the glottal area was found to be a significant marker for distinguishing healthy from pathological vocal function (p=0.006). Conclusions: The detection-gated architecture provides a lightweight, computationally efficient solution (~35 frames/s) for real-time clinical use. By enabling robust zero-shot transfer, this framework facilitates the standardized, large-scale extraction of clinical biomarkers across diverse endoscopy platforms. Code, trained weights, and evaluation scripts are released at https://github.com/hari-krishnan/openglottal.

  </details>



- **From Pixels to Patches: Pooling Strategies for Earth Embeddings**  
  Isaac Corley, Caleb Robinson, Inbal Becker-Reshef, Juan M. Lavista Ferres  
  _2026-03-02_ · https://arxiv.org/abs/2603.02080v1  
  <details><summary>Abstract</summary>

  As geospatial foundation models shift from patch-level to pixel-level embeddings, practitioners must aggregate thousands of pixel vectors into patch representations that preserve class-discriminative signal while matching downstream label resolution. The default choice, mean pooling, discards within-patch variability and can drop accuracy by more than 10% under spatial shift. To evaluate this effect, we introduce EuroSAT-Embed: 81,000 embedding GeoTIFFs derived from three foundation models: AlphaEarth, OlmoEarth, and Tessera. We benchmark 11 training-free and 2 parametric pooling methods under both random and geographically disjoint test splits. Our results show that richer pooling schemes reduce the geographic generalization gap by up to 40% relative to mean pooling and increases accuracy by up to 5% on spatial splits. We recommend Generalized Mean Pooling (GeM) as a drop-in replacement for mean pooling: it improves accuracy without increasing embedding dimensionality. For maximum accuracy, Stats pooling (concatenation of min/max/mean/std pooling) performs best at 4x the embedding size. We further find that pooling effectiveness varies across embedding sources and that higher-dimensional embeddings benefit most from distributional statistics.

  </details>



- **MMNavAgent: Multi-Magnification WSI Navigation Agent for Clinically Consistent Whole-Slide Analysis**  
  Zhengyang Xu, Han Li, Jingsong Liu, Linrui Xie, Xun Ma, Xin You, Shihui Zu, Ayako Ito, Xinyu Hao, Hongming Xu, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.02079v1  
  <details><summary>Abstract</summary>

  Recent AI navigation approaches aim to improve Whole-Slide Image (WSI) diagnosis by modeling spatial exploration and selecting diagnostically relevant regions, yet most operate at a single fixed magnification or rely on predefined magnification traversal. In clinical practice, pathologists examine slides across multiple magnifications and selectively inspect only necessary scales, dynamically integrating global and cellular evidence in a sequential manner. This mismatch prevents existing methods from modeling cross-magnification interactions and adaptive magnification selection inherent to real diagnostic workflows. To these, we propose a clinically consistent Multi-Magnification WSI Navigation Agent (MMNavAgent) that explicitly models multi magnification interaction and adaptive magnification selection. Specifically, we introduce a Cross-Magnification navigation Tool (CMT) that aggregates contextual information from adjacent magnifications to enhance discriminative representations along the navigation path. We further introduce a Magnification Selection Tool (MST) that leverages memory-driven reasoning within the agent framework to enable interactive and adaptive magnification selection, mimicking the sequential decision process of pathologists. Extensive experiments on a public dataset demonstrate improved diagnostic performance, with 1.45% gain of AUC and 2.93% gain of BACC over a non-agent baseline. Code will be public upon acceptance.

  </details>



- **ORGAN: Object-Centric Representation Learning using Cycle Consistent Generative Adversarial Networks**  
  Joël Küchler, Ellen van Maren, Vaiva Vasiliauskaitė, Katarina Vulić, Reza Abbasi-Asl, Stephan J. Ihle  
  _2026-03-02_ · https://arxiv.org/abs/2603.02063v1  
  <details><summary>Abstract</summary>

  Although data generation is often straightforward, extracting information from data is more difficult. Object-centric representation learning can extract information from images in an unsupervised manner. It does so by segmenting an image into its subcomponents: the objects. Each object is then represented in a low-dimensional latent space that can be used for downstream processing. Object-centric representation learning is dominated by autoencoder architectures (AEs). Here, we present ORGAN, a novel approach for object-centric representation learning, which is based on cycle-consistent Generative Adversarial Networks instead. We show that it performs similarly to other state-of-the-art approaches on synthetic datasets, while at the same time being the only approach tested here capable of handling more challenging real-world datasets with many objects and low visual contrast. Complementing these results, ORGAN creates expressive latent space representations that allow for object manipulation. Finally, we show that ORGAN scales well both with respect to the number of objects and the size of the images, giving it a unique edge over current state-of-the-art approaches.

  </details>



- **NICO-RAG: Multimodal Hypergraph Retrieval-Augmented Generation for Understanding the Nicotine Public Health Crisis**  
  Manuel Serna-Aguilera, Raegan Anderes, Page Dobbs, Khoa Luu  
  _2026-03-02_ · https://arxiv.org/abs/2603.02047v1  
  <details><summary>Abstract</summary>

  The nicotine addiction public health crisis continues to be pervasive. In this century alone, the tobacco industry has released and marketed new products in an aggressive effort to lure new and young customers for life. Such innovations and product development, namely flavored nicotine or tobacco such as nicotine pouches, have undone years of anti-tobacco campaign work. Past work is limited both in scope and in its ability to connect large-scale data points. Thus, we introduce the Nicotine Innovation Counter-Offensive (NICO) Dataset to provide public health researchers with over 200,000 multimodal samples, including images and text descriptions, on 55 tobacco and nicotine product brands. In addition, to provide public health researchers with factual connections across a large-scale dataset, we propose NICO-RAG, a retrieval-augmented generation (RAG) framework that can retrieve image features without incurring the high-cost of language models, as well as the added cost of processing image tokens with large-scale datasets such as NICO. At construction time, NICO-RAG organizes image- and text-extracted entities and relations into hypergraphs to produce as factual responses as possible. This joint multimodal knowledge representation enables NICO-RAG to retrieve images for query answering not only by visual similarity but also by the semantic similarity of image descriptions. Experimentals show that without needing to process additional tokens from images for over 100 questions, NICO-RAG performs comparably to the state-of-the-art RAG method adapted for images.

  </details>



- **LAD-Drive: Bridging Language and Trajectory with Action-Aware Diffusion Transformers**  
  Fabian Schmidt, Karol Fedurko, Markus Enzweiler, Abhinav Valada  
  _2026-03-02_ · https://arxiv.org/abs/2603.02035v1  
  <details><summary>Abstract</summary>

  While multimodal large language models (MLLMs) provide advanced reasoning for autonomous driving, translating their discrete semantic knowledge into continuous trajectories remains a fundamental challenge. Existing methods often rely on unimodal planning heads that inherently limit their ability to represent multimodal driving behavior. Furthermore, most generative approaches frequently condition on one-hot encoded actions, discarding the nuanced navigational uncertainty critical for complex scenarios. To resolve these limitations, we introduce LAD-Drive, a generative framework that structurally disentangles high-level intention from low-level spatial planning. LAD-Drive employs an action decoder to infer a probabilistic meta-action distribution, establishing an explicit belief state that preserves the nuanced intent typically lost by one-hot encodings. This distribution, fused with the vehicle's kinematic state, conditions an action-aware diffusion decoder that utilizes a truncated denoising process to refine learned motion anchors into safe, kinematically feasible trajectories. Extensive evaluations on the LangAuto benchmark demonstrate that LAD-Drive achieves state-of-the-art results, outperforming competitive baselines by up to 59% in Driving Score while significantly reducing route deviations and collisions. We will publicly release the code and models on https://github.com/iis-esslingen/lad-drive.

  </details>



- **MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning**  
  Jiachun Li, Shaoping Huang, Zhuoran Jin, Chenlong Zhang, Pengfei Cao, Yubo Chen, Kang Liu, Jun Zhao  
  _2026-03-02_ · https://arxiv.org/abs/2603.02024v1  
  <details><summary>Abstract</summary>

  Recent progress in the reasoning capabilities of multimodal large language models (MLLMs) has empowered them to address more complex tasks such as scientific analysis and mathematical reasoning. Despite their promise, MLLMs' reasoning abilities across different scenarios in real life remain largely unexplored and lack standardized benchmarks for evaluation. To address this gap, we introduce MMR-Life, a comprehensive benchmark designed to evaluate the diverse multimodal multi-image reasoning capabilities of MLLMs across real-life scenarios. MMR-Life consists of 2,646 multiple-choice questions based on 19,108 images primarily sourced from real-world contexts, comprehensively covering seven reasoning types: abductive, analogical, causal, deductive, inductive, spatial, and temporal. Unlike existing reasoning benchmarks, MMR-Life does not rely on domain-specific expertise but instead requires models to integrate information across multiple images and apply diverse reasoning abilities. The evaluation of 37 advanced models highlights the substantial challenge posed by MMR-Life. Even top models like GPT-5 achieve only 58% accuracy and display considerable variance in performance across reasoning types. Moreover, we analyze the reasoning paradigms of existing MLLMs, exploring how factors such as thinking length, reasoning method, and reasoning type affect their performance. In summary, MMR-Life establishes a comprehensive foundation for evaluating, analyzing, and improving the next generation of multimodal reasoning systems.

  </details>



- **MAP-Diff: Multi-Anchor Guided Diffusion for Progressive 3D Whole-Body Low-Dose PET Denoising**  
  Peiyuan Jing, Chun-Wun Cheng, Liutao Yang, Zhenxuan Zhang, Thiago V. Lima, Klaus Strobel, Antoine Leimgruber, Angelica Aviles-Rivero, Guang Yang, Javier A. Montoya-Zegarra  
  _2026-03-02_ · https://arxiv.org/abs/2603.02012v1  
  <details><summary>Abstract</summary>

  Low-dose Positron Emission Tomography (PET) reduces radiation exposure but suffers from severe noise and quantitative degradation. Diffusion-based denoising models achieve strong final reconstructions, yet their reverse trajectories are typically unconstrained and not aligned with the progressive nature of PET dose formation. We propose MAP-Diff, a multi-anchor guided diffusion framework for progressive 3D whole-body PET denoising. MAP-Diff introduces clinically observed intermediate-dose scans as trajectory anchors and enforces timestep-dependent supervision to regularize the reverse process toward dose-aligned intermediate states. Anchor timesteps are calibrated via degradation matching between simulated diffusion corruption and real multi-dose PET pairs, and a timestep-weighted anchor loss stabilizes stage-wise learning. At inference, the model requires only ultra-low-dose input while enabling progressive, dose-consistent intermediate restoration. Experiments on internal (Siemens Biograph Vision Quadra) and cross-scanner (United Imaging uEXPLORER) datasets show consistent improvements over strong CNN-, Transformer-, GAN-, and diffusion-based baselines. On the internal dataset, MAP-Diff improves PSNR from 42.48 dB to 43.71 dB (+1.23 dB), increases SSIM to 0.986, and reduces NMAE from 0.115 to 0.103 (-0.012) compared to 3D DDPM. Performance gains generalize across scanners, achieving 34.42 dB PSNR and 0.141 NMAE on the external cohort, outperforming all competing methods.

  </details>



- **CHOP: Counterfactual Human Preference Labels Improve Obstacle Avoidance in Visuomotor Navigation Policies**  
  Gershom Seneviratne, Jianyu An, Vaibhav Shende, Sahire Ellahy, Yaxita Amin, Kondapi Manasanjani, Samarth Chopra, Jonathan Deepak Kannan, Dinesh Manocha  
  _2026-03-02_ · https://arxiv.org/abs/2603.02004v1  
  <details><summary>Abstract</summary>

  Visuomotor navigation policies have shown strong perception-action coupling for embodied agents, yet they often struggle with safe navigation and dynamic obstacle avoidance in complex real-world environments. We introduce CHOP, a novel approach that leverages Counterfactual Human Preference Labels to align visuomotor navigation policies towards human intuition of safety and obstacle avoidance in navigation. In CHOP, for each visual observation, the robot's executed trajectory is included among a set of counterfactual navigation trajectories: alternative trajectories the robot could have followed under identical conditions. Human annotators provide pairwise preference labels over these trajectories based on anticipated outcomes such as collision risk and path efficiency. These aggregated preferences are then used to fine-tune visuomotor navigation policies, aligning their behavior with human preferences in navigation. Experiments on the SCAND dataset show that visuomotor navigation policies fine-tuned with CHOP reduce near-collision events by 49.7%, decrease deviation from human-preferred trajectories by 45.0%, and increase average obstacle clearance by 19.8% on average across multiple state-of-the-art models, compared to their pretrained baselines. These improvements transfer to real-world deployments on a Ghost Robotics Vision60 quadruped, where CHOP-aligned policies improve average goal success rates by 24.4%, increase minimum obstacle clearance by 6.8%, reduce collision and intervention events by 45.7%, and improve normalized path completion by 38.6% on average across navigation scenarios, compared to their pretrained baselines. Our results highlight the value of counterfactual preference supervision in bridging the gap between large-scale visuomotor policies and human-aligned, safety-aware embodied navigation.

  </details>



- **Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering**  
  Hari Prasanth S. M., Pejman Habibiroudkenar, Eerik Alamikkotervo, Dimitrios Bouzoulas, Risto Ojala  
  _2026-03-02_ · https://arxiv.org/abs/2603.01997v1  
  <details><summary>Abstract</summary>

  Event cameras provide high-temporal-resolution visual sensing that is well suited for observing fast-moving aerial objects; however, their use for drone trajectory prediction remains limited. This work introduces an event-only drone forecasting method that exploits propeller-induced motion cues. Propeller rotational speed are extracted directly from raw event data and fused within an RPM-aware Kalman filtering framework. Evaluations on the FRED dataset show that the proposed method outperforms learning-based approaches and vanilla kalman filter in terms of average distance error and final distance error at 0.4s and 0.8s forecasting horizons. The results demonstrate robust and accurate short- and medium-horizon trajectory forecasting without reliance on RGB imagery or training data.

  </details>



- **Process Over Outcome: Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection**  
  Yuchen Zhang, Yaxiong Wang, Kecheng Han, Yujiao Wu, Lianwei Wu, Li Zhu, Zhedong Zheng  
  _2026-03-02_ · https://arxiv.org/abs/2603.01993v1  
  <details><summary>Abstract</summary>

  Recent advances in generative AI have significantly enhanced the realism of multimodal media manipulation, thereby posing substantial challenges to manipulation detection. Existing manipulation detection and grounding approaches predominantly focus on manipulation type classification under result-oriented supervision, which not only lacks interpretability but also tends to overfit superficial artifacts. In this paper, we argue that generalizable detection requires incorporating explicit forensic reasoning, rather than merely classifying a limited set of manipulation types, which fails to generalize to unseen manipulation patterns. To this end, we propose REFORM, a reasoning-driven framework that shifts learning from outcome fitting to process modeling. REFORM adopts a three-stage curriculum that first induces forensic rationales, then aligns reasoning with final judgments, and finally refines logical consistency via reinforcement learning. To support this paradigm, we introduce ROM, a large-scale dataset with rich reasoning annotations. Extensive experiments show that REFORM establishes new state-of-the-art performance with superior generalization, achieving 81.52% ACC on ROM, 76.65% ACC on DGM4, and 74.9 F1 on MMFakeBench.

  </details>



- **According to Me: Long-Term Personalized Referential Memory QA**  
  Jingbiao Mei, Jinghong Chen, Guangyu Yang, Xinyu Hou, Margaret Li, Bill Byrne  
  _2026-03-02_ · https://arxiv.org/abs/2603.01990v1  
  <details><summary>Abstract</summary>

  Personalized AI assistants must recall and reason over long-term user memory, which naturally spans multiple modalities and sources such as images, videos, and emails. However, existing Long-term Memory benchmarks focus primarily on dialogue history, failing to capture realistic personalized references grounded in lived experience. We introduce ATM-Bench, the first benchmark for multimodal, multi-source personalized referential Memory QA. ATM-Bench contains approximately four years of privacy-preserving personal memory data and human-annotated question-answer pairs with ground-truth memory evidence, including queries that require resolving personal references, multi-evidence reasoning from multi-source and handling conflicting evidence. We propose Schema-Guided Memory (SGM) to structurally represent memory items originated from different sources. In experiments, we implement 5 state-of-the-art memory systems along with a standard RAG baseline and evaluate variants with different memory ingestion, retrieval, and answer generation techniques. We find poor performance (under 20\% accuracy) on the ATM-Bench-Hard set, and that SGM improves performance over Descriptive Memory commonly adopted in prior works. Code available at: https://github.com/JingbiaoMei/ATM-Bench

  </details>



- **Semantic Similarity is a Spurious Measure of Comic Understanding: Lessons Learned from Hallucinations in a Benchmarking Experiment**  
  Christopher Driggers-Ellis, Nachiketh Tibrewal, Rohit Bogulla, Harsh Khanna, Sangpil Youm, Christan Grant, Bonnie Dorr  
  _2026-03-02_ · https://arxiv.org/abs/2603.01950v1  
  <details><summary>Abstract</summary>

  A system that enables blind or visually impaired users to access comics/manga would introduce a new medium of storytelling to this community. However, no such system currently exists. Generative vision-language models (VLMs) have shown promise in describing images and understanding comics, but most research on comic understanding is limited to panel-level analysis. To fully support blind and visually impaired users, greater attention must be paid to page-level understanding and interpretation. In this work, we present a preliminary benchmark of VLM performance on comic interpretation tasks. We identify and categorize hallucinations that emerge during this process, organizing them into generalized object-hallucination taxonomies. We conclude with guidance on future research, emphasizing hallucination mitigation and improved data curation for comic interpretation.

  </details>



- **MobileMold: A Smartphone-Based Microscopy Dataset for Food Mold Detection**  
  Dinh Nam Pham, Leonard Prokisch, Bennet Meyer, Jonas Thumbs  
  _2026-03-02_ · https://arxiv.org/abs/2603.01944v1  
  <details><summary>Abstract</summary>

  Smartphone clip-on microscopes turn everyday devices into low-cost, portable imaging systems that can even reveal fungal structures at the microscopic level, enabling mold inspection beyond unaided visual checks. In this paper, we introduce MobileMold, an open smartphone-based microscopy dataset for food mold detection and food classification. MobileMold contains 4,941 handheld microscopy images spanning 11 food types, 4 smartphones, 3 microscopes, and diverse real-world conditions. Beyond the dataset release, we establish baselines for (i) mold detection and (ii) food-type classification, including a multi-task setting that predicts both attributes. Across multiple pretrained deep learning architectures and augmentation strategies, we obtain near-ceiling performance (accuracy = 0.9954, F1 = 0.9954, MCC = 0.9907), validating the utility of our dataset for detecting food spoilage. To increase transparency, we complement our evaluation with saliency-based visual explanations highlighting mold regions associated with the model's predictions. MobileMold aims to contribute to research on accessible food-safety sensing, mobile imaging, and exploring the potential of smartphones enhanced with attachments.

  </details>



- **BAWSeg: A UAV Multispectral Benchmark for Barley Weed Segmentation**  
  Haitian Wang, Xinyu Wang, Muhammad Ibrahim, Dustin Severtson, Ajmal Mian  
  _2026-03-02_ · https://arxiv.org/abs/2603.01932v1  
  <details><summary>Abstract</summary>

  Accurate weed mapping in cereal fields requires pixel-level segmentation from UAV imagery that remains reliable across fields, seasons, and illumination. Existing multispectral pipelines often depend on thresholded vegetation indices, which are brittle under radiometric drift and mixed crop--weed pixels, or on single-stream CNN and Transformer backbones that ingest stacked bands and indices, where radiance cues and normalized index cues interfere and reduce sensitivity to small weed clusters embedded in crop canopies. We propose VISA (Vegetation-Index and Spectral Attention), a two-stream segmentation network that decouples these cues and fuses them at native resolution. The radiance stream learns from calibrated five-band reflectance using residual spectral-spatial attention to preserve fine textures and row boundaries that are attenuated by ratio indices. The index stream operates on vegetation-index maps with windowed self-attention to model local structure efficiently, state-space layers to propagate field-scale context without quadratic attention cost, and Slot Attention to form stable region descriptors that improve discrimination of sparse weeds under canopy mixing. To support supervised training and deployment-oriented evaluation, we introduce BAWSeg, a four-year UAV multispectral dataset collected over commercial barley paddocks in Western Australia, providing radiometrically calibrated blue, green, red, red edge, and near-infrared orthomosaics, derived vegetation indices, and dense crop, weed, and other labels with leakage-free block splits. On BAWSeg, VISA achieves 75.6% mIoU and 63.5% weed IoU with 22.8M parameters, outperforming a multispectral SegFormer-B1 baseline by 1.2 mIoU and 1.9 weed IoU. Under cross-plot and cross-year protocols, VISA maintains 71.2% and 69.2% mIoU, respectively. The BAWSeg data, VISA code, and trained models will be released upon publication.

  </details>



- **MealRec: Multi-granularity Sequential Modeling via Hierarchical Diffusion Models for Micro-Video Recommendation**  
  Xinxin Dong, Haokai Ma, Yuze Zheng, Yongfu Zha, Yonghui Yang, Xiaodong Wang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01926v1  
  <details><summary>Abstract</summary>

  Micro-video recommendation aims to capture user preferences from the collaborative and context information of the interacted micro-videos, thereby predicting the appropriate videos. This target is often hindered by the inherent noise within multimodal content and unreliable implicit feedback, which weakens the correspondence between behaviors and underlying interests. While conventional works have predominantly approached such scenario through behavior-augmented modeling and content-centric multimodal analysis, these paradigms can inadvertently give rise to two non-trivial challenges: preference-irrelative video representation extraction and inherent modality conflicts. To address these issues, we propose a Multi-granularity sequential modeling method via hierarchical diffusion models for micro-video Recommendation (MealRec), which simultaneously considers temporal correlations during preference modeling from intra- and inter-video perspectives. Specifically, we first propose Temporal-guided Content Diffusion (TCD) to refine video representations under intra-video temporal guidance and personalized collaborative signals to emphasize salient content while suppressing redundancy. To achieve the semantically coherent preference modeling, we further design the Noise-unconditional Preference Denoising (NPD) to recovers informative user preferences from corrupted states under the blind denoising. Extensive experiments and analyses on four micro-video datasets from two platforms demonstrate the effectiveness, universality, and robustness of our MealRec, further uncovering the effective mechanism of our proposed TCD and NPD. The source code and corresponding dataset will be available upon acceptance.

  </details>



- **Generative Visual Chain-of-Thought for Image Editing**  
  Zijin Yin, Tiankai Hang, Yiji Cheng, Shiyi Zhang, Runze He, Yu Xu, Chunyu Wang, Bing Li, Zheng Chang, Kongming Liang, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01893v1  
  <details><summary>Abstract</summary>

  Existing image editing methods struggle to perceive where to edit, especially under complex scenes and nuanced spatial instructions. To address this issue, we propose Generative Visual Chain-of-Thought (GVCoT), a unified framework that performs native visual reasoning by first generating spatial cues to localize the target region and then executing the edit. Unlike prior text-only CoT or tool-dependent visual CoT paradigms, GVCoT jointly optimizes visual tokens generated during the reasoning and editing phases in an end-to-end manner. This way fosters the emergence of innate spatial reasoning ability and enables more effective utilization of visual-domain cues. The main challenge of training GCVoT lies in the scarcity of large-scale editing data with precise edit region annotations; to this end, we construct GVCoT-Edit-Instruct, a dataset of 1.8M high-quality samples spanning 19 tasks. We adopt a progressive training strategy: supervised fine-tuning to build foundational localization ability in reasoning trace before final editing, followed by reinforcement learning to further improve reasoning and editing quality. Finally, we introduce SREdit-Bench, a new benchmark designed to comprehensively stress-test models under sophisticated scenes and fine-grained referring expressions. Experiments demonstrate that GVCoT consistently outperforms state-of-the-art models on SREdit-Bench and ImgEdit. We hope our GVCoT will inspire future research toward interpretable and precise image editing.

  </details>



- **CTForensics: A Comprehensive Dataset and Method for AI-Generated CT Image Detection**  
  Yiheng Li, Zichang Tan, Guoqing Xu, Yijun Ye, Yang Yang, Zhen Lei  
  _2026-03-02_ · https://arxiv.org/abs/2603.01878v1  
  <details><summary>Abstract</summary>

  With the rapid development of generative AI in medical imaging, synthetic Computed Tomography (CT) images have demonstrated great potential in applications such as data augmentation and clinical diagnosis, but they also introduce serious security risks. Despite the increasing security concerns, existing studies on CT forgery detection are still limited and fail to adequately address real-world challenges. These limitations are mainly reflected in two aspects: the absence of datasets that can effectively evaluate model generalization to reflect the real-world application requirements, and the reliance on detection methods designed for natural images that are insensitive to CT-specific forgery artifacts. In this view, we propose CTForensics, a comprehensive dataset designed to systematically evaluate the generalization capability of CT forgery detection methods, which includes ten diverse CT generative methods. Moreover, we introduce the Enhanced Spatial-Frequency CT Forgery Detector (ESF-CTFD), an efficient CNN-based neural network that captures forgery cues across the wavelet, spatial, and frequency domains. First, it transforms the input CT image into three scales and extracts features at each scale via the Wavelet-Enhanced Central Stem. Then, starting from the largest-scale features, the Spatial Process Block gradually performs feature fusion with the smaller-scale ones. Finally, the Frequency Process Block learns frequency-domain information for predicting the final results. Experiments demonstrate that ESF-CTFD consistently outperforms existing methods and exhibits superior generalization across different CT generative models.

  </details>



- **FireRed-OCR Technical Report**  
  Hao Wu, Haoran Lou, Xinyue Li, Zuodong Zhong, Zhaojun Sun, Phellon Chen, Xuanhe Zhou, Kai Zuo, Yibo Chen, Xu Tang, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01840v1  
  <details><summary>Abstract</summary>

  We present FireRed-OCR, a systematic framework to specialize general VLMs into high-performance OCR models. Large Vision-Language Models (VLMs) have demonstrated impressive general capabilities but frequently suffer from ``structural hallucination'' when processing complex documents, limiting their utility in industrial OCR applications. In this paper, we introduce FireRed-OCR, a novel framework designed to transform general-purpose VLMs (based on Qwen3-VL) into pixel-precise structural document parsing experts. To address the scarcity of high-quality structured data, we construct a ``Geometry + Semantics'' Data Factory. Unlike traditional random sampling, our pipeline leverages geometric feature clustering and multi-dimensional tagging to synthesize and curate a highly balanced dataset, effectively handling long-tail layouts and rare document types. Furthermore, we propose a Three-Stage Progressive Training strategy that guides the model from pixel-level perception to logical structure generation. This curriculum includes: (1) Multi-task Pre-alignment to ground the model's understanding of document structure; (2) Specialized SFT for standardizing full-image Markdown output; and (3) Format-Constrained Group Relative Policy Optimization (GRPO), which utilizes reinforcement learning to enforce strict syntactic validity and structural integrity (e.g., table closure, formula syntax). Extensive evaluations on OmniDocBench v1.5 demonstrate that FireRed-OCR achieves state-of-the-art performance with an overall score of 92.94\%, significantly outperforming strong baselines such as DeepSeek-OCR 2 and OCRVerse across text, formula, table, and reading order metrics. We open-source our code and model weights to facilitate the ``General VLM to Specialized Structural Expert'' paradigm.

  </details>



- **Downstream Task Inspired Underwater Image Enhancement: A Perception-Aware Study from Dataset Construction to Network Design**  
  Bosen Lin, Feng Gao, Yanwei Yu, Junyu Dong, Qian Du  
  _2026-03-02_ · https://arxiv.org/abs/2603.01767v1  
  <details><summary>Abstract</summary>

  In real underwater environments, downstream image recognition tasks such as semantic segmentation and object detection often face challenges posed by problems like blurring and color inconsistencies. Underwater image enhancement (UIE) has emerged as a promising preprocessing approach, aiming to improve the recognizability of targets in underwater images. However, most existing UIE methods mainly focus on enhancing images for human visual perception, frequently failing to reconstruct high-frequency details that are critical for task-specific recognition. To address this issue, we propose a Downstream Task-Inspired Underwater Image Enhancement (DTI-UIE) framework, which leverages human visual perception model to enhance images effectively for underwater vision tasks. Specifically, we design an efficient two-branch network with task-aware attention module for feature mixing. The network benefits from a multi-stage training framework and a task-driven perceptual loss. Additionally, inspired by human perception, we automatically construct a Task-Inspired UIE Dataset (TI-UIED) using various task-specific networks. Experimental results demonstrate that DTI-UIE significantly improves task performance by generating preprocessed images that are beneficial for downstream tasks such as semantic segmentation, object detection, and instance segmentation. The codes are publicly available at https://github.com/oucailab/DTIUIE.

  </details>



- **An Analysis of Multi-Task Architectures for the Hierarchic Multi-Label Problem of Vehicle Model and Make Classification**  
  Alexandru Manole, Laura Diosan  
  _2026-03-02_ · https://arxiv.org/abs/2603.01746v1  
  <details><summary>Abstract</summary>

  Most information in our world is organized hierarchically; however, many Deep Learning approaches do not leverage this semantically rich structure. Research suggests that human learning benefits from exploiting the hierarchical structure of information, and intelligent models could similarly take advantage of this through multi-task learning. In this work, we analyze the advantages and limitations of multi-task learning in a hierarchical multi-label classification problem: car make and model classification. Considering both parallel and cascaded multi-task architectures, we evaluate their impact on different Deep Learning classifiers (CNNs, Transformers) while varying key factors such as dropout rate and loss weighting to gain deeper insight into the effectiveness of this approach. The tests are conducted on two established benchmarks: StanfordCars and CompCars. We observe the effectiveness of the multi-task paradigm on both datasets, improving the performance of the investigated CNN in almost all scenarios. Furthermore, the approach yields significant improvements on the CompCars dataset for both types of models.

  </details>



- **Action-Guided Attention for Video Action Anticipation**  
  Tsung-Ming Tai, Sofia Casarin, Andrea Pilzer, Werner Nutt, Oswald Lanz  
  _2026-03-02_ · https://arxiv.org/abs/2603.01743v1  
  <details><summary>Abstract</summary>

  Anticipating future actions in videos is challenging, as the observed frames provide only evidence of past activities, requiring the inference of latent intentions to predict upcoming actions. Existing transformer-based approaches, which rely on dot-product attention over pixel representations, often lack the high-level semantics necessary to model video sequences for effective action anticipation. As a result, these methods tend to overfit to explicit visual cues present in the past frames, limiting their ability to capture underlying intentions and degrading generalization to unseen samples. To address this, we propose Action-Guided Attention (AGA), an attention mechanism that explicitly leverages predicted action sequences as queries and keys to guide sequence modeling. Our approach fosters the attention module to emphasize relevant moments from the past based on the upcoming activity and combine this information with the current frame embedding via a dedicated gating function. The design of AGA enables post-training analysis of the knowledge discovered from the training set. Experiments on the widely adopted EPIC-Kitchens-100 benchmark demonstrate that AGA generalizes well from validation to unseen test sets. Post-training analysis can further examine the action dependencies captured by the model and the counterfactual evidence it has internalized, offering transparent and interpretable insights into its anticipative predictions.

  </details>



- **Preoperative-to-intraoperative Liver Registration for Laparoscopic Surgery via Latent-Grounded Correspondence Constraints**  
  Ruize Cui, Jialun Pei, Haiqiao Wang, Jun Zhou, Jeremy Yuen-Chun Teoh, Pheng-Ann Heng, Jing Qin  
  _2026-03-02_ · https://arxiv.org/abs/2603.01720v1  
  <details><summary>Abstract</summary>

  In laparoscopic liver surgery, augmented reality technology enhances intraoperative anatomical guidance by overlaying 3D liver models from preoperative CT/MRI onto laparoscopic 2D views. However, existing registration methods lack explicit modeling of reliable 2D-3D geometric correspondences supported by latent evidence, leading to limited interpretability and potentially unstable alignment in clinical scenarios. In this work, we introduce Land-Reg, a correspondence-driven deformable registration framework that explicitly learns latent-grounded 2D-3D landmark correspondences as an interpretable intermediate representation to bridge cross-modal alignment. For rigid registration, Land-Reg embraces a Cross-modal Latent Alignment module to map multi-modal features into a unified latent space. Further, an Uncertainty-enhanced Overlap Landmark Detector with similarity matching is proposed to robustly estimate explicit 2D-3D landmark correspondences. For non-rigid registration, we design a novel shape-constrained supervision strategy that anchors shape deformation to matched landmarks through reprojection consistency and incorporates local-isometric regularization to alleviate inherent 2D-3D depth ambiguity, while a rendered-mask alignment enforces global shape consistency. Experimental results on the P2ILF dataset demonstrate the superiority of our method on both rigid pose estimation and non-rigid deformation. Our code will be available at https://github.com/cuiruize/Land-Reg.

  </details>



- **Dual Distillation for Few-Shot Anomaly Detection**  
  Le Dong, Qinzhong Tan, Chunlei Li, Jingliang Hu, Yilei Shi, Weisheng Dong, Xiao Xiang Zhu, Lichao Mou  
  _2026-03-02_ · https://arxiv.org/abs/2603.01713v1  
  <details><summary>Abstract</summary>

  Anomaly detection is a critical task in computer vision with profound implications for medical imaging, where identifying pathologies early can directly impact patient outcomes. While recent unsupervised anomaly detection approaches show promise, they require substantial normal training data and struggle to generalize across anatomical contexts. We introduce D$^2$4FAD, a novel dual distillation framework for few-shot anomaly detection that identifies anomalies in previously unseen tasks using only a small number of normal reference images. Our approach leverages a pre-trained encoder as a teacher network to extract multi-scale features from both support and query images, while a student decoder learns to distill knowledge from the teacher on query images and self-distill on support images. We further propose a learn-to-weight mechanism that dynamically assesses the reference value of each support image conditioned on the query, optimizing anomaly detection performance. To evaluate our method, we curate a comprehensive benchmark dataset comprising 13,084 images across four organs, four imaging modalities, and five disease categories. Extensive experiments demonstrate that D$^2$4FAD significantly outperforms existing approaches, establishing a new state-of-the-art in few-shot medical anomaly detection. Code is available at https://github.com/ttttqz/D24FAD.

  </details>



- **Towards Principled Dataset Distillation: A Spectral Distribution Perspective**  
  Ruixi Wu, Shaobo Wang, Jiahuan Chen, Zhiyuan Liu, Yicun Yang, Zhaorun Chen, Zekai Li, Kaixin Li, Xinming Wang, Hongzhu Yi, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01698v1  
  <details><summary>Abstract</summary>

  Dataset distillation (DD) aims to compress large-scale datasets into compact synthetic counterparts for efficient model training. However, existing DD methods exhibit substantial performance degradation on long-tailed datasets. We identify two fundamental challenges: heuristic design choices for distribution discrepancy measure and uniform treatment of imbalanced classes. To address these limitations, we propose Class-Aware Spectral Distribution Matching (CSDM), which reformulates distribution alignment via the spectrum of a well-behaved kernel function. This technique maps the original samples into frequency space, resulting in the Spectral Distribution Distance (SDD). To mitigate class imbalance, we exploit the unified form of SDD to perform amplitude-phase decomposition, which adaptively prioritizes the realism in tail classes. On CIFAR-10-LT, with 10 images per class, CSDM achieves a 14.0% improvement over state-of-the-art DD methods, with only a 5.7% performance drop when the number of images in tail classes decreases from 500 to 25, demonstrating strong stability on long-tailed data.

  </details>



- **Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning**  
  Haonan Jia, Shichao Dong, Xin Dong, Zenghui Sun, Jin Wang, Jinsong Lan, Xiaoyong Zhu, Bo Zheng, Kaifu Zhang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01696v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (LVLMs) often omit or misrepresent critical visual content in generated image captions. Minimizing such information loss will force LVLMs to focus on image details to generate precise descriptions. However, measuring information loss during modality conversion is inherently challenging due to the modal gap between visual content and text output. In this paper, we argue that the quality of an image caption is positively correlated with the similarity between images retrieved via text search using that caption. Based on this insight, we further propose Cross-modal Identity Mapping (CIM), a reinforcement learning framework that enhances image captioning without requiring additional annotations. Specifically, the method quantitatively evaluates the information loss from two perspectives: Gallery Representation Consistency and Query-gallery Image Relevance. Supervised under these metrics, LVLM minimizes information loss and aims to achieve identity mapping from images to captions. The experimental results demonstrate the superior performance of our method in image captioning, even when compared with Supervised Fine-Tuning. Particularly, on the COCO-LN500 benchmark, CIM achieves a 20% improvement in relation reasoning on Qwen2.5-VL-7B.The code will be released when the paper is accepted.

  </details>



- **DriveCombo: Benchmarking Compositional Traffic Rule Reasoning in Autonomous Driving**  
  Enhui Ma, Jiahuan Zhang, Guantian Zheng, Tao Tang, Shengbo Eben Li, Yuhang Lu, Xia Zhou, Xueyang Zhang, Yifei Zhan, Kun Zhan, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01637v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) are rapidly becoming the intelligence brain of end-to-end autonomous driving systems. A key challenge is to assess whether MLLMs can truly understand and follow complex real-world traffic rules. However, existing benchmarks mainly focus on single-rule scenarios like traffic sign recognition, neglecting the complexity of multi-rule concurrency and conflicts in real driving. Consequently, models perform well on simple tasks but often fail or violate rules in real world complex situations. To bridge this gap, we propose DriveCombo, a text and vision-based benchmark for compositional traffic rule reasoning. Inspired by human drivers' cognitive development, we propose a systematic Five-Level Cognitive Ladder that evaluates reasoning from single-rule understanding to multi-rule integration and conflict resolution, enabling quantitative assessment across cognitive stages. We further propose a Rule2Scene Agent that maps language-based traffic rules to dynamic driving scenes through rule crafting and scene generation, enabling scene-level traffic rule visual reasoning. Evaluations of 14 mainstream MLLMs reveal performance drops as task complexity grows, particularly during rule conflicts. After splitting the dataset and fine-tuning on the training set, we further observe substantial improvements in both traffic rule reasoning and downstream planning capabilities. These results highlight the effectiveness of DriveCombo in advancing compliant and intelligent autonomous driving systems.

  </details>



- **Coarse-to-Fine Monocular Re-Localization in OpenStreetMap via Semantic Alignment**  
  Yuchen Zou, Xiao Hu, Dexing Zhong, Yuqing Tang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01613v1  
  <details><summary>Abstract</summary>

  Monocular re-localization plays a crucial role in enabling intelligent agents to achieve human-like perception. However, traditional methods rely on dense maps, which face scalability limitations and privacy risks. OpenStreetMap (OSM), as a lightweight map that protects privacy, offers semantic and geometric information with global scalability. Nonetheless, there are still challenges in using OSM for localization: the inherent cross-modal discrepancies between natural images and OSM, as well as the high computational cost of global map-based localization. In this paper, we propose a hierarchical search framework with semantic alignment for localization in OSM. First, the semantic awareness capability of DINO-ViT is utilised to deconstruct visual elements to establish semantic relationships with OSM. Second, a coarse-to-fine search paradigm is designed to replace global dense matching, enabling efficient progressive refinement. Extensive experiments demonstrate that our method significantly improves both localization accuracy and speed. When trained on a single dataset, the 3° orientation recall of our method even outperforms the 5° recall of state-of-the-art methods.

  </details>



- **InterCoG: Towards Spatially Precise Image Editing with Interleaved Chain-of-Grounding Reasoning**  
  Yecong Wan, Fan Li, Chunwei Wang, Hao Wu, Mingwen Shao, Wangmeng Zuo  
  _2026-03-02_ · https://arxiv.org/abs/2603.01586v1  
  <details><summary>Abstract</summary>

  Emerging unified editing models have demonstrated strong capabilities in general object editing tasks. However, it remains a significant challenge to perform fine-grained editing in complex multi-entity scenes, particularly those where targets are not visually salient and require spatial reasoning. To this end, we propose InterCoG, a novel text-vision Interleaved Chain-of-Grounding reasoning framework for fine-grained image editing in complex real-world scenes. The key insight of InterCoG is to first perform object position reasoning solely within text that includes spatial relation details to explicitly deduce the location and identity of the edited target. It then conducts visual grounding via highlighting the editing targets with generated bounding boxes and masks in pixel space, and finally rewrites the editing description to specify the intended outcomes. To further facilitate this paradigm, we propose two auxiliary training modules: multimodal grounding reconstruction supervision and multimodal grounding reasoning alignment to enforce spatial localization accuracy and reasoning interpretability, respectively. We also construct GroundEdit-45K, a dataset comprising 45K grounding-oriented editing samples with detailed reasoning annotations, and GroundEdit-Bench for grounding-aware editing evaluation. Extensive experiments substantiate the superiority of our approach in highly precise edits under spatially intricate and multi-entity scenes.

  </details>



- **Cryo-Bench: Benchmarking Foundation Models for Cryosphere Applications**  
  Saurabh Kaushik, Lalit Maurya, Beth Tellman  
  _2026-03-02_ · https://arxiv.org/abs/2603.01576v1  
  <details><summary>Abstract</summary>

  Geo-Foundation Models (GFMs) have been evaluated across diverse Earth observation task including multiple domains and have demonstrated strong potential of producing reliable maps even with sparse labels. However, benchmarking GFMs for Cryosphere applications has remained limited, primarily due to the lack of suitable evaluation datasets. To address this gap, we introduce \textbf{Cryo-Bench}, a benchmark compiled to evaluate GFM performance across key Cryospheric components. Cryo-Bench includes debris-covered glaciers, glacial lakes, sea ice, and calving fronts, spanning multiple sensors and broad geographic regions. We evaluate 14 GFMs alongside UNet and ViT baselines to assess their advantages, limitations, and optimal usage strategies. With a frozen encoder, UNet achieves the highest average mIoU of \textbf{66.38}, followed by TerraMind at \textbf{64.02} across five evluation dataset included in Cryo-Bench. In the few-shot setting (10\% input data), GFMs such as DOFA and TerraMind outperform UNet, achieving mIoU scores of \textbf{59.53}, \textbf{56.62}, and \textbf{56.60}, respectively, comapred to U-Net's 56.60. When fully finetuning GFMs, we observe inconsistent performance across datasets and models. However, tuning learning rate along with finetuning substantially improves GFM performance. For example, evaluation on two representative datasets (GLID and CaFFe) shows an average relative improvement of \textbf{12.77\%}. Despite having minimal Cryosphere representation in their pretraining data, GFMs exhibit notable domain adaptation capabilities and produce meaningful results across tasks. Based on our findings, We recommend encoder fine-tuning with hyperparameter optimization optimization to achieve the best possible performance, while using frozen encoders when users need quick results without extensive experimentation.(\href{https://github.com/Sk-2103/Cryo-Bench}{GitHub}).

  </details>



- **TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding**  
  Muhammet Esat Kalfaoglu, Halil Ibrahim Ozturk, Ozsel Kilinc, Alptekin Temizel  
  _2026-03-02_ · https://arxiv.org/abs/2603.01558v1  
  <details><summary>Abstract</summary>

  Mask-based paradigms for road topology understanding, such as TopoMaskV2, offer a complementary alternative to query-based methods by generating centerlines via a dense rasterized intermediate representation. However, prior work was limited to 2D predictions and suffered from severe discretization artifacts, necessitating fusion with parametric heads. We introduce TopoMaskV3, which advances this pipeline into a robust, standalone 3D predictor via two novel dense prediction heads: a dense offset field for sub-grid discretization correction within the existing BEV resolution, and a dense height map for direct 3D estimation. Beyond the architecture, we are the first to address geographic data leakage in road topology evaluation by introducing (1) geographically distinct splits to prevent memorization and ensure fair generalization, and (2) a long-range (+/-100 m) benchmark. TopoMaskV3 achieves state-of-the-art 28.5 OLS on this geographically disjoint benchmark, surpassing all prior methods. Our analysis shows that the mask representation is more robust to geographic overfitting than Bezier, while LiDAR fusion is most beneficial at long range and exhibits larger relative gains on the overlapping original split, suggesting overlap-induced memorization effects.

  </details>



- **PathMoE: Interpretable Multimodal Interaction Experts for Pediatric Brain Tumor Classification**  
  Jian Yu, Joakim Nguyen, Jinrui Fang, Awais Naeem, Zeyuan Cao, Sanjay Krishnan, Nicholas Konz, Tianlong Chen, Chandra Krishnan, Hairong Wang, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01547v1  
  <details><summary>Abstract</summary>

  Accurate classification of pediatric central nervous system tumors remains challenging due to histological complexity and limited training data. While pathology foundation models have advanced whole-slide image (WSI) analysis, they often fail to leverage the rich, complementary information found in clinical text and tissue microarchitecture. To this end, we propose PathMoE, an interpretable multimodal framework that integrates H\&E slides, pathology reports, and nuclei-level cell graphs via an interaction-aware mixture-of-experts architecture built on state-of-the-art foundation models for each modality. By training specialized experts to capture modality uniqueness, redundancy, and synergy, PathMoE employs an input-dependent gating mechanism that dynamically weights these interactions, providing sample-level interpretability. We evaluate our framework on two dataset-specific classification tasks on an internal pediatric brain tumor dataset (PBT) and external TCGA datasets. PathMoE improves macro-F1 from 0.762 to 0.799 (+0.037) on PBT when integrating WSI, text, and graph modalities; on TCGA, augmenting WSI with graph knowledge improves macro-F1 from 0.668 to 0.709 (+0.041). These results demonstrate significant performance gains over state-of-the-art image-only baselines while revealing the specific modality interactions driving individual predictions. This interpretability is particularly critical for rare tumor subtypes, where transparent model reasoning is essential for clinical trust and diagnostic validation.

  </details>



- **Training-Free Spatio-temporal Decoupled Reasoning Video Segmentation with Adaptive Object Memory**  
  Zhengtong Zhu, Jiaqing Fan, Zhixuan Liu, Fanzhang Li  
  _2026-03-02_ · https://arxiv.org/abs/2603.01545v1  
  <details><summary>Abstract</summary>

  Reasoning Video Object Segmentation (ReasonVOS) is a challenging task that requires stable object segmentation across video sequences using implicit and complex textual inputs. Previous methods fine-tune Multimodal Large Language Models (MLLMs) to produce segmentation outputs, which demand substantial resources. Additionally, some existing methods are coupled in the processing of spatio-temporal information, which affects the temporal stability of the model to some extent. To address these issues, we propose Training-Free \textbf{S}patio-temporal \textbf{D}ecoupled Reasoning Video Segmentation with \textbf{A}daptive Object \textbf{M}emory (SDAM). We aim to design a training-free reasoning video segmentation framework that outperforms existing methods requiring fine-tuning, using only pre-trained models. Meanwhile, we propose an Adaptive Object Memory module that selects and memorizes key objects based on motion cues in different video sequences. Finally, we propose Spatio-temporal Decoupling for stable temporal propagation. In the spatial domain, we achieve precise localization and segmentation of target objects, while in the temporal domain, we leverage key object temporal information to drive stable cross-frame propagation. Our method achieves excellent results on five benchmark datasets, including Ref-YouTubeVOS, Ref-DAVIS17, MeViS, ReasonVOS, and ReVOS.

  </details>



- **Benchmarking Semantic Segmentation Models via Appearance and Geometry Attribute Editing**  
  Zijin Yin, Bing Li, Kongming Liang, Hao Sun, Zhongjiang He, Zhanyu Ma, Jun Guo  
  _2026-03-02_ · https://arxiv.org/abs/2603.01535v1  
  <details><summary>Abstract</summary>

  Semantic segmentation takes pivotal roles in various applications such as autonomous driving and medical image analysis. When deploying segmentation models in practice, it is critical to test their behaviors in varied and complex scenes in advance. In this paper, we construct an automatic data generation pipeline Gen4Seg to stress-test semantic segmentation models by generating various challenging samples with different attribute changes. Beyond previous evaluation paradigms focusing solely on global weather and style transfer, we investigate variations in both appearance and geometry attributes at the object and image level. These include object color, material, size, position, as well as image-level variations such as weather and style. To achieve this, we propose to edit visual attributes of existing real images with precise control of structural information, empowered by diffusion models. In this way, the existing segmentation labels can be reused for the edited images, which greatly reduces the labor costs. Using our pipeline, we construct two new benchmarks, Pascal-EA and COCO-EA. We benchmark a wide variety of semantic segmentation models, spanning from closed-set models to open-vocabulary large models. We have several key findings: 1) advanced open-vocabulary models do not exhibit greater robustness compared to closed-set methods under geometric variations; 2) data augmentation techniques, such as CutOut and CutMix, are limited in enhancing robustness against appearance variations; 3) our pipeline can also be employed as a data augmentation tool and improve both in-distribution and out-of-distribution performances. Our work suggests the potential of generative models as effective tools for automatically analyzing segmentation models, and we hope our findings will assist practitioners and researchers in developing more robust and reliable segmentation models.

  </details>



- **Boosting AI Reliability with an FSM-Driven Streaming Inference Pipeline: An Industrial Case**  
  Yutian Zhang, Zhongyi Pei, Yi Mao, Chen Wang, Lin Liu, Jianmin Wang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01528v1  
  <details><summary>Abstract</summary>

  The widespread adoption of AI in industry is often hampered by its limited robustness when faced with scenarios absent from training data, leading to prediction bias and vulnerabilities. To address this, we propose a novel streaming inference pipeline that enhances data-driven models by explicitly incorporating prior knowledge. This paper presents the work on an industrial AI application that automatically counts excavator workloads from surveillance videos. Our approach integrates an object detection model with a Finite State Machine (FSM), which encodes knowledge of operational scenarios to guide and correct the AI's predictions on streaming data. In experiments on a real-world dataset of over 7,000 images from 12 site videos, encompassing more than 300 excavator workloads, our method demonstrates superior performance and greater robustness compared to the original solution based on manual heuristic rules. We will release the code at https://github.com/thulab/video-streamling-inference-pipeline.

  </details>



- **Better Matching, Less Forgetting: A Quality-Guided Matcher for Transformer-based Incremental Object Detection**  
  Qirui Wu, Shizhou Zhang, De Cheng, Yinghui Xing, Lingyan Ran, Dahu Shi, Peng Wang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01524v1  
  <details><summary>Abstract</summary>

  Incremental Object Detection (IOD) aims to continuously learn new object classes without forgetting previously learned ones. A persistent challenge is catastrophic forgetting, primarily attributed to background shift in conventional detectors. While pseudo-labeling mitigates this in dense detectors, we identify a novel, distinct source of forgetting specific to DETR-like architectures: background foregrounding. This arises from the exhaustiveness constraint of the Hungarian matcher, which forcibly assigns every ground truth target to one prediction, even when predictions primarily cover background regions (i.e., low IoU). This erroneous supervision compels the model to misclassify background features as specific foreground classes, disrupting learned representations and accelerating forgetting. To address this, we propose a Quality-guided Min-Cost Max-Flow (Q-MCMF) matcher. To avoid forced assignments, Q-MCMF builds a flow graph and prunes implausible matches based on geometric quality. It then optimizes for the final matching that minimizes cost and maximizes valid assignments. This strategy eliminates harmful supervision from background foregrounding while maximizing foreground learning signals. Extensive experiments on the COCO dataset under various incremental settings demonstrate that our method consistently outperforms existing state-of-the-art approaches.

  </details>



- **Retrieval, Refinement, and Ranking for Text-to-Video Generation via Prompt Optimization and Test-Time Scaling**  
  Zillur Rahman, Alex Sheng, Cristian Meo  
  _2026-03-02_ · https://arxiv.org/abs/2603.01509v1  
  <details><summary>Abstract</summary>

  While large-scale datasets have driven significant progress in Text-to-Video (T2V) generative models, these models remain highly sensitive to input prompts, demonstrating that prompt design is critical to generation quality. Current methods for improving video output often fall short: they either depend on complex, post-editing models, risking the introduction of artifacts, or require expensive fine-tuning of the core generator, which severely limits both scalability and accessibility. In this work, we introduce 3R, a novel RAG based prompt optimization framework. 3R utilizes the power of current state-of-the-art T2V diffusion model and vision language model. It can be used with any T2V model without any kind of model training. The framework leverages three key strategies: RAG-based modifiers extraction for enriched contextual grounding, diffusion-based Preference Optimization for aligning outputs with human preferences, and temporal frame interpolation for producing temporally consistent visual contents. Together, these components enable more accurate, efficient, and contextually aligned text-to-video generation. Experimental results demonstrate the efficacy of 3R in enhancing the static fidelity and dynamic coherence of generated videos, underscoring the importance of optimizing user prompts.

  </details>



- **Tri-path DINO: Feature Complementary Learning for Remote Sensing Multi-Class Change Detection**  
  Kai Zheng, Hang-Cheng Dong, Zhenkai Wu, Fupeng Wei, Wei Zhang  
  _2026-03-02_ · https://arxiv.org/abs/2603.01498v1  
  <details><summary>Abstract</summary>

  In remote sensing imagery, multi class change detection (MCD) is crucial for fine grained monitoring, yet it has long been constrained by complex scene variations and the scarcity of detailed annotations. To address this, we propose the Tripath DINO architecture, which adopts a three path complementary feature learning strategy to facilitate the rapid adaptation of pre trained foundation models to complex vertical domains. Specifically, we employ the DINOv3 pre trained model as the backbone feature extraction network to learn coarse grained features. An auxiliary path also adopts a siamese structure, progressively aggregating intermediate features from the siamese encoder to enhance the learning of fine grained features. Finally, a multi scale attention mechanism is introduced to augment the decoder network, where parallel convolutions adaptively capture and enhance contextual information under different receptive fields. The proposed method achieves optimal performance on the MCD task on both the Gaza facility damage assessment dataset (Gaza change) and the classic SECOND dataset. GradCAM visualizations further confirm that the main and auxiliary paths naturally focus on coarse grained semantic changes and fine grained structural details, respectively. This synergistic complementarity provides a robust and interpretable solution for advanced change detection tasks, offering a basis for rapid and accurate damage assessment.

  </details>



- **PhotoBench: Beyond Visual Matching Towards Personalized Intent-Driven Photo Retrieval**  
  Tianyi Xu, Rong Shan, Junjie Wu, Jiadeng Huang, Teng Wang, Jiachen Zhu, Wenteng Chen, Minxin Tu, Quantao Dou, Zhaoxiang Wang, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01493v1  
  <details><summary>Abstract</summary>

  Personal photo albums are not merely collections of static images but living, ecological archives defined by temporal continuity, social entanglement, and rich metadata, which makes the personalized photo retrieval non-trivial. However, existing retrieval benchmarks rely heavily on context-isolated web snapshots, failing to capture the multi-source reasoning required to resolve authentic, intent-driven user queries. To bridge this gap, we introduce PhotoBench, the first benchmark constructed from authentic, personal albums. It is designed to shift the paradigm from visual matching to personalized multi-source intent-driven reasoning. Based on a rigorous multi-source profiling framework, which integrates visual semantics, spatial-temporal metadata, social identity, and temporal events for each image, we synthesize complex intent-driven queries rooted in users' life trajectories. Extensive evaluation on PhotoBench exposes two critical limitations: the modality gap, where unified embedding models collapse on non-visual constraints, and the source fusion paradox, where agentic systems perform poor tool orchestration. These findings indicate that the next frontier in personal multimodal retrieval lies beyond unified embeddings, necessitating robust agentic reasoning systems capable of precise constraint satisfaction and multi-source fusion. Our PhotoBench is available.

  </details>



- **ATA: Bridging Implicit Reasoning with Attention-Guided and Action-Guided Inference for Vision-Language Action Models**  
  Cheng Yang, Jianhao Jiao, Lingyi Huang, Jinqi Xiao, Zhexiang Tang, Yu Gong, Yibiao Ying, Yang Sui, Jintian Lin, Wen Huang, et al.  
  _2026-03-02_ · https://arxiv.org/abs/2603.01490v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models rely on current observations, including images, language instructions, and robot states, to predict actions and complete tasks. While accurate visual perception is crucial for precise action prediction and execution, recent work has attempted to further improve performance by introducing explicit reasoning during inference. However, such approaches face significant limitations. They often depend on data-intensive resources such as Chain-of-Thought (CoT) style annotations to decompose tasks into step-by-step reasoning, and in many cases require additional visual grounding annotations (e.g., bounding boxes or masks) to highlight relevant image regions. Moreover, they involve time-consuming dataset construction, labeling, and retraining, which ultimately results in longer inference sequences and reduced efficiency. To address these challenges, we propose ATA, a novel training-free framework that introduces implicit reasoning into VLA inference through complementary attention-guided and action-guided strategies. Unlike CoT or explicit visual-grounding methods, ATA formulates reasoning implicitly by integrating attention maps with an action-based region of interest (RoI), thereby adaptively refining visual inputs without requiring extra training or annotations. ATA is a plug-and-play implicit reasoning approach for VLA models, lightweight yet effective. Extensive experiments show that it consistently improves task success and robustness while preserving, and even enhancing, inference efficiency.

  </details>


