# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-16 13:08 UTC_

Total papers shown: **50**


---

- **WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments**  
  Xuweiyi Chen, Wentao Zhou, Zezhou Cheng  
  _2026-01-15_ · https://arxiv.org/abs/2601.10716v1  
  <details><summary>Abstract</summary>

  We present WildRayZer, a self-supervised framework for novel view synthesis (NVS) in dynamic environments where both the camera and objects move. Dynamic content breaks the multi-view consistency that static NVS models rely on, leading to ghosting, hallucinated geometry, and unstable pose estimation. WildRayZer addresses this by performing an analysis-by-synthesis test: a camera-only static renderer explains rigid structure, and its residuals reveal transient regions. From these residuals, we construct pseudo motion masks, distill a motion estimator, and use it to mask input tokens and gate loss gradients so supervision focuses on cross-view background completion. To enable large-scale training and evaluation, we curate Dynamic RealEstate10K (D-RE10K), a real-world dataset of 15K casually captured dynamic sequences, and D-RE10K-iPhone, a paired transient and clean benchmark for sparse-view transient-aware NVS. Experiments show that WildRayZer consistently outperforms optimization-based and feed-forward baselines in both transient-region removal and full-frame NVS quality with a single feed-forward pass.

  </details>



- **Alterbute: Editing Intrinsic Attributes of Objects in Images**  
  Tal Reiss, Daniel Winter, Matan Cohen, Alex Rav-Acha, Yael Pritch, Ariel Shamir, Yedid Hoshen  
  _2026-01-15_ · https://arxiv.org/abs/2601.10714v1  
  <details><summary>Abstract</summary>

  We introduce Alterbute, a diffusion-based method for editing an object's intrinsic attributes in an image. We allow changing color, texture, material, and even the shape of an object, while preserving its perceived identity and scene context. Existing approaches either rely on unsupervised priors that often fail to preserve identity or use overly restrictive supervision that prevents meaningful intrinsic variations. Our method relies on: (i) a relaxed training objective that allows the model to change both intrinsic and extrinsic attributes conditioned on an identity reference image, a textual prompt describing the target intrinsic attributes, and a background image and object mask defining the extrinsic context. At inference, we restrict extrinsic changes by reusing the original background and object mask, thereby ensuring that only the desired intrinsic attributes are altered; (ii) Visual Named Entities (VNEs) - fine-grained visual identity categories (e.g., ''Porsche 911 Carrera'') that group objects sharing identity-defining features while allowing variation in intrinsic attributes. We use a vision-language model to automatically extract VNE labels and intrinsic attribute descriptions from a large public image dataset, enabling scalable, identity-preserving supervision. Alterbute outperforms existing methods on identity-preserving object intrinsic attribute editing.

  </details>



- **CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning**  
  Darshan Singh, Arsha Nagrani, Kawshik Manikantan, Harman Singh, Dinesh Tewari, Tobias Weyand, Cordelia Schmid, Anelia Angelova, Shachi Dave  
  _2026-01-15_ · https://arxiv.org/abs/2601.10649v1  
  <details><summary>Abstract</summary>

  Recent advancements in video models have shown tremendous progress, particularly in long video understanding. However, current benchmarks predominantly feature western-centric data and English as the dominant language, introducing significant biases in evaluation. To address this, we introduce CURVE (Cultural Understanding and Reasoning in Video Evaluation), a challenging benchmark for multicultural and multilingual video reasoning. CURVE comprises high-quality, entirely human-generated annotations from diverse, region-specific cultural videos across 18 global locales. Unlike prior work that relies on automatic translations, CURVE provides complex questions, answers, and multi-step reasoning steps, all crafted in native languages. Making progress on CURVE requires a deeply situated understanding of visual cultural context. Furthermore, we leverage CURVE's reasoning traces to construct evidence-based graphs and propose a novel iterative strategy using these graphs to identify fine-grained errors in reasoning. Our evaluations reveal that SoTA Video-LLMs struggle significantly, performing substantially below human-level accuracy, with errors primarily stemming from the visual perception of cultural elements. CURVE will be publicly available under https://github.com/google-deepmind/neptune?tab=readme-ov-file\#minerva-cultural

  </details>



- **CoMoVi: Co-Generation of 3D Human Motions and Realistic Videos**  
  Chengfeng Zhao, Jiazhi Shu, Yubo Zhao, Tianyu Huang, Jiahao Lu, Zekai Gu, Chengwei Ren, Zhiyang Dou, Qing Shuai, Yuan Liu  
  _2026-01-15_ · https://arxiv.org/abs/2601.10632v1  
  <details><summary>Abstract</summary>

  In this paper, we find that the generation of 3D human motions and 2D human videos is intrinsically coupled. 3D motions provide the structural prior for plausibility and consistency in videos, while pre-trained video models offer strong generalization capabilities for motions, which necessitate coupling their generation processes. Based on this, we present CoMoVi, a co-generative framework that couples two video diffusion models (VDMs) to generate 3D human motions and videos synchronously within a single diffusion denoising loop. To achieve this, we first propose an effective 2D human motion representation that can inherit the powerful prior of pre-trained VDMs. Then, we design a dual-branch diffusion model to couple human motion and video generation process with mutual feature interaction and 3D-2D cross attentions. Moreover, we curate CoMoVi Dataset, a large-scale real-world human video dataset with text and motion annotations, covering diverse and challenging human motions. Extensive experiments demonstrate the effectiveness of our method in both 3D human motion and video generation tasks.

  </details>



- **Molmo2: Open Weights and Data for Vision-Language Models with Video Understanding and Grounding**  
  Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, et al.  
  _2026-01-15_ · https://arxiv.org/abs/2601.10611v1  
  <details><summary>Abstract</summary>

  Today's strongest video-language models (VLMs) remain proprietary. The strongest open-weight models either rely on synthetic data from proprietary VLMs, effectively distilling from them, or do not disclose their training data or recipe. As a result, the open-source community lacks the foundations needed to improve on the state-of-the-art video (and image) language models. Crucially, many downstream applications require more than just high-level video understanding; they require grounding -- either by pointing or by tracking in pixels. Even proprietary models lack this capability. We present Molmo2, a new family of VLMs that are state-of-the-art among open-source models and demonstrate exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks. Our key contribution is a collection of 7 new video datasets and 2 multi-image datasets, including a dataset of highly detailed video captions for pre-training, a free-form video Q&A dataset for fine-tuning, a new object tracking dataset with complex queries, and an innovative new video pointing dataset, all collected without the use of closed VLMs. We also present a training recipe for this data utilizing an efficient packing and message-tree encoding scheme, and show bi-directional attention on vision tokens and a novel token-weight strategy improves performance. Our best-in-class 8B model outperforms others in the class of open weight and data models on short videos, counting, and captioning, and is competitive on long-videos. On video-grounding Molmo2 significantly outperforms existing open-weight models like Qwen3-VL (35.5 vs 29.6 accuracy on video counting) and surpasses proprietary models like Gemini 3 Pro on some tasks (38.4 vs 20.0 F1 on video pointing and 56.2 vs 41.1 J&F on video tracking).

  </details>



- **Multi-Objective Pareto-Front Optimization for Efficient Adaptive VVC Streaming**  
  Angeliki Katsenou, Vignesh V. Menon, Guoda Laurinaviciute, Benjamin Bross, Detlev Marpe  
  _2026-01-15_ · https://arxiv.org/abs/2601.10607v1  
  <details><summary>Abstract</summary>

  Adaptive video streaming has facilitated improved video streaming over the past years. A balance among coding performance objectives such as bitrate, video quality, and decoding complexity is required to achieve efficient, content- and codec-dependent, adaptive video streaming. This paper proposes a multi-objective Pareto-front (PF) optimization framework to construct quality-monotonic, content-adaptive bitrate ladders Versatile Video Coding (VVC) streaming that jointly optimize video quality, bitrate, and decoding time, which is used as a practical proxy for decoding energy. Two strategies are introduced: the Joint Rate-Quality-Time Pareto Front (JRQT-PF) and the Joint Quality-Time Pareto Front (JQT-PF), each exploring different tradeoff formulations and objective prioritizations. The ladders are constructed under quality monotonicity constraints during adaptive streaming to ensure a consistent Quality of Experience (QoE). Experiments are conducted on a large-scale UHD dataset (Inter-4K), with quality assessed using PSNR, VMAF, and XPSNR, and complexity measured via decoding time and energy consumption. The JQT-PF method achieves 11.76% average bitrate savings while reducing average decoding time by 0.29% to maintain the same XPSNR, compared to a widely-used fixed ladder. More aggressive configurations yield up to 27.88% bitrate savings at the cost of increased complexity. The JRQT-PF strategy, on the other hand, offers more controlled tradeoffs, achieving 6.38 % bitrate savings and 6.17 % decoding time reduction. This framework outperforms existing methods, including fixed ladders, VMAF- and XPSNR-based dynamic resolution selection, and complexity-aware benchmarks. The results confirm that PF optimization with decoding time constraints enables sustainable, high-quality streaming tailored to network and device capabilities.

  </details>



- **RSATalker: Realistic Socially-Aware Talking Head Generation for Multi-Turn Conversation**  
  Peng Chen, Xiaobao Wei, Yi Yang, Naiming Yao, Hui Chen, Feng Tian  
  _2026-01-15_ · https://arxiv.org/abs/2601.10606v1  
  <details><summary>Abstract</summary>

  Talking head generation is increasingly important in virtual reality (VR), especially for social scenarios involving multi-turn conversation. Existing approaches face notable limitations: mesh-based 3D methods can model dual-person dialogue but lack realistic textures, while large-model-based 2D methods produce natural appearances but incur prohibitive computational costs. Recently, 3D Gaussian Splatting (3DGS) based methods achieve efficient and realistic rendering but remain speaker-only and ignore social relationships. We introduce RSATalker, the first framework that leverages 3DGS for realistic and socially-aware talking head generation with support for multi-turn conversation. Our method first drives mesh-based 3D facial motion from speech, then binds 3D Gaussians to mesh facets to render high-fidelity 2D avatar videos. To capture interpersonal dynamics, we propose a socially-aware module that encodes social relationships, including blood and non-blood as well as equal and unequal, into high-level embeddings through a learnable query mechanism. We design a three-stage training paradigm and construct the RSATalker dataset with speech-mesh-image triplets annotated with social relationships. Extensive experiments demonstrate that RSATalker achieves state-of-the-art performance in both realism and social awareness. The code and dataset will be released.

  </details>



- **Action100M: A Large-scale Video Action Dataset**  
  Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, Pascale Fung  
  _2026-01-15_ · https://arxiv.org/abs/2601.10592v1  
  <details><summary>Abstract</summary>

  Inferring physical actions from visual observations is a fundamental capability for advancing machine intelligence in the physical world. Achieving this requires large-scale, open-vocabulary video action datasets that span broad domains. We introduce Action100M, a large-scale dataset constructed from 1.2M Internet instructional videos (14.6 years of duration), yielding O(100 million) temporally localized segments with open-vocabulary action supervision and rich captions. Action100M is generated by a fully automated pipeline that (i) performs hierarchical temporal segmentation using V-JEPA 2 embeddings, (ii) produces multi-level frame and segment captions organized as a Tree-of-Captions, and (iii) aggregates evidence with a reasoning model (GPT-OSS-120B) under a multi-round Self-Refine procedure to output structured annotations (brief/detailed action, actor, brief/detailed caption). Training VL-JEPA on Action100M demonstrates consistent data-scaling improvements and strong zero-shot performance across diverse action recognition benchmarks, establishing Action100M as a new foundation for scalable research in video understanding and world modeling.

  </details>



- **DeepUrban: Interaction-Aware Trajectory Prediction and Planning for Automated Driving by Aerial Imagery**  
  Constantin Selzer, Fabian B. Flohr  
  _2026-01-15_ · https://arxiv.org/abs/2601.10554v1  
  <details><summary>Abstract</summary>

  The efficacy of autonomous driving systems hinges critically on robust prediction and planning capabilities. However, current benchmarks are impeded by a notable scarcity of scenarios featuring dense traffic, which is essential for understanding and modeling complex interactions among road users. To address this gap, we collaborated with our industrial partner, DeepScenario, to develop DeepUrban-a new drone dataset designed to enhance trajectory prediction and planning benchmarks focusing on dense urban settings. DeepUrban provides a rich collection of 3D traffic objects, extracted from high-resolution images captured over urban intersections at approximately 100 meters altitude. The dataset is further enriched with comprehensive map and scene information to support advanced modeling and simulation tasks. We evaluate state-of-the-art (SOTA) prediction and planning methods, and conducted experiments on generalization capabilities. Our findings demonstrate that adding DeepUrban to nuScenes can boost the accuracy of vehicle predictions and planning, achieving improvements up to 44.1 % / 44.3% on the ADE / FDE metrics. Website: https://iv.ee.hm.edu/deepurban

  </details>



- **Unleashing the Capabilities of Large Vision-Language Models for Intelligent Perception of Roadside Infrastructure**  
  Luxuan Fu, Chong Liu, Bisheng Yang, Zhen Dong  
  _2026-01-15_ · https://arxiv.org/abs/2601.10551v1  
  <details><summary>Abstract</summary>

  Automated perception of urban roadside infrastructure is crucial for smart city management, yet general-purpose models often struggle to capture the necessary fine-grained attributes and domain rules. While Large Vision Language Models (VLMs) excel at open-world recognition, they often struggle to accurately interpret complex facility states in compliance with engineering standards, leading to unreliable performance in real-world applications. To address this, we propose a domain-adapted framework that transforms VLMs into specialized agents for intelligent infrastructure analysis. Our approach integrates a data-efficient fine-tuning strategy with a knowledge-grounded reasoning mechanism. Specifically, we leverage open-vocabulary fine-tuning on Grounding DINO to robustly localize diverse assets with minimal supervision, followed by LoRA-based adaptation on Qwen-VL for deep semantic attribute reasoning. To mitigate hallucinations and enforce professional compliance, we introduce a dual-modality Retrieval-Augmented Generation (RAG) module that dynamically retrieves authoritative industry standards and visual exemplars during inference. Evaluated on a comprehensive new dataset of urban roadside scenes, our framework achieves a detection performance of 58.9 mAP and an attribute recognition accuracy of 95.5%, demonstrating a robust solution for intelligent infrastructure monitoring.

  </details>



- **Enhancing the quality of gauge images captured in smoke and haze scenes through deep learning**  
  Oscar H. Ramírez-Agudelo, Akshay N. Shewatkar, Edoardo Milana, Roland C. Aydin, Kai Franke  
  _2026-01-15_ · https://arxiv.org/abs/2601.10537v1  
  <details><summary>Abstract</summary>

  Images captured in hazy and smoky environments suffer from reduced visibility, posing a challenge when monitoring infrastructures and hindering emergency services during critical situations. The proposed work investigates the use of the deep learning models to enhance the automatic, machine-based readability of gauge in smoky environments, with accurate gauge data interpretation serving as a valuable tool for first responders. The study utilizes two deep learning architectures, FFA-Net and AECR-Net, to improve the visibility of gauge images, corrupted with light up to dense haze and smoke. Since benchmark datasets of analog gauge images are unavailable, a new synthetic dataset, containing over 14,000 images, was generated using the Unreal Engine. The models were trained with an 80\% train, 10\% validation, and 10\% test split for the haze and smoke dataset, respectively. For the synthetic haze dataset, the SSIM and PSNR metrics are about 0.98 and 43\,dB, respectively, comparing well to state-of-the art results. Additionally, more robust results are retrieved from the AECR-Net, when compared to the FFA-Net. Although the results from the synthetic smoke dataset are poorer, the trained models achieve interesting results. In general, imaging in the presence of smoke are more difficult to enhance given the inhomogeneity and high density. Secondly, FFA-Net and AECR-Net are implemented to dehaze and not to desmoke images. This work shows that use of deep learning architectures can improve the quality of analog gauge images captured in smoke and haze scenes immensely. Finally, the enhanced output images can be successfully post-processed for automatic autonomous reading of gauges

  </details>



- **A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5**  
  Xingjun Ma, Yixu Wang, Hengyuan Xu, Yutao Wu, Yifan Ding, Yunhan Zhao, Zilong Wang, Jiabin Hua, Ming Wen, Jianan Liu, et al.  
  _2026-01-15_ · https://arxiv.org/abs/2601.10527v1  
  <details><summary>Abstract</summary>

  The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has produced substantial gains in reasoning, perception, and generative capability across language and vision. However, whether these advances yield commensurate improvements in safety remains unclear, in part due to fragmented evaluation practices limited to single modalities or threat models. In this report, we present an integrated safety evaluation of 7 frontier models: GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5. We evaluate each model across language, vision-language, and image generation settings using a unified protocol that integrates benchmark evaluation, adversarial evaluation, multilingual evaluation, and compliance evaluation. Aggregating our evaluations into safety leaderboards and model safety profiles across multiple evaluation modes reveals a sharply heterogeneous safety landscape. While GPT-5.2 demonstrates consistently strong and balanced safety performance across evaluations, other models exhibit pronounced trade-offs among benchmark safety, adversarial alignment, multilingual generalization, and regulatory compliance. Both language and vision-language modalities show significant vulnerability under adversarial evaluation, with all models degrading substantially despite strong results on standard benchmarks. Text-to-image models achieve relatively stronger alignment in regulated visual risk categories, yet remain brittle under adversarial or semantically ambiguous prompts. Overall, these results show that safety in frontier models is inherently multidimensional--shaped by modality, language, and evaluation scheme, underscoring the need for standardized safety evaluations to accurately assess real-world risk and guide responsible model development and deployment.

  </details>



- **BikeActions: An Open Platform and Benchmark for Cyclist-Centric VRU Action Recognition**  
  Max A. Buettner, Kanak Mazumder, Luca Koecher, Mario Finkbeiner, Sebastian Niebler, Fabian B. Flohr  
  _2026-01-15_ · https://arxiv.org/abs/2601.10521v1  
  <details><summary>Abstract</summary>

  Anticipating the intentions of Vulnerable Road Users (VRUs) is a critical challenge for safe autonomous driving (AD) and mobile robotics. While current research predominantly focuses on pedestrian crossing behaviors from a vehicle's perspective, interactions within dense shared spaces remain underexplored. To bridge this gap, we introduce FUSE-Bike, the first fully open perception platform of its kind. Equipped with two LiDARs, a camera, and GNSS, it facilitates high-fidelity, close-range data capture directly from a cyclist's viewpoint. Leveraging this platform, we present BikeActions, a novel multi-modal dataset comprising 852 annotated samples across 5 distinct action classes, specifically tailored to improve VRU behavior modeling. We establish a rigorous benchmark by evaluating state-of-the-art graph convolution and transformer-based models on our publicly released data splits, establishing the first performance baselines for this challenging task. We release the full dataset together with data curation tools, the open hardware design, and the benchmark code to foster future research in VRU action understanding under https://iv.ee.hm.edu/bikeactions/.

  </details>



- **SatMap: Revisiting Satellite Maps as Prior for Online HD Map Construction**  
  Kanak Mazumder, Fabian B. Flohr  
  _2026-01-15_ · https://arxiv.org/abs/2601.10512v1  
  <details><summary>Abstract</summary>

  Online high-definition (HD) map construction is an essential part of a safe and robust end-to-end autonomous driving (AD) pipeline. Onboard camera-based approaches suffer from limited depth perception and degraded accuracy due to occlusion. In this work, we propose SatMap, an online vectorized HD map estimation method that integrates satellite maps with multi-view camera observations and directly predicts a vectorized HD map for downstream prediction and planning modules. Our method leverages lane-level semantics and texture from satellite imagery captured from a Bird's Eye View (BEV) perspective as a global prior, effectively mitigating depth ambiguity and occlusion. In our experiments on the nuScenes dataset, SatMap achieves 34.8% mAP performance improvement over the camera-only baseline and 8.5% mAP improvement over the camera-LiDAR fusion baseline. Moreover, we evaluate our model in long-range and adverse weather conditions to demonstrate the advantages of using a satellite prior map. Source code will be available at https://iv.ee.hm.edu/satmap/.

  </details>



- **mergetune: Continued fine-tuning of vision-language models**  
  Wenqing Wang, Da Li, Xiatian Zhu, Josef Kittler  
  _2026-01-15_ · https://arxiv.org/abs/2601.10497v1  
  <details><summary>Abstract</summary>

  Fine-tuning vision-language models (VLMs) such as CLIP often leads to catastrophic forgetting of pretrained knowledge. Prior work primarily aims to mitigate forgetting during adaptation; however, forgetting often remains inevitable during this process. We introduce a novel paradigm, \emph{continued fine-tuning (CFT)}, which seeks to recover pretrained knowledge after a zero-shot model has already been adapted. We propose a simple, model-agnostic CFT strategy (named MERGETUNE) guided by linear mode connectivity (LMC), which can be applied post hoc to existing fine-tuned models without requiring architectural changes. Given a fine-tuned model, we continue fine-tuning its trainable parameters (e.g., soft prompts or linear heads) to search for a continued model which has two low-loss paths to the zero-shot (e.g., CLIP) and the fine-tuned (e.g., CoOp) solutions. By exploiting the geometry of the loss landscape, the continued model implicitly merges the two solutions, restoring pretrained knowledge lost in the fine-tuned counterpart. A challenge is that the vanilla LMC constraint requires data replay from the pretraining task. We approximate this constraint for the zero-shot model via a second-order surrogate, eliminating the need for large-scale data replay. Experiments show that MERGETUNE improves the harmonic mean of CoOp by +5.6\% on base-novel generalisation without adding parameters. % We show \emph{the first time} superior performance than CLIP on both DTD and EuroSAT, on cross-dataset transfer. On robust fine-tuning evaluations, the LMC-merged model from MERGETUNE surpasses ensemble baselines with lower inference cost, achieving further gains and state-of-the-art results when ensembled with the zero-shot model. Our code is available at \href{https://github.com/Surrey-UP-Lab/MERGETUNE}{https://github.com/Surrey-UP-Lab/MERGETUNE}.

  </details>



- **Urban Socio-Semantic Segmentation with Vision-Language Reasoning**  
  Yu Wang, Yi Wang, Rui Dai, Yujie Wang, Kaikui Liu, Xiangxiang Chu, Yansheng Li  
  _2026-01-15_ · https://arxiv.org/abs/2601.10477v1  
  <details><summary>Abstract</summary>

  As hubs of human activity, urban surfaces consist of a wealth of semantic entities. Segmenting these various entities from satellite imagery is crucial for a range of downstream applications. Current advanced segmentation models can reliably segment entities defined by physical attributes (e.g., buildings, water bodies) but still struggle with socially defined categories (e.g., schools, parks). In this work, we achieve socio-semantic segmentation by vision-language model reasoning. To facilitate this, we introduce the Urban Socio-Semantic Segmentation dataset named SocioSeg, a new resource comprising satellite imagery, digital maps, and pixel-level labels of social semantic entities organized in a hierarchical structure. Additionally, we propose a novel vision-language reasoning framework called SocioReasoner that simulates the human process of identifying and annotating social semantic entities via cross-modal recognition and multi-stage reasoning. We employ reinforcement learning to optimize this non-differentiable process and elicit the reasoning capabilities of the vision-language model. Experiments demonstrate our approach's gains over state-of-the-art models and strong zero-shot generalization. Our dataset and code are available in https://github.com/AMAP-ML/SocioReasoner.

  </details>



- **ChartComplete: A Taxonomy-based Inclusive Chart Dataset**  
  Ahmad Mustapha, Charbel Toumieh, Mariette Awad  
  _2026-01-15_ · https://arxiv.org/abs/2601.10462v1  
  <details><summary>Abstract</summary>

  With advancements in deep learning (DL) and computer vision techniques, the field of chart understanding is evolving rapidly. In particular, multimodal large language models (MLLMs) are proving to be efficient and accurate in understanding charts. To accurately measure the performance of MLLMs, the research community has developed multiple datasets to serve as benchmarks. By examining these datasets, we found that they are all limited to a small set of chart types. To bridge this gap, we propose the ChartComplete dataset. The dataset is based on a chart taxonomy borrowed from the visualization community, and it covers thirty different chart types. The dataset is a collection of classified chart images and does not include a learning signal. We present the ChartComplete dataset as is to the community to build upon it.

  </details>



- **SurgGoal: Rethinking Surgical Planning Evaluation via Goal-Satisfiability**  
  Ruochen Li, Kun Yuan, Yufei Xia, Yue Zhou, Qingyu Lu, Weihang Li, Youxiang Zhu, Nassir Navab  
  _2026-01-15_ · https://arxiv.org/abs/2601.10455v1  
  <details><summary>Abstract</summary>

  Surgical planning integrates visual perception, long-horizon reasoning, and procedural knowledge, yet it remains unclear whether current evaluation protocols reliably assess vision-language models (VLMs) in safety-critical settings. Motivated by a goal-oriented view of surgical planning, we define planning correctness via phase-goal satisfiability, where plan validity is determined by expert-defined surgical rules. Based on this definition, we introduce a multicentric meta-evaluation benchmark with valid procedural variations and invalid plans containing order and content errors. Using this benchmark, we show that sequence similarity metrics systematically misjudge planning quality, penalizing valid plans while failing to identify invalid ones. We therefore adopt a rule-based goal-satisfiability metric as a high-precision meta-evaluation reference to assess Video-LLMs under progressively constrained settings, revealing failures due to perception errors and under-constrained reasoning. Structural knowledge consistently improves performance, whereas semantic guidance alone is unreliable and benefits larger models only when combined with structural constraints.

  </details>



- **Subjective evaluation of UHD video coded using VVC with LCEVC and ML-VVC**  
  Naeem Ramzan, Muhammad Tufail Khan  
  _2026-01-15_ · https://arxiv.org/abs/2601.10448v1  
  <details><summary>Abstract</summary>

  This paper presents the results of a subjective quality assessment of a multilayer video coding configuration in which Low Complexity Enhancement Video Coding (LCEVC) is applied as an enhancement layer on top of a Versatile Video Coding (VVC) base layer. The evaluation follows the same test methodology and conditions previously defined for MPEG multilayer video coding assessments, with the LCEVC enhancement layer encoded using version 8.1 of the LCEVC Test Model (LTM). The test compares reconstructed UHD output generated from an HD VVC base layer with LCEVC enhancement against two reference cases: upsampled VVC base layer decoding and multilayer VVC (ML-VVC). Two operating points are considered, corresponding to enhancement layers representing approximately 10% and 50% of the total bitrate. Subjective assessment was conducted using the Degradation Category Rating (DCR) methodology with twenty five participants, across a dataset comprising fifteen SDR and HDR sequences. The reported results include Mean Opinion Scores (MOS) with associated 95% confidence intervals, enabling comparison of perceptual quality across coding approaches and operating points within the defined test scope.

  </details>



- **Multi-Temporal Frames Projection for Dynamic Processes Fusion in Fluorescence Microscopy**  
  Hassan Eshkiki, Sarah Costa, Mostafa Mohammadpour, Farinaz Tanhaei, Christopher H. George, Fabio Caraffini  
  _2026-01-15_ · https://arxiv.org/abs/2601.10392v1  
  <details><summary>Abstract</summary>

  Fluorescence microscopy is widely employed for the analysis of living biological samples; however, the utility of the resulting recordings is frequently constrained by noise, temporal variability, and inconsistent visualisation of signals that oscillate over time. We present a unique computational framework that integrates information from multiple time-resolved frames into a single high-quality image, while preserving the underlying biological content of the original video. We evaluate the proposed method through an extensive number of configurations (n = 111) and on a challenging dataset comprising dynamic, heterogeneous, and morphologically complex 2D monolayers of cardiac cells. Results show that our framework, which consists of a combination of explainable techniques from different computer vision application fields, is capable of generating composite images that preserve and enhance the quality and information of individual microscopy frames, yielding 44% average increase in cell count compared to previous methods. The proposed pipeline is applicable to other imaging domains that require the fusion of multi-temporal image stacks into high-quality 2D images, thereby facilitating annotation and downstream segmentation.

  </details>



- **Fine-Grained Human Pose Editing Assessment via Layer-Selective MLLMs**  
  Ningyu Sun, Zhaolin Cai, Zitong Xu, Peihang Chen, Huiyu Duan, Yichao Yan, Xiongkuo Min, Xiaokang Yang  
  _2026-01-15_ · https://arxiv.org/abs/2601.10369v1  
  <details><summary>Abstract</summary>

  Text-guided human pose editing has gained significant traction in AIGC applications. However,it remains plagued by structural anomalies and generative artifacts. Existing evaluation metrics often isolate authenticity detection from quality assessment, failing to provide fine-grained insights into pose-specific inconsistencies. To address these limitations, we introduce HPE-Bench, a specialized benchmark comprising 1,700 standardized samples from 17 state-of-the-art editing models, offering both authenticity labels and multi-dimensional quality scores. Furthermore, we propose a unified framework based on layer-selective multimodal large language models (MLLMs). By employing contrastive LoRA tuning and a novel layer sensitivity analysis (LSA) mechanism, we identify the optimal feature layer for pose evaluation. Our framework achieves superior performance in both authenticity detection and multi-dimensional quality regression, effectively bridging the gap between forensic detection and quality assessment.

  </details>



- **An analytic theory of convolutional neural network inverse problems solvers**  
  Minh Hai Nguyen, Quoc Bao Do, Edouard Pauwels, Pierre Weiss  
  _2026-01-15_ · https://arxiv.org/abs/2601.10334v1  
  <details><summary>Abstract</summary>

  Supervised convolutional neural networks (CNNs) are widely used to solve imaging inverse problems, achieving state-of-the-art performance in numerous applications. However, despite their empirical success, these methods are poorly understood from a theoretical perspective and often treated as black boxes. To bridge this gap, we analyze trained neural networks through the lens of the Minimum Mean Square Error (MMSE) estimator, incorporating functional constraints that capture two fundamental inductive biases of CNNs: translation equivariance and locality via finite receptive fields. Under the empirical training distribution, we derive an analytic, interpretable, and tractable formula for this constrained variant, termed Local-Equivariant MMSE (LE-MMSE). Through extensive numerical experiments across various inverse problems (denoising, inpainting, deconvolution), datasets (FFHQ, CIFAR-10, FashionMNIST), and architectures (U-Net, ResNet, PatchMLP), we demonstrate that our theory matches the neural networks outputs (PSNR $\gtrsim25$dB). Furthermore, we provide insights into the differences between \emph{physics-aware} and \emph{physics-agnostic} estimators, the impact of high-density regions in the training (patch) distribution, and the influence of other factors (dataset size, patch size, etc).

  </details>



- **ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding**  
  Xueyun Tian, Wei Li, Bingbing Xu, Heng Dong, Yuanzhuo Wang, Huawei Shen  
  _2026-01-15_ · https://arxiv.org/abs/2601.10323v1  
  <details><summary>Abstract</summary>

  Recent Omni-multimodal Large Language Models show promise in unified audio, vision, and text modeling. However, streaming audio-video understanding remains challenging, as existing approaches suffer from disjointed capabilities: they typically exhibit incomplete modality support or lack autonomous proactive monitoring. To address this, we present ROMA, a real-time omni-multimodal assistant for unified reactive and proactive interaction. ROMA processes continuous inputs as synchronized multimodal units, aligning dense audio with discrete video frames to handle granularity mismatches. For online decision-making, we introduce a lightweight speak head that decouples response initiation from generation to ensure precise triggering without task conflict. We train ROMA with a curated streaming dataset and a two-stage curriculum that progressively optimizes for streaming format adaptation and proactive responsiveness. To standardize the fragmented evaluation landscape, we reorganize diverse benchmarks into a unified suite covering both proactive (alert, narration) and reactive (QA) settings. Extensive experiments across 12 benchmarks demonstrate ROMA achieves state-of-the-art performance on proactive tasks while competitive in reactive settings, validating its robustness in unified real-time omni-multimodal understanding.

  </details>



- **DanQing: An Up-to-Date Large-Scale Chinese Vision-Language Pre-training Dataset**  
  Hengyu Shen, Tiancheng Gu, Bin Qin, Lan Wu, Yuling Wu, Shuo Tan, Zelong Sun, Jun Wang, Nan Wu, Xiang An, et al.  
  _2026-01-15_ · https://arxiv.org/abs/2601.10305v1  
  <details><summary>Abstract</summary>

  Vision-Language Pre-training (VLP) models demonstrate strong performance across various downstream tasks by learning from large-scale image-text pairs through contrastive pretraining. The release of extensive English image-text datasets (e.g., COYO-700M and LAION-400M) has enabled widespread adoption of models such as CLIP and SigLIP in tasks including cross-modal retrieval and image captioning. However, the advancement of Chinese vision-language pretraining has substantially lagged behind, due to the scarcity of high-quality Chinese image-text data. To address this gap, we develop a comprehensive pipeline for constructing a high-quality Chinese cross-modal dataset. As a result, we propose DanQing, which contains 100 million image-text pairs collected from Common Crawl. Different from existing datasets, DanQing is curated through a more rigorous selection process, yielding superior data quality. Moreover, DanQing is primarily built from 2024-2025 web data, enabling models to better capture evolving semantic trends and thus offering greater practical utility. We compare DanQing with existing datasets by continual pre-training of the SigLIP2 model. Experimental results show that DanQing consistently achieves superior performance across a range of Chinese downstream tasks, including zero-shot classification, cross-modal retrieval, and LMM-based evaluations. To facilitate further research in Chinese vision-language pre-training, we will open-source the DanQing dataset under the Creative Common CC-BY 4.0 license.

  </details>



- **Cell Behavior Video Classification Challenge, a benchmark for computer vision methods in time-lapse microscopy**  
  Raffaella Fiamma Cabini, Deborah Barkauskas, Guangyu Chen, Zhi-Qi Cheng, David E Cicchetti, Judith Drazba, Rodrigo Fernandez-Gonzalez, Raymond Hawkins, Yujia Hu, Jyoti Kini, et al.  
  _2026-01-15_ · https://arxiv.org/abs/2601.10250v1  
  <details><summary>Abstract</summary>

  The classification of microscopy videos capturing complex cellular behaviors is crucial for understanding and quantifying the dynamics of biological processes over time. However, it remains a frontier in computer vision, requiring approaches that effectively model the shape and motion of objects without rigid boundaries, extract hierarchical spatiotemporal features from entire image sequences rather than static frames, and account for multiple objects within the field of view. To this end, we organized the Cell Behavior Video Classification Challenge (CBVCC), benchmarking 35 methods based on three approaches: classification of tracking-derived features, end-to-end deep learning architectures to directly learn spatiotemporal features from the entire video sequence without explicit cell tracking, or ensembling tracking-derived with image-derived features. We discuss the results achieved by the participants and compare the potential and limitations of each approach, serving as a basis to foster the development of computer vision methods for studying cellular dynamics.

  </details>



- **Attend to what I say: Highlighting relevant content on slides**  
  Megha Mariam K M, C. V. Jawahar  
  _2026-01-15_ · https://arxiv.org/abs/2601.10244v1  
  <details><summary>Abstract</summary>

  Imagine sitting in a presentation, trying to follow the speaker while simultaneously scanning the slides for relevant information. While the entire slide is visible, identifying the relevant regions can be challenging. As you focus on one part of the slide, the speaker moves on to a new sentence, leaving you scrambling to catch up visually. This constant back-and-forth creates a disconnect between what is being said and the most important visual elements, making it hard to absorb key details, especially in fast-paced or content-heavy presentations such as conference talks. This requires an understanding of slides, including text, graphics, and layout. We introduce a method that automatically identifies and highlights the most relevant slide regions based on the speaker's narrative. By analyzing spoken content and matching it with textual or graphical elements in the slides, our approach ensures better synchronization between what listeners hear and what they need to attend to. We explore different ways of solving this problem and assess their success and failure cases. Analyzing multimedia documents is emerging as a key requirement for seamless understanding of content-rich videos, such as educational videos and conference talks, by reducing cognitive strain and improving comprehension. Code and dataset are available at: https://github.com/meghamariamkm2002/Slide_Highlight

  </details>



- **Beyond Inpainting: Unleash 3D Understanding for Precise Camera-Controlled Video Generation**  
  Dong-Yu Chen, Yixin Guo, Shuojin Yang, Tai-Jiang Mu, Shi-Min Hu  
  _2026-01-15_ · https://arxiv.org/abs/2601.10214v1  
  <details><summary>Abstract</summary>

  Camera control has been extensively studied in conditioned video generation; however, performing precisely altering the camera trajectories while faithfully preserving the video content remains a challenging task. The mainstream approach to achieving precise camera control is warping a 3D representation according to the target trajectory. However, such methods fail to fully leverage the 3D priors of video diffusion models (VDMs) and often fall into the Inpainting Trap, resulting in subject inconsistency and degraded generation quality. To address this problem, we propose DepthDirector, a video re-rendering framework with precise camera controllability. By leveraging the depth video from explicit 3D representation as camera-control guidance, our method can faithfully reproduce the dynamic scene of an input video under novel camera trajectories. Specifically, we design a View-Content Dual-Stream Condition mechanism that injects both the source video and the warped depth sequence rendered under the target viewpoint into the pretrained video generation model. This geometric guidance signal enables VDMs to comprehend camera movements and leverage their 3D understanding capabilities, thereby facilitating precise camera control and consistent content generation. Next, we introduce a lightweight LoRA-based video diffusion adapter to train our framework, fully preserving the knowledge priors of VDMs. Additionally, we construct a large-scale multi-camera synchronized dataset named MultiCam-WarpData using Unreal Engine 5, containing 8K videos across 1K dynamic scenes. Extensive experiments show that DepthDirector outperforms existing methods in both camera controllability and visual quality. Our code and dataset will be publicly available.

  </details>



- **RAG-3DSG: Enhancing 3D Scene Graphs with Re-Shot Guided Retrieval-Augmented Generation**  
  Yue Chang, Rufeng Chen, Zhaofan Zhang, Yi Chen, Sihong Xie  
  _2026-01-15_ · https://arxiv.org/abs/2601.10168v1  
  <details><summary>Abstract</summary>

  Open-vocabulary 3D Scene Graph (3DSG) generation can enhance various downstream tasks in robotics, such as manipulation and navigation, by leveraging structured semantic representations. A 3DSG is constructed from multiple images of a scene, where objects are represented as nodes and relationships as edges. However, existing works for open-vocabulary 3DSG generation suffer from both low object-level recognition accuracy and speed, mainly due to constrained viewpoints, occlusions, and redundant surface density. To address these challenges, we propose RAG-3DSG to mitigate aggregation noise through re-shot guided uncertainty estimation and support object-level Retrieval-Augmented Generation (RAG) via reliable low-uncertainty objects. Furthermore, we propose a dynamic downsample-mapping strategy to accelerate cross-image object aggregation with adaptive granularity. Experiments on Replica dataset demonstrate that RAG-3DSG significantly improves node captioning accuracy in 3DSG generation while reducing the mapping time by two-thirds compared to the vanilla version.

  </details>



- **Advancing Adaptive Multi-Stage Video Anomaly Reasoning: A Benchmark Dataset and Method**  
  Chao Huang, Benfeng Wang, Wei Wang, Jie Wen, Li Shen, Wenqi Ren, Yong Xu, Xiaochun Cao  
  _2026-01-15_ · https://arxiv.org/abs/2601.10165v1  
  <details><summary>Abstract</summary>

  Recent progress in reasoning capabilities of Multimodal Large Language Models(MLLMs) has highlighted their potential for performing complex video understanding tasks. However, in the domain of Video Anomaly Detection and Understanding (VAD&U), existing MLLM-based methods are largely limited to anomaly localization or post-hoc description, lacking explicit reasoning processes, risk awareness, and decision-oriented interpretation. To address this gap, we define a new task termed Video Anomaly Reasoning (VAR), which elevates video anomaly analysis from descriptive understanding to structured, multi-stage reasoning. VAR explicitly requires models to perform progressive reasoning over anomalous events before answering anomaly-related questions, encompassing visual perception, causal interpretation, and risk-aware decision making. To support this task, we present a new dataset with 8,641 videos, where each video is annotated with diverse question types corresponding to different reasoning depths, totaling more than 50,000 samples, making it one of the largest datasets for video anomaly. The annotations are based on a structured Perception-Cognition-Action Chain-of-Thought (PerCoAct-CoT), which formalizes domain-specific reasoning priors for video anomaly understanding. This design enables systematic evaluation of multi-stage and adaptive anomaly reasoning. In addition, we propose Anomaly-Aware Group Relative Policy Optimization to further enhance reasoning reliability under weak supervision. Building upon the proposed task and dataset, we develop an end-to-end MLLM-based VAR model termed Vad-R1-Plus, which supports adaptive hierarchical reasoning and risk-aware decision making. Extensive experiments demonstrate that the proposed benchmark and method effectively advance the reasoning capabilities of MLLMs on VAR tasks, outperforming both open-source and proprietary baselines.

  </details>



- **VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation**  
  Sicheng Yang, Zhaohu Xing, Lei Zhu  
  _2026-01-15_ · https://arxiv.org/abs/2601.10124v1  
  <details><summary>Abstract</summary>

  Consistency learning with feature perturbation is a widely used strategy in semi-supervised medical image segmentation. However, many existing perturbation methods rely on dropout, and thus require a careful manual tuning of the dropout rate, which is a sensitive hyperparameter and often difficult to optimize and may lead to suboptimal regularization. To overcome this limitation, we propose VQ-Seg, the first approach to employ vector quantization (VQ) to discretize the feature space and introduce a novel and controllable Quantized Perturbation Module (QPM) that replaces dropout. Our QPM perturbs discrete representations by shuffling the spatial locations of codebook indices, enabling effective and controllable regularization. To mitigate potential information loss caused by quantization, we design a dual-branch architecture where the post-quantization feature space is shared by both image reconstruction and segmentation tasks. Moreover, we introduce a Post-VQ Feature Adapter (PFA) to incorporate guidance from a foundation model (FM), supplementing the high-level semantic information lost during quantization. Furthermore, we collect a large-scale Lung Cancer (LC) dataset comprising 828 CT scans annotated for central-type lung carcinoma. Extensive experiments on the LC dataset and other public benchmarks demonstrate the effectiveness of our method, which outperforms state-of-the-art approaches. Code available at: https://github.com/script-Yang/VQ-Seg.

  </details>



- **MathDoc: Benchmarking Structured Extraction and Active Refusal on Noisy Mathematics Exam Papers**  
  Chenyue Zhou, Jiayi Tuo, Shitong Qin, Wei Dai, Mingxuan Wang, Ziwei Zhao, Duoyang Li, Shiyang Su, Yanxi Lu, Yanbiao Ma  
  _2026-01-15_ · https://arxiv.org/abs/2601.10104v1  
  <details><summary>Abstract</summary>

  The automated extraction of structured questions from paper-based mathematics exams is fundamental to intelligent education, yet remains challenging in real-world settings due to severe visual noise. Existing benchmarks mainly focus on clean documents or generic layout analysis, overlooking both the structural integrity of mathematical problems and the ability of models to actively reject incomplete inputs. We introduce MathDoc, the first benchmark for document-level information extraction from authentic high school mathematics exam papers. MathDoc contains \textbf{3,609} carefully curated questions with real-world artifacts and explicitly includes unrecognizable samples to evaluate active refusal behavior. We propose a multi-dimensional evaluation framework covering stem accuracy, visual similarity, and refusal capability. Experiments on SOTA MLLMs, including Qwen3-VL and Gemini-2.5-Pro, show that although end-to-end models achieve strong extraction performance, they consistently fail to refuse illegible inputs, instead producing confident but invalid outputs. These results highlight a critical gap in current MLLMs and establish MathDoc as a benchmark for assessing model reliability under degraded document conditions. Our project repository is available at \href{https://github.com/winnk123/papers/tree/master}{GitHub repository}

  </details>



- **InfoSculpt: Sculpting the Latent Space for Generalized Category Discovery**  
  Wenwen Liao, Hang Ruan, Jianbo Yu, Yuansong Wang, Qingchao Jiang, Xiaofeng Yang  
  _2026-01-15_ · https://arxiv.org/abs/2601.10098v1  
  <details><summary>Abstract</summary>

  Generalized Category Discovery (GCD) aims to classify instances from both known and novel categories within a large-scale unlabeled dataset, a critical yet challenging task for real-world, open-world applications. However, existing methods often rely on pseudo-labeling, or two-stage clustering, which lack a principled mechanism to explicitly disentangle essential, category-defining signals from instance-specific noise. In this paper, we address this fundamental limitation by re-framing GCD from an information-theoretic perspective, grounded in the Information Bottleneck (IB) principle. We introduce InfoSculpt, a novel framework that systematically sculpts the representation space by minimizing a dual Conditional Mutual Information (CMI) objective. InfoSculpt uniquely combines a Category-Level CMI on labeled data to learn compact and discriminative representations for known classes, and a complementary Instance-Level CMI on all data to distill invariant features by compressing augmentation-induced noise. These two objectives work synergistically at different scales to produce a disentangled and robust latent space where categorical information is preserved while noisy, instance-specific details are discarded. Extensive experiments on 8 benchmarks demonstrate that InfoSculpt validating the effectiveness of our information-theoretic approach.

  </details>



- **V-Zero: Self-Improving Multimodal Reasoning with Zero Annotation**  
  Han Wang, Yi Yang, Jingyuan Hu, Minfeng Zhu, Wei Chen  
  _2026-01-15_ · https://arxiv.org/abs/2601.10094v1  
  <details><summary>Abstract</summary>

  Recent advances in multimodal learning have significantly enhanced the reasoning capabilities of vision-language models (VLMs). However, state-of-the-art approaches rely heavily on large-scale human-annotated datasets, which are costly and time-consuming to acquire. To overcome this limitation, we introduce V-Zero, a general post-training framework that facilitates self-improvement using exclusively unlabeled images. V-Zero establishes a co-evolutionary loop by instantiating two distinct roles: a Questioner and a Solver. The Questioner learns to synthesize high-quality, challenging questions by leveraging a dual-track reasoning reward that contrasts intuitive guesses with reasoned results. The Solver is optimized using pseudo-labels derived from majority voting over its own sampled responses. Both roles are trained iteratively via Group Relative Policy Optimization (GRPO), driving a cycle of mutual enhancement. Remarkably, without a single human annotation, V-Zero achieves consistent performance gains on Qwen2.5-VL-7B-Instruct, improving visual mathematical reasoning by +1.7 and general vision-centric by +2.6, demonstrating the potential of self-improvement in multimodal systems. Code is available at https://github.com/SatonoDia/V-Zero

  </details>



- **Difficulty-guided Sampling: Bridging the Target Gap between Dataset Distillation and Downstream Tasks**  
  Mingzhuo Li, Guang Li, Linfeng Ye, Jiafeng Mao, Takahiro Ogawa, Konstantinos N. Plataniotis, Miki Haseyama  
  _2026-01-15_ · https://arxiv.org/abs/2601.10090v1  
  <details><summary>Abstract</summary>

  In this paper, we propose difficulty-guided sampling (DGS) to bridge the target gap between the distillation objective and the downstream task, therefore improving the performance of dataset distillation. Deep neural networks achieve remarkable performance but have time and storage-consuming training processes. Dataset distillation is proposed to generate compact, high-quality distilled datasets, enabling effective model training while maintaining downstream performance. Existing approaches typically focus on features extracted from the original dataset, overlooking task-specific information, which leads to a target gap between the distillation objective and the downstream task. We propose leveraging characteristics that benefit the downstream training into data distillation to bridge this gap. Focusing on the downstream task of image classification, we introduce the concept of difficulty and propose DGS as a plug-in post-stage sampling module. Following the specific target difficulty distribution, the final distilled dataset is sampled from image pools generated by existing methods. We also propose difficulty-aware guidance (DAG) to explore the effect of difficulty in the generation process. Extensive experiments across multiple settings demonstrate the effectiveness of the proposed methods. It also highlights the broader potential of difficulty for diverse downstream tasks.

  </details>



- **CoF-T2I: Video Models as Pure Visual Reasoners for Text-to-Image Generation**  
  Chengzhuo Tong, Mingkun Chang, Shenglong Zhang, Yuran Wang, Cheng Liang, Zhizheng Zhao, Ruichuan An, Bohan Zeng, Yang Shi, Yifan Dai, et al.  
  _2026-01-15_ · https://arxiv.org/abs/2601.10061v1  
  <details><summary>Abstract</summary>

  Recent video generation models have revealed the emergence of Chain-of-Frame (CoF) reasoning, enabling frame-by-frame visual inference. With this capability, video models have been successfully applied to various visual tasks (e.g., maze solving, visual puzzles). However, their potential to enhance text-to-image (T2I) generation remains largely unexplored due to the absence of a clearly defined visual reasoning starting point and interpretable intermediate states in the T2I generation process. To bridge this gap, we propose CoF-T2I, a model that integrates CoF reasoning into T2I generation via progressive visual refinement, where intermediate frames act as explicit reasoning steps and the final frame is taken as output. To establish such an explicit generation process, we curate CoF-Evol-Instruct, a dataset of CoF trajectories that model the generation process from semantics to aesthetics. To further improve quality and avoid motion artifacts, we enable independent encoding operation for each frame. Experiments show that CoF-T2I significantly outperforms the base video model and achieves competitive performance on challenging benchmarks, reaching 0.86 on GenEval and 7.468 on Imagine-Bench. These results indicate the substantial promise of video models for advancing high-quality text-to-image generation.

  </details>



- **UEOF: A Benchmark Dataset for Underwater Event-Based Optical Flow**  
  Nick Truong, Pritam P. Karmokar, William J. Beksi  
  _2026-01-15_ · https://arxiv.org/abs/2601.10054v1  
  <details><summary>Abstract</summary>

  Underwater imaging is fundamentally challenging due to wavelength-dependent light attenuation, strong scattering from suspended particles, turbidity-induced blur, and non-uniform illumination. These effects impair standard cameras and make ground-truth motion nearly impossible to obtain. On the other hand, event cameras offer microsecond resolution and high dynamic range. Nonetheless, progress on investigating event cameras for underwater environments has been limited due to the lack of datasets that pair realistic underwater optics with accurate optical flow. To address this problem, we introduce the first synthetic underwater benchmark dataset for event-based optical flow derived from physically-based ray-traced RGBD sequences. Using a modern video-to-event pipeline applied to rendered underwater videos, we produce realistic event data streams with dense ground-truth flow, depth, and camera motion. Moreover, we benchmark state-of-the-art learning-based and model-based optical flow prediction methods to understand how underwater light transport affects event formation and motion estimation accuracy. Our dataset establishes a new baseline for future development and evaluation of underwater event-based perception algorithms. The source code and dataset for this project are publicly available at https://robotic-vision-lab.github.io/ueof.

  </details>



- **VERHallu: Evaluating and Mitigating Event Relation Hallucination in Video Large Language Models**  
  Zefan Zhang, Kehua Zhu, Shijie Jiang, Hongyuan Lu, Shengkai Sun, Tian Bai  
  _2026-01-15_ · https://arxiv.org/abs/2601.10010v1  
  <details><summary>Abstract</summary>

  Video Large Language Models (VideoLLMs) exhibit various types of hallucinations. Existing research has primarily focused on hallucinations involving the presence of events, objects, and scenes in videos, while largely neglecting event relation hallucination. In this paper, we introduce a novel benchmark for evaluating the Video Event Relation Hallucination, named VERHallu. This benchmark focuses on causal, temporal, and subevent relations between events, encompassing three types of tasks: relation classification, question answering, and counterfactual question answering, for a comprehensive evaluation of event relation hallucination. Additionally, it features counterintuitive video scenarios that deviate from typical pretraining distributions, with each sample accompanied by human-annotated candidates covering both vision-language and pure language biases. Our analysis reveals that current state-of-the-art VideoLLMs struggle with dense-event relation reasoning, often relying on prior knowledge due to insufficient use of frame-level cues. Although these models demonstrate strong grounding capabilities for key events, they often overlook the surrounding subevents, leading to an incomplete and inaccurate understanding of event relations. To tackle this, we propose a Key-Frame Propagating (KFP) strategy, which reallocates frame-level attention within intermediate layers to enhance multi-event understanding. Experiments show it effectively mitigates the event relation hallucination without affecting inference speed.

  </details>



- **OT-Drive: Out-of-Distribution Off-Road Traversable Area Segmentation via Optimal Transport**  
  Zhihua Zhao, Guoqiang Li, Chen Min, Kangping Lu  
  _2026-01-15_ · https://arxiv.org/abs/2601.09952v1  
  <details><summary>Abstract</summary>

  Reliable traversable area segmentation in unstructured environments is critical for planning and decision-making in autonomous driving. However, existing data-driven approaches often suffer from degraded segmentation performance in out-of-distribution (OOD) scenarios, consequently impairing downstream driving tasks. To address this issue, we propose OT-Drive, an Optimal Transport--driven multi-modal fusion framework. The proposed method formulates RGB and surface normal fusion as a distribution transport problem. Specifically, we design a novel Scene Anchor Generator (SAG) to decompose scene information into the joint distribution of weather, time-of-day, and road type, thereby constructing semantic anchors that can generalize to unseen scenarios. Subsequently, we design an innovative Optimal Transport-based multi-modal fusion module (OT Fusion) to transport RGB and surface normal features onto the manifold defined by the semantic anchors, enabling robust traversable area segmentation under OOD scenarios. Experimental results demonstrate that our method achieves 95.16% mIoU on ORFD OOD scenarios, outperforming prior methods by 6.35%, and 89.79% mIoU on cross-dataset transfer tasks, surpassing baselines by 13.99%.These results indicate that the proposed model can attain strong OOD generalization with only limited training data, substantially enhancing its practicality and efficiency for real-world deployment.

  </details>



- **The Algorithmic Gaze: An Audit and Ethnography of the LAION-Aesthetics Predictor Model**  
  Jordan Taylor, William Agnew, Maarten Sap, Sarah E. Fox, Haiyi Zhu  
  _2026-01-14_ · https://arxiv.org/abs/2601.09896v1  
  <details><summary>Abstract</summary>

  Visual generative AI models are trained using a one-size-fits-all measure of aesthetic appeal. However, what is deemed "aesthetic" is inextricably linked to personal taste and cultural values, raising the question of whose taste is represented in visual generative AI models. In this work, we study an aesthetic evaluation model--LAION Aesthetic Predictor (LAP)--that is widely used to curate datasets to train visual generative image models, like Stable Diffusion, and evaluate the quality of AI-generated images. To understand what LAP measures, we audited the model across three datasets. First, we examined the impact of aesthetic filtering on the LAION-Aesthetics Dataset (approximately 1.2B images), which was curated from LAION-5B using LAP. We find that the LAP disproportionally filters in images with captions mentioning women, while filtering out images with captions mentioning men or LGBTQ+ people. Then, we used LAP to score approximately 330k images across two art datasets, finding the model rates realistic images of landscapes, cityscapes, and portraits from western and Japanese artists most highly. In doing so, the algorithmic gaze of this aesthetic evaluation model reinforces the imperial and male gazes found within western art history. In order to understand where these biases may have originated, we performed a digital ethnography of public materials related to the creation of LAP. We find that the development of LAP reflects the biases we found in our audits, such as the aesthetic scores used to train LAP primarily coming from English-speaking photographers and western AI-enthusiasts. In response, we discuss how aesthetic evaluation can perpetuate representational harms and call on AI developers to shift away from prescriptive measures of "aesthetics" toward more pluralistic evaluation.

  </details>



- **MedVL-SAM2: A unified 3D medical vision-language model for multimodal reasoning and prompt-driven segmentation**  
  Yang Xing, Jiong Wu, Savas Ozdemir, Ying Zhang, Yang Yang, Wei Shao, Kuang Gong  
  _2026-01-14_ · https://arxiv.org/abs/2601.09879v1  
  <details><summary>Abstract</summary>

  Recent progress in medical vision-language models (VLMs) has achieved strong performance on image-level text-centric tasks such as report generation and visual question answering (VQA). However, achieving fine-grained visual grounding and volumetric spatial reasoning in 3D medical VLMs remains challenging, particularly when aiming to unify these capabilities within a single, generalizable framework. To address this challenge, we proposed MedVL-SAM2, a unified 3D medical multimodal model that concurrently supports report generation, VQA, and multi-paradigm segmentation, including semantic, referring, and interactive segmentation. MedVL-SAM2 integrates image-level reasoning and pixel-level perception through a cohesive architecture tailored for 3D medical imaging, and incorporates a SAM2-based volumetric segmentation module to enable precise multi-granular spatial reasoning. The model is trained in a multi-stage pipeline: it is first pre-trained on a large-scale corpus of 3D CT image-text pairs to align volumetric visual features with radiology-language embeddings. It is then jointly optimized with both language-understanding and segmentation objectives using a comprehensive 3D CT segmentation dataset. This joint training enables flexible interaction via language, point, or box prompts, thereby unifying high-level visual reasoning with spatially precise localization. Our unified architecture delivers state-of-the-art performance across report generation, VQA, and multiple 3D segmentation tasks. Extensive analyses further show that the model provides reliable 3D visual grounding, controllable interactive segmentation, and robust cross-modal reasoning, demonstrating that high-level semantic reasoning and precise 3D localization can be jointly achieved within a unified 3D medical VLM.

  </details>



- **Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP**  
  Anant Mehta, Xiyuan Wei, Xingyu Chen, Tianbao Yang  
  _2026-01-14_ · https://arxiv.org/abs/2601.09859v1  
  <details><summary>Abstract</summary>

  CLIP has become a cornerstone of multimodal representation learning, yet improving its performance typically requires a prohibitively costly process of training from scratch on billions of samples. We ask a different question: Can we improve the performance of open-weight CLIP models across various downstream tasks using only existing self-supervised datasets? Unlike supervised fine-tuning, which adapts a pretrained model to a single downstream task, our setting seeks to improve general performance across various tasks. However, as both our experiments and prior studies reveal, simply applying standard training protocols starting from an open-weight CLIP model often fails, leading to performance degradation. In this paper, we introduce TuneCLIP, a self-supervised fine-tuning framework that overcomes the performance degradation. TuneCLIP has two key components: (1) a warm-up stage of recovering optimization statistics to reduce cold-start bias, inspired by theoretical analysis, and (2) a fine-tuning stage of optimizing a new contrastive loss to mitigate the penalization on false negative pairs. Our extensive experiments show that TuneCLIP consistently improves performance across model architectures and scales. Notably, it elevates leading open-weight models like SigLIP (ViT-B/16), achieving gains of up to +2.5% on ImageNet and related out-of-distribution benchmarks, and +1.2% on the highly competitive DataComp benchmark, setting a new strong baseline for efficient post-pretraining adaptation.

  </details>



- **Explainable Deep Learning for Pediatric Pneumonia Detection in Chest X-Ray Images**  
  Adil O. Khadidos, Aziida Nanyonga, Alaa O. Khadidos, Olfat M. Mirza, Mustafa Tahsin Yilmaz  
  _2026-01-14_ · https://arxiv.org/abs/2601.09814v1  
  <details><summary>Abstract</summary>

  Background: Pneumonia remains a leading cause of morbidity and mortality among children worldwide, emphasizing the need for accurate and efficient diagnostic support tools. Deep learning has shown strong potential in medical image analysis, particularly for chest X-ray interpretation. This study compares two state-of-the-art convolutional neural network (CNN) architectures for automated pediatric pneumonia detection. Methods: A publicly available dataset of 5,863 pediatric chest X-ray images was used. Images were preprocessed through normalization, resizing, and data augmentation to enhance generalization. DenseNet121 and EfficientNet-B0 were fine-tuned using pretrained ImageNet weights under identical training settings. Performance was evaluated using accuracy, F1-score, Matthews Correlation Coefficient (MCC), and recall. Model explainability was incorporated using Gradient-weighted Class Activation Mapping (Grad-CAM) and Local Interpretable Model-agnostic Explanations (LIME) to visualize image regions influencing predictions. Results: EfficientNet-B0 outperformed DenseNet121, achieving an accuracy of 84.6%, F1-score of 0.8899, and MCC of 0.6849. DenseNet121 achieved 79.7% accuracy, an F1-score of 0.8597, and MCC of 0.5852. Both models demonstrated high recall values above 0.99, indicating strong sensitivity to pneumonia detection. Grad-CAM and LIME visualizations showed consistent focus on clinically relevant lung regions, supporting the reliability of model decisions. Conclusions: EfficientNet-B0 provided a more balanced and computationally efficient performance compared to DenseNet121, making it a strong candidate for clinical deployment. The integration of explainability techniques enhances transparency and trustworthiness in AI-assisted pediatric pneumonia diagnosis.

  </details>



- **LCF3D: A Robust and Real-Time Late-Cascade Fusion Framework for 3D Object Detection in Autonomous Driving**  
  Carlo Sgaravatti, Riccardo Pieroni, Matteo Corno, Sergio M. Savaresi, Luca Magri, Giacomo Boracchi  
  _2026-01-14_ · https://arxiv.org/abs/2601.09812v1  
  <details><summary>Abstract</summary>

  Accurately localizing 3D objects like pedestrians, cyclists, and other vehicles is essential in Autonomous Driving. To ensure high detection performance, Autonomous Vehicles complement RGB cameras with LiDAR sensors, but effectively combining these data sources for 3D object detection remains challenging. We propose LCF3D, a novel sensor fusion framework that combines a 2D object detector on RGB images with a 3D object detector on LiDAR point clouds. By leveraging multimodal fusion principles, we compensate for inaccuracies in the LiDAR object detection network. Our solution combines two key principles: (i) late fusion, to reduce LiDAR False Positives by matching LiDAR 3D detections with RGB 2D detections and filtering out unmatched LiDAR detections; and (ii) cascade fusion, to recover missed objects from LiDAR by generating new 3D frustum proposals corresponding to unmatched RGB detections. Experiments show that LCF3D is beneficial for domain generalization, as it turns out to be successful in handling different sensor configurations between training and testing domains. LCF3D achieves significant improvements over LiDAR-based methods, particularly for challenging categories like pedestrians and cyclists in the KITTI dataset, as well as motorcycles and bicycles in nuScenes. Code can be downloaded from: https://github.com/CarloSgaravatti/LCF3D.

  </details>



- **A continental-scale dataset of ground beetles with high-resolution images and validated morphological trait measurements**  
  S M Rayeed, Mridul Khurana, Alyson East, Isadora E. Fluck, Elizabeth G. Campolongo, Samuel Stevens, Iuliia Zarubiieva, Scott C. Lowe, Michael W. Denslow, Evan D. Donoso, et al.  
  _2026-01-14_ · https://arxiv.org/abs/2601.10687v1  
  <details><summary>Abstract</summary>

  Despite the ecological significance of invertebrates, global trait databases remain heavily biased toward vertebrates and plants, limiting comprehensive ecological analyses of high-diversity groups like ground beetles. Ground beetles (Coleoptera: Carabidae) serve as critical bioindicators of ecosystem health, providing valuable insights into biodiversity shifts driven by environmental changes. While the National Ecological Observatory Network (NEON) maintains an extensive collection of carabid specimens from across the United States, these primarily exist as physical collections, restricting widespread research access and large-scale analysis. To address these gaps, we present a multimodal dataset digitizing over 13,200 NEON carabids from 30 sites spanning the continental US and Hawaii through high-resolution imaging, enabling broader access and computational analysis. The dataset includes digitally measured elytra length and width of each specimen, establishing a foundation for automated trait extraction using AI. Validated against manual measurements, our digital trait extraction achieves sub-millimeter precision, ensuring reliability for ecological and computational studies. By addressing invertebrate under-representation in trait databases, this work supports AI-driven tools for automated species identification and trait-based research, fostering advancements in biodiversity monitoring and conservation.

  </details>



- **Self-Supervised Animal Identification for Long Videos**  
  Xuyang Fang, Sion Hannuna, Edwin Simpson, Neill Campbell  
  _2026-01-14_ · https://arxiv.org/abs/2601.09663v1  
  <details><summary>Abstract</summary>

  Identifying individual animals in long-duration videos is essential for behavioral ecology, wildlife monitoring, and livestock management. Traditional methods require extensive manual annotation, while existing self-supervised approaches are computationally demanding and ill-suited for long sequences due to memory constraints and temporal error propagation. We introduce a highly efficient, self-supervised method that reframes animal identification as a global clustering task rather than a sequential tracking problem. Our approach assumes a known, fixed number of individuals within a single video -- a common scenario in practice -- and requires only bounding box detections and the total count. By sampling pairs of frames, using a frozen pre-trained backbone, and employing a self-bootstrapping mechanism with the Hungarian algorithm for in-batch pseudo-label assignment, our method learns discriminative features without identity labels. We adapt a Binary Cross Entropy loss from vision-language models, enabling state-of-the-art accuracy ($>$97\%) while consuming less than 1 GB of GPU memory per batch -- an order of magnitude less than standard contrastive methods. Evaluated on challenging real-world datasets (3D-POP pigeons and 8-calves feeding videos), our framework matches or surpasses supervised baselines trained on over 1,000 labeled frames, effectively removing the manual annotation bottleneck. This work enables practical, high-accuracy animal identification on consumer-grade hardware, with broad applicability in resource-constrained research settings. All code written for this paper are \href{https://huggingface.co/datasets/tonyFang04/8-calves}{here}.

  </details>



- **Image2Garment: Simulation-ready Garment Generation from a Single Image**  
  Selim Emir Can, Jan Ackermann, Kiyohiro Nakayama, Ruofan Liu, Tong Wu, Yang Zheng, Hugo Bertiche, Menglei Chai, Thabo Beeler, Gordon Wetzstein  
  _2026-01-14_ · https://arxiv.org/abs/2601.09658v1  
  <details><summary>Abstract</summary>

  Estimating physically accurate, simulation-ready garments from a single image is challenging due to the absence of image-to-physics datasets and the ill-posed nature of this problem. Prior methods either require multi-view capture and expensive differentiable simulation or predict only garment geometry without the material properties required for realistic simulation. We propose a feed-forward framework that sidesteps these limitations by first fine-tuning a vision-language model to infer material composition and fabric attributes from real images, and then training a lightweight predictor that maps these attributes to the corresponding physical fabric parameters using a small dataset of material-physics measurements. Our approach introduces two new datasets (FTAG and T2P) and delivers simulation-ready garments from a single image without iterative optimization. Experiments show that our estimator achieves superior accuracy in material composition estimation and fabric attribute prediction, and by passing them through our physics parameter estimator, we further achieve higher-fidelity simulations compared to state-of-the-art image-to-garment methods.

  </details>



- **AquaFeat+: an Underwater Vision Learning-based Enhancement Method for Object Detection, Classification, and Tracking**  
  Emanuel da Costa Silva, Tatiana Taís Schein, José David García Ramos, Eduardo Lawson da Silva, Stephanie Loi Brião, Felipe Gomes de Oliveira, Paulo Lilles Jorge Drews-Jr  
  _2026-01-14_ · https://arxiv.org/abs/2601.09652v1  
  <details><summary>Abstract</summary>

  Underwater video analysis is particularly challenging due to factors such as low lighting, color distortion, and turbidity, which compromise visual data quality and directly impact the performance of perception modules in robotic applications. This work proposes AquaFeat+, a plug-and-play pipeline designed to enhance features specifically for automated vision tasks, rather than for human perceptual quality. The architecture includes modules for color correction, hierarchical feature enhancement, and an adaptive residual output, which are trained end-to-end and guided directly by the loss function of the final application. Trained and evaluated in the FishTrack23 dataset, AquaFeat+ achieves significant improvements in object detection, classification, and tracking metrics, validating its effectiveness for enhancing perception tasks in underwater robotic applications.

  </details>



- **PersonalAlign: Hierarchical Implicit Intent Alignment for Personalized GUI Agent with Long-Term User-Centric Records**  
  Yibo Lyu, Gongwei Chen, Rui Shao, Weili Guan, Liqiang Nie  
  _2026-01-14_ · https://arxiv.org/abs/2601.09636v1  
  <details><summary>Abstract</summary>

  While GUI agents have shown strong performance under explicit and completion instructions, real-world deployment requires aligning with users' more complex implicit intents. In this work, we highlight Hierarchical Implicit Intent Alignment for Personalized GUI Agent (PersonalAlign), a new agent task that requires agents to leverage long-term user records as persistent context to resolve omitted preferences in vague instructions and anticipate latent routines by user state for proactive assistance. To facilitate this study, we introduce AndroidIntent, a benchmark designed to evaluate agents' ability in resolving vague instructions and providing proactive suggestions through reasoning over long-term user records. We annotated 775 user-specific preferences and 215 routines from 20k long-term records across different users for evaluation. Furthermore, we introduce Hierarchical Intent Memory Agent (HIM-Agent), which maintains a continuously updating personal memory and hierarchically organizes user preferences and routines for personalization. Finally, we evaluate a range of GUI agents on AndroidIntent, including GPT-5, Qwen3-VL, and UI-TARS, further results show that HIM-Agent significantly improves both execution and proactive performance by 15.7% and 7.3%.

  </details>



- **CogRail: Benchmarking VLMs in Cognitive Intrusion Perception for Intelligent Railway Transportation Systems**  
  Yonglin Tian, Qiyao Zhang, Wei Xu, Yutong Wang, Yihao Wu, Xinyi Li, Xingyuan Dai, Hui Zhang, Zhiyong Cui, Baoqing Guo, et al.  
  _2026-01-14_ · https://arxiv.org/abs/2601.09613v1  
  <details><summary>Abstract</summary>

  Accurate and early perception of potential intrusion targets is essential for ensuring the safety of railway transportation systems. However, most existing systems focus narrowly on object classification within fixed visual scopes and apply rule-based heuristics to determine intrusion status, often overlooking targets that pose latent intrusion risks. Anticipating such risks requires the cognition of spatial context and temporal dynamics for the object of interest (OOI), which presents challenges for conventional visual models. To facilitate deep intrusion perception, we introduce a novel benchmark, CogRail, which integrates curated open-source datasets with cognitively driven question-answer annotations to support spatio-temporal reasoning and prediction. Building upon this benchmark, we conduct a systematic evaluation of state-of-the-art visual-language models (VLMs) using multimodal prompts to identify their strengths and limitations in this domain. Furthermore, we fine-tune VLMs for better performance and propose a joint fine-tuning framework that integrates three core tasks, position perception, movement prediction, and threat analysis, facilitating effective adaptation of general-purpose foundation models into specialized models tailored for cognitive intrusion perception. Extensive experiments reveal that current large-scale multimodal models struggle with the complex spatial-temporal reasoning required by the cognitive intrusion perception task, underscoring the limitations of existing foundation models in this safety-critical domain. In contrast, our proposed joint fine-tuning framework significantly enhances model performance by enabling targeted adaptation to domain-specific reasoning demands, highlighting the advantages of structured multi-task learning in improving both accuracy and interpretability. Code will be available at https://github.com/Hub-Tian/CogRail.

  </details>



- **Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server**  
  Yue Yao, Ruining Yang, Tom Gedeon  
  _2026-01-14_ · https://arxiv.org/abs/2601.09531v1  
  <details><summary>Abstract</summary>

  We explore a situation in which the target domain is accessible, but real-time data annotation is not feasible. Instead, we would like to construct an alternative training set from a large-scale data server so that a competitive model can be obtained. For this problem, because the target domain usually exhibits distinct modes (i.e., semantic clusters representing data distribution), if the training set does not contain these target modes, the model performance would be compromised. While prior existing works improve algorithms iteratively, our research explores the often-overlooked potential of optimizing the structure of the data server. Inspired by the hierarchical nature of web search engines, we introduce a hierarchical data server, together with a bipartite mode matching algorithm (BMM) to align source and target modes. For each target mode, we look in the server data tree for the best mode match, which might be large or small in size. Through bipartite matching, we aim for all target modes to be optimally matched with source modes in a one-on-one fashion. Compared with existing training set search algorithms, we show that the matched server modes constitute training sets that have consistently smaller domain gaps with the target domain across object re-identification (re-ID) and detection tasks. Consequently, models trained on our searched training sets have higher accuracy than those trained otherwise. BMM allows data-centric unsupervised domain adaptation (UDA) orthogonal to existing model-centric UDA methods. By combining the BMM with existing UDA methods like pseudo-labeling, further improvement is observed.

  </details>


