# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-28 07:11 UTC_

Total papers shown: **50**


---

- **RefAlign: Representation Alignment for Reference-to-Video Generation**  
  Lei Wang, YuXin Song, Ge Wu, Haocheng Feng, Hang Zhou, Jingdong Wang, Yaxing Wang, jian Yang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25743v1  
  <details><summary>Abstract</summary>

  Reference-to-video (R2V) generation is a controllable video synthesis paradigm that constrains the generation process using both text prompts and reference images, enabling applications such as personalized advertising and virtual try-on. In practice, existing R2V methods typically introduce additional high-level semantic or cross-modal features alongside the VAE latent representation of the reference image and jointly feed them into the diffusion Transformer (DiT). These auxiliary representations provide semantic guidance and act as implicit alignment signals, which can partially alleviate pixel-level information leakage in the VAE latent space. However, they may still struggle to address copy--paste artifacts and multi-subject confusion caused by modality mismatch across heterogeneous encoder features. In this paper, we propose RefAlign, a representation alignment framework that explicitly aligns DiT reference-branch features to the semantic space of a visual foundation model (VFM). The core of RefAlign is a reference alignment loss that pulls the reference features and VFM features of the same subject closer to improve identity consistency, while pushing apart the corresponding features of different subjects to enhance semantic discriminability. This simple yet effective strategy is applied only during training, incurring no inference-time overhead, and achieves a better balance between text controllability and reference fidelity. Extensive experiments on the OpenS2V-Eval benchmark demonstrate that RefAlign outperforms current state-of-the-art methods in TotalScore, validating the effectiveness of explicit reference alignment for R2V tasks.

  </details>



- **Vega: Learning to Drive with Natural Language Instructions**  
  Sicheng Zuo, Yuxuan Li, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25741v1  
  <details><summary>Abstract</summary>

  Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified Vision-Language-World-Action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.

  </details>



- **Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving**  
  Zehao Wang, Huaide Jiang, Shuaiwu Dong, Yuping Wang, Hang Qiu, Jiachen Li  
  _2026-03-26_ · https://arxiv.org/abs/2603.25740v1  
  <details><summary>Abstract</summary>

  Human driving behavior is inherently personal, which is shaped by long-term habits and influenced by short-term intentions. Individuals differ in how they accelerate, brake, merge, yield, and overtake across diverse situations. However, existing end-to-end autonomous driving systems either optimize for generic objectives or rely on fixed driving modes, lacking the ability to adapt to individual preferences or interpret natural language intent. To address this gap, we propose Drive My Way (DMW), a personalized Vision-Language-Action (VLA) driving framework that aligns with users' long-term driving habits and adapts to real-time user instructions. DMW learns a user embedding from our personalized driving dataset collected across multiple real drivers and conditions the policy on this embedding during planning, while natural language instructions provide additional short-term guidance. Closed-loop evaluation on the Bench2Drive benchmark demonstrates that DMW improves style instruction adaptation, and user studies show that its generated behaviors are recognizable as each driver's own style, highlighting personalization as a key capability for human-centered autonomous driving. Our data and code are available at https://dmw-cvpr.github.io/.

  </details>



- **PSDesigner: Automated Graphic Design with a Human-Like Creative Workflow**  
  Xincheng Shuai, Song Tang, Yutong Huang, Henghui Ding, Dacheng Tao  
  _2026-03-26_ · https://arxiv.org/abs/2603.25738v1  
  <details><summary>Abstract</summary>

  Graphic design is a creative and innovative process that plays a crucial role in applications such as e-commerce and advertising. However, developing an automated design system that can faithfully translate user intentions into editable design files remains an open challenge. Although recent studies have leveraged powerful text-to-image models and MLLMs to assist graphic design, they typically simplify professional workflows, resulting in limited flexibility and intuitiveness. To address these limitations, we propose PSDesigner, an automated graphic design system that emulates the creative workflow of human designers. Building upon multiple specialized components, PSDesigner collects theme-related assets based on user instructions, and autonomously infers and executes tool calls to manipulate design files, such as integrating new assets or refining inferior elements. To endow the system with strong tool-use capabilities, we construct a design dataset, CreativePSD, which contains a large amount of high-quality PSD design files annotated with operation traces across a wide range of design scenarios and artistic styles, enabling models to learn expert design procedures. Extensive experiments demonstrate that PSDesigner outperforms existing methods across diverse graphic design tasks, empowering non-specialists to conveniently create production-quality designs.

  </details>



- **How good was my shot? Quantifying Player Skill Level in Table Tennis**  
  Akihiro Kubota, Tomoya Hasegawa, Ryo Kawahara, Ko Nishino  
  _2026-03-26_ · https://arxiv.org/abs/2603.25736v1  
  <details><summary>Abstract</summary>

  Gauging an individual's skill level is crucial, as it inherently shapes their behavior. Quantifying skill, however, is challenging because it is latent to the observed actions. To explore skill understanding in human behavior, we focus on dyadic sports -- specifically table tennis -- where skill manifests not just in complex movements, but in the subtle nuances of execution conditioned on game context. Our key idea is to learn a generative model of each player's tactical racket strokes and jointly embed them in a common latent space that encodes individual characteristics, including those pertaining to skill levels. By training these player models on a large-scale dataset of 3D-reconstructed professional matches and conditioning them on comprehensive game context -- including player positioning and opponent behaviors -- the models capture individual tactical identities within their latent space. We probe this learned player space and find that it reflects distinct play styles and attributes that collectively represent skill. By training a simple relative ranking network on these embeddings, we demonstrate that both relative and absolute skill predictions can be achieved. These results demonstrate that the learned player space effectively quantifies skill levels, providing a foundation for automated skill assessment in complex, interactive behaviors.

  </details>



- **SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding**  
  Jiwook Han, Geo Ahn, Youngrae Kim, Jinwoo Choi  
  _2026-03-26_ · https://arxiv.org/abs/2603.25733v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have shown strong performance on Video Temporal Grounding (VTG). However, their coarse recognition capabilities are insufficient for fine-grained temporal understanding, making task-specific fine-tuning indispensable. This fine-tuning causes models to memorize dataset-specific shortcuts rather than faithfully grounding in the actual visual content, leading to poor Out-of-Domain (OOD) generalization. Object-centric learning offers a promising remedy by decomposing scenes into entity-level representations, but existing approaches require re-running the entire multi-stage training pipeline from scratch. We propose SlotVTG, a framework that steers MLLMs toward object-centric, input-grounded visual reasoning at minimal cost. SlotVTG introduces a lightweight slot adapter that decomposes visual tokens into abstract slots via slot attention and reconstructs the original sequence, where objectness priors from a self-supervised vision model encourage semantically coherent slot formation. Cross-domain evaluation on standard VTG benchmarks demonstrates that our approach significantly improves OOD robustness while maintaining competitive In-Domain (ID) performance with minimal overhead.

  </details>



- **BizGenEval: A Systematic Benchmark for Commercial Visual Content Generation**  
  Yan Li, Zezi Zeng, Ziwei Zhou, Xin Gao, Muzhao Tian, Yifan Yang, Mingxi Cheng, Qi Dai, Yuqing Yang, Lili Qiu, et al.  
  _2026-03-26_ · https://arxiv.org/abs/2603.25732v1  
  <details><summary>Abstract</summary>

  Recent advances in image generation models have expanded their applications beyond aesthetic imagery toward practical visual content creation. However, existing benchmarks mainly focus on natural image synthesis and fail to systematically evaluate models under the structured and multi-constraint requirements of real-world commercial design tasks. In this work, we introduce BizGenEval, a systematic benchmark for commercial visual content generation. The benchmark spans five representative document types: slides, charts, webpages, posters, and scientific figures, and evaluates four key capability dimensions: text rendering, layout control, attribute binding, and knowledge-based reasoning, forming 20 diverse evaluation tasks. BizGenEval contains 400 carefully curated prompts and 8000 human-verified checklist questions to rigorously assess whether generated images satisfy complex visual and semantic constraints. We conduct large-scale benchmarking on 26 popular image generation systems, including state-of-the-art commercial APIs and leading open-source models. The results reveal substantial capability gaps between current generative models and the requirements of professional visual content creation. We hope BizGenEval serves as a standardized benchmark for real-world commercial visual content generation.

  </details>



- **PixelSmile: Toward Fine-Grained Facial Expression Editing**  
  Jiabin Hua, Hengyuan Xu, Aojie Li, Wei Cheng, Gang Yu, Xingjun Ma, Yu-Gang Jiang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25728v1  
  <details><summary>Abstract</summary>

  Fine-grained facial expression editing has long been limited by intrinsic semantic overlap. To address this, we construct the Flex Facial Expression (FFE) dataset with continuous affective annotations and establish FFE-Bench to evaluate structural confusion, editing accuracy, linear controllability, and the trade-off between expression editing and identity preservation. We propose PixelSmile, a diffusion framework that disentangles expression semantics via fully symmetric joint training. PixelSmile combines intensity supervision with contrastive learning to produce stronger and more distinguishable expressions, achieving precise and stable linear expression control through textual latent interpolation. Extensive experiments demonstrate that PixelSmile achieves superior disentanglement and robust identity preservation, confirming its effectiveness for continuous, controllable, and fine-grained expression editing, while naturally supporting smooth expression blending.

  </details>



- **AnyHand: A Large-Scale Synthetic Dataset for RGB(-D) Hand Pose Estimation**  
  Chen Si, Yulin Liu, Bo Ai, Jianwen Xie, Rolandos Alexandros Potamias, Chuanxia Zheng, Hao Su  
  _2026-03-26_ · https://arxiv.org/abs/2603.25726v1  
  <details><summary>Abstract</summary>

  We present AnyHand, a large-scale synthetic dataset designed to advance the state of the art in 3D hand pose estimation from both RGB-only and RGB-D inputs. While recent works with foundation approaches have shown that an increase in the quantity and diversity of training data can markedly improve performance and robustness in hand pose estimation, existing real-world-collected datasets on this task are limited in coverage, and prior synthetic datasets rarely provide occlusions, arm details, and aligned depth together at scale. To address this bottleneck, our AnyHand contains 2.5M single-hand and 4.1M hand-object interaction RGB-D images, with rich geometric annotations. In the RGB-only setting, we show that extending the original training sets of existing baselines with AnyHand yields significant gains on multiple benchmarks (FreiHAND and HO-3D), even when keeping the architecture and training scheme fixed. More impressively, the model trained with AnyHand shows stronger generalization to the out-of-domain HO-Cap dataset, without any fine-tuning. We also contribute a lightweight depth fusion module that can be easily integrated into existing RGB-based models. Trained with AnyHand, the resulting RGB-D model achieves superior performance on the HO-3D benchmark, showing the benefits of depth integration and the effectiveness of our synthetic data.

  </details>



- **SoftMimicGen: A Data Generation System for Scalable Robot Learning in Deformable Object Manipulation**  
  Masoud Moghani, Mahdi Azizian, Animesh Garg, Yuke Zhu, Sean Huver, Ajay Mandlekar  
  _2026-03-26_ · https://arxiv.org/abs/2603.25725v1  
  <details><summary>Abstract</summary>

  Large-scale robot datasets have facilitated the learning of a wide range of robot manipulation skills, but these datasets remain difficult to collect and scale further, owing to the intractable amount of human time, effort, and cost required. Simulation and synthetic data generation have proven to be an effective alternative to fuel this need for data, especially with the advent of recent work showing that such synthetic datasets can dramatically reduce real-world data requirements and facilitate generalization to novel scenarios unseen in real-world demonstrations. However, this paradigm has been limited to rigid-body tasks, which are easy to simulate. Deformable object manipulation encompasses a large portion of real-world manipulation and remains a crucial gap to address towards increasing adoption of the synthetic simulation data paradigm. In this paper, we introduce SoftMimicGen, an automated data generation pipeline for deformable object manipulation tasks. We introduce a suite of high-fidelity simulation environments that encompasses a wide range of deformable objects (stuffed animal, rope, tissue, towel) and manipulation behaviors (high-precision threading, dynamic whipping, folding, pick-and-place), across four robot embodiments: a single-arm manipulator, bimanual arms, a humanoid, and a surgical robot. We apply SoftMimicGen to generate datasets across the task suite, train high-performing policies from the data, and systematically analyze the data generation system. Project website: \href{https://softmimicgen.github.io}{softmimicgen.github.io}.

  </details>



- **No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models**  
  Hai X. Pham, David T. Hoffmann, Ricardo Guerrero, Brais Martinez  
  _2026-03-26_ · https://arxiv.org/abs/2603.25722v1  
  <details><summary>Abstract</summary>

  Contrastive vision-language (V&L) models remain a popular choice for various applications. However, several limitations have emerged, most notably the limited ability of V&L models to learn compositional representations. Prior methods often addressed this limitation by generating custom training data to obtain hard negative samples. Hard negatives have been shown to improve performance on compositionality tasks, but are often specific to a single benchmark, do not generalize, and can cause substantial degradation of basic V&L capabilities such as zero-shot or retrieval performance, rendering them impractical. In this work we follow a different approach. We identify two root causes that limit compositionality performance of V&Ls: 1) Long training captions do not require a compositional representation; and 2) The final global pooling in the text and image encoders lead to a complete loss of the necessary information to learn binding in the first place. As a remedy, we propose two simple solutions: 1) We obtain short concept centric caption parts using standard NLP software and align those with the image; and 2) We introduce a parameter-free cross-modal attention-pooling to obtain concept centric visual embeddings from the image encoder. With these two changes and simple auxiliary contrastive losses, we obtain SOTA performance on standard compositionality benchmarks, while maintaining or improving strong zero-shot and retrieval capabilities. This is achieved without increasing inference cost. We release the code for this work at https://github.com/SamsungLabs/concept_centric_clip.

  </details>



- **Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models**  
  Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, Xiang Bai  
  _2026-03-26_ · https://arxiv.org/abs/2603.25716v1  
  <details><summary>Abstract</summary>

  Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily treat environments as static canvases. When dynamic subjects hide out of sight and later re-emerge, current methods often struggle, leading to frozen, distorted, or vanishing subjects. To address this, we introduce Hybrid Memory, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals. To facilitate research in this direction, we construct HM-World, the first large-scale video dataset dedicated to hybrid memory. It features 59K high-fidelity clips with decoupled camera and subject trajectories, encompassing 17 diverse scenes, 49 distinct subjects, and meticulously designed exit-entry events to rigorously evaluate hybrid coherence. Furthermore, we propose HyDRA, a specialized memory architecture that compresses memory into tokens and utilizes a spatiotemporal relevance-driven retrieval mechanism. By selectively attending to relevant motion cues, HyDRA effectively preserves the identity and motion of hidden subjects. Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality.

  </details>



- **Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training**  
  Jinbo Xing, Zeyinzi Jiang, Yuxiang Tuo, Chaojie Mao, Xiaotang Gai, Xi Chen, Jingfeng Zhang, Yulin Pan, Zhen Han, Jie Xiao, et al.  
  _2026-03-26_ · https://arxiv.org/abs/2603.25706v1  
  <details><summary>Abstract</summary>

  Recent unified models have made unprecedented progress in both understanding and generation. However, while most of them accept multi-modal inputs, they typically produce only single-modality outputs. This challenge of producing interleaved content is mainly due to training data scarcity and the difficulty of modeling long-range cross-modal context. To address this issue, we decompose interleaved generation into textual planning and visual consistency modeling, and introduce a framework consisting of a planner and a visualizer. The planner produces dense textual descriptions for visual content, while the visualizer synthesizes images accordingly. Under this guidance, we construct large-scale textual-proxy interleaved data (where visual content is represented in text) to train the planner, and curate reference-guided image data to train the visualizer. These designs give rise to Wan-Weaver, which exhibits emergent interleaved generation ability with long-range textual coherence and visual consistency. Meanwhile, the integration of diverse understanding and generation data into planner training enables Wan-Weaver to achieve robust task reasoning and generation proficiency. To assess the model's capability in interleaved generation, we further construct a benchmark that spans a wide range of use cases across multiple dimensions. Extensive experiments demonstrate that, even without access to any real interleaved data, Wan-Weaver achieves superior performance over existing methods.

  </details>



- **LEMMA: Laplacian pyramids for Efficient Marine SeMAntic Segmentation**  
  Ishaan Gakhar, Laven Srivastava, Sankarshanaa Sagaram, Aditya Kasliwal, Ujjwal Verma  
  _2026-03-26_ · https://arxiv.org/abs/2603.25689v1  
  <details><summary>Abstract</summary>

  Semantic segmentation in marine environments is crucial for the autonomous navigation of unmanned surface vessels (USVs) and coastal Earth Observation events such as oil spills. However, existing methods, often relying on deep CNNs and transformer-based architectures, face challenges in deployment due to their high computational costs and resource-intensive nature. These limitations hinder the practicality of real-time, low-cost applications in real-world marine settings. To address this, we propose LEMMA, a lightweight semantic segmentation model designed specifically for accurate remote sensing segmentation under resource constraints. The proposed architecture leverages Laplacian Pyramids to enhance edge recognition, a critical component for effective feature extraction in complex marine environments for disaster response, environmental surveillance, and coastal monitoring. By integrating edge information early in the feature extraction process, LEMMA eliminates the need for computationally expensive feature map computations in deeper network layers, drastically reducing model size, complexity and inference time. LEMMA demonstrates state-of-the-art performance across datasets captured from diverse platforms while reducing trainable parameters and computational requirements by up to 71x, GFLOPs by up to 88.5\%, and inference time by up to 84.65\%, as compared to existing models. Experimental results highlight its effectiveness and real-world applicability, including 93.42\% IoU on the Oil Spill dataset and 98.97\% mIoU on Mastr1325.

  </details>



- **Just Zoom In: Cross-View Geo-Localization via Autoregressive Zooming**  
  Yunus Talha Erzurumlu, Jiyong Kwag, Alper Yilmaz  
  _2026-03-26_ · https://arxiv.org/abs/2603.25686v1  
  <details><summary>Abstract</summary>

  Cross-view geo-localization (CVGL) estimates a camera's location by matching a street-view image to geo-referenced overhead imagery, enabling GPS-denied localization and navigation. Existing methods almost universally formulate CVGL as an image-retrieval problem in a contrastively trained embedding space. This ties performance to large batches and hard negative mining, and it ignores both the geometric structure of maps and the coverage mismatch between street-view and overhead imagery. In particular, salient landmarks visible from the street view can fall outside a fixed satellite crop, making retrieval targets ambiguous and limiting explicit spatial inference over the map. We propose Just Zoom In, an alternative formulation that performs CVGL via autoregressive zooming over a city-scale overhead map. Starting from a coarse satellite view, the model takes a short sequence of zoom-in decisions to select a terminal satellite cell at a target resolution, without contrastive losses or hard negative mining. We further introduce a realistic benchmark with crowd-sourced street views and high-resolution satellite imagery that reflects real capture conditions. On this benchmark, Just Zoom In achieves state-of-the-art performance, improving Recall@1 within 50 m by 5.5% and Recall@1 within 100 m by 9.6% over the strongest contrastive-retrieval baseline. These results demonstrate the effectiveness of sequential coarse-to-fine spatial reasoning for cross-view geo-localization.

  </details>



- **Persistent Robot World Models: Stabilizing Multi-Step Rollouts via Reinforcement Learning**  
  Jai Bardhan, Patrik Drozdik, Josef Sivic, Vladimir Petrik  
  _2026-03-26_ · https://arxiv.org/abs/2603.25685v1  
  <details><summary>Abstract</summary>

  Action-conditioned robot world models generate future video frames of the manipulated scene given a robot action sequence, offering a promising alternative for simulating tasks that are difficult to model with traditional physics engines. However, these models are optimized for short-term prediction and break down when deployed autoregressively: each predicted clip feeds back as context for the next, causing errors to compound and visual quality to rapidly degrade. We address this through the following contributions. First, we introduce a reinforcement learning (RL) post-training scheme that trains the world model on its own autoregressive rollouts rather than on ground-truth histories. We achieve this by adapting a recent contrastive RL objective for diffusion models to our setting and show that its convergence guarantees carry over exactly. Second, we design a training protocol that generates and compares multiple candidate variable-length futures from the same rollout state, reinforcing higher-fidelity predictions over lower-fidelity ones. Third, we develop efficient, multi-view visual fidelity rewards that combine complementary perceptual metrics across camera views and are aggregated at the clip level for dense, low-variance training signal. Fourth, we show that our approach establishes a new state-of-the-art for rollout fidelity on the DROID dataset, outperforming the strongest baseline on all metrics (e.g., LPIPS reduced by 14% on external cameras, SSIM improved by 9.1% on the wrist camera), winning 98% of paired comparisons, and achieving an 80% preference rate in a blind human study.

  </details>



- **Can Users Specify Driving Speed? Bench2Drive-Speed: Benchmark and Baselines for Desired-Speed Conditioned Autonomous Driving**  
  Yuqian Shao, Xiaosong Jia, Langechuan Liu, Junchi Yan  
  _2026-03-26_ · https://arxiv.org/abs/2603.25672v1  
  <details><summary>Abstract</summary>

  End-to-end autonomous driving (E2E-AD) has achieved remarkable progress. However, one practical and useful function has been long overlooked: users may wish to customize the desired speed of the policy or specify whether to allow the autonomous vehicle to overtake. To bridge this gap, we present Bench2Drive-Speed, a benchmark with metrics, dataset, and baselines for desired-speed conditioned autonomous driving. We introduce explicit inputs of users' desired target-speed and overtake/follow instructions to driving policy models. We design quantitative metrics, including Speed-Adherence Score and Overtake Score, to measure how faithfully policies follow user specifications, while remaining compatible with standard autonomous driving metrics. To enable training of speed-conditioned policies, one approach is to collect expert demonstrations that strictly follow speed requirements, an expensive and unscalable process in the real world. An alternative is to adapt existing regular driving data by treating the speed observed in future frames as the target speed for training. To investigate this, we construct CustomizedSpeedDataset, composed of 2,100 clips annotated with experts demonstrations, enabling systematic investigation of supervision strategies. Our experiments show that, under proper re-annotation, models trained on regular driving data perform comparably to on expert demonstrations, suggesting that speed supervision can be introduced without additional complex real-world data collection. Furthermore, we find that while target-speed following can be achieved without degrading regular driving performance, executing overtaking commands remains challenging due to the inherent difficulty of interactive behaviors. All code, datasets and baselines are available at https://github.com/Thinklab-SJTU/Bench2Drive-Speed

  </details>



- **Colon-Bench: An Agentic Workflow for Scalable Dense Lesion Annotation in Full-Procedure Colonoscopy Videos**  
  Abdullah Hamdi, Changchun Yang, Xin Gao  
  _2026-03-26_ · https://arxiv.org/abs/2603.25645v1  
  <details><summary>Abstract</summary>

  Early screening via colonoscopy is critical for colon cancer prevention, yet developing robust AI systems for this domain is hindered by the lack of densely annotated, long-sequence video datasets. Existing datasets predominantly focus on single-class polyp detection and lack the rich spatial, temporal, and linguistic annotations required to evaluate modern Multimodal Large Language Models (MLLMs). To address this critical gap, we introduce Colon-Bench, generated via a novel multi-stage agentic workflow. Our pipeline seamlessly integrates temporal proposals, bounding-box tracking, AI-driven visual confirmation, and human-in-the-loop review to scalably annotate full-procedure videos. The resulting verified benchmark is unprecedented in scope, encompassing 528 videos, 14 distinct lesion categories (including polyps, ulcers, and bleeding), over 300,000 bounding boxes, 213,000 segmentation masks, and 133,000 words of clinical descriptions. We utilize Colon-Bench to rigorously evaluate state-of-the-art MLLMs across lesion classification, Open-Vocabulary Video Object Segmentation (OV-VOS), and video Visual Question Answering (VQA). The MLLM results demonstrate surprisingly high localization performance in medical domains compared to SAM-3. Finally, we analyze common VQA errors from MLLMs to introduce a novel "colon-skill" prompting strategy, improving zero-shot MLLM performance by up to 9.7% across most MLLMs. The dataset and the code are available at https://abdullahamdi.com/colon-bench .

  </details>



- **Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification**  
  Ünsal Öztürk, Hatef Otroshi Shahreza, Sébastien Marcel  
  _2026-03-26_ · https://arxiv.org/abs/2603.25613v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently been explored as face verification systems that determine whether two face images are of the same person. Unlike dedicated face recognition systems, MLLMs approach this task through visual prompting and rely on general visual and reasoning abilities. However, the demographic fairness of these models remains largely unexplored. In this paper, we present a benchmarking study that evaluates nine open-source MLLMs from six model families, ranging from 2B to 8B parameters, on the IJB-C and RFW face verification protocols across four ethnicity groups and two gender groups. We measure verification accuracy with the Equal Error Rate and True Match Rate at multiple operating points per demographic group, and we quantify demographic disparity with four FMR-based fairness metrics. Our results show that FaceLLM-8B, the only face-specialised model in our study, substantially outperforms general-purpose MLLMs on both benchmarks. The bias patterns we observe differ from those commonly reported for traditional face recognition, with different groups being most affected depending on the benchmark and the model. We also note that the most accurate models are not necessarily the fairest and that models with poor overall accuracy can appear fair simply because they produce uniformly high error rates across all demographic groups.

  </details>



- **DeepFAN, a transformer-based deep learning model for human-artificial intelligence collaborative assessment of incidental pulmonary nodules in CT scans: a multi-reader, multi-case trial**  
  Zhenchen Zhu, Ge Hu, Weixiong Tan, Kai Gao, Chao Sun, Zhen Zhou, Kepei Xu, Wei Han, Meixia Shang, Xiaoming Qiu, et al.  
  _2026-03-26_ · https://arxiv.org/abs/2603.25607v1  
  <details><summary>Abstract</summary>

  The widespread adoption of CT has notably increased the number of detected lung nodules. However, current deep learning methods for classifying benign and malignant nodules often fail to comprehensively integrate global and local features, and most of them have not been validated through clinical trials. To address this, we developed DeepFAN, a transformer-based model trained on over 10K pathology-confirmed nodules and further conducted a multi-reader, multi-case clinical trial to evaluate its efficacy in assisting junior radiologists. DeepFAN achieved diagnostic area under the curve (AUC) of 0.939 (95% CI 0.930-0.948) on an internal test set and 0.954 (95% CI 0.934-0.973) on the clinical trial dataset involving 400 cases across three independent medical institutions. Explainability analysis indicated higher contributions from global than local features. Twelve readers' average performance significantly improved by 10.9% (95% CI 8.3%-13.5%) in AUC, 10.0% (95% CI 8.9%-11.1%) in accuracy, 7.6% (95% CI 6.1%-9.2%) in sensitivity, and 12.6% (95% CI 10.9%-14.3%) in specificity (P<0.001 for all). Nodule-level inter-reader diagnostic consistency improved from fair to moderate (overall k: 0.313 vs. 0.421; P=0.019). In conclusion, DeepFAN effectively assisted junior radiologists and may help homogenize diagnostic quality and reduce unnecessary follow-up of indeterminate pulmonary nodules. Chinese Clinical Trial Registry: ChiCTR2400084624.

  </details>



- **BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning**  
  Ning Ding, Keisuke Fujii, Toru Tamaki  
  _2026-03-26_ · https://arxiv.org/abs/2603.25533v1  
  <details><summary>Abstract</summary>

  Understanding tactical dynamics in badminton requires analyzing entire matches rather than isolated clips. However, existing badminton datasets mainly focus on short clips or task-specific annotations and rarely provide full-match data with dense multimodal annotations. This limitation makes it difficult to generate accurate shot captions and perform match-level analysis. To address this limitation, we introduce the first Badminton Full Match Dense (BFMD) dataset, with 19 broadcast matches (including both singles and doubles) covering over 20 hours of play, comprising 1,687 rallies and 16,751 hit events, each annotated with a shot caption. The dataset provides hierarchical annotations including match segments, rally events, and dense rally-level multimodal annotations such as shot types, shuttle trajectories, player pose keypoints, and shot captions. We develop a VideoMAE-based multimodal captioning framework with a Semantic Feedback mechanism that leverages shot semantics to guide caption generation and improve semantic consistency. Experimental results demonstrate that multimodal modeling and semantic feedback improve shot caption quality over RGB-only baselines. We further showcase the potential of BFMD by analyzing the temporal evolution of tactical patterns across full matches.

  </details>



- **CHIRP dataset: towards long-term, individual-level, behavioral monitoring of bird populations in the wild**  
  Alex Hoi Hang Chan, Neha Singhal, Onur Kocahan, Andrea Meltzer, Saverio Lubrano, Miyako H. Warrington, Michel Griesser, Fumihiro Kano, Hemal Naik  
  _2026-03-26_ · https://arxiv.org/abs/2603.25524v1  
  <details><summary>Abstract</summary>

  Long-term behavioral monitoring of individual animals is crucial for studying behavioral changes that occur over different time scales, especially for conservation and evolutionary biology. Computer vision methods have proven to benefit biodiversity monitoring, but automated behavior monitoring in wild populations remains challenging. This stems from the lack of datasets that cover a range of computer vision tasks necessary to extract biologically meaningful measurements of individual animals. Here, we introduce such a dataset (CHIRP) with a new method (CORVID) for individual re-identification of wild birds. The CHIRP (Combining beHaviour, Individual Re-identification and Postures) dataset is curated from a long-term population of wild Siberian jays studied in Swedish Lapland, supporting re-identification (re-id), action recognition, 2D keypoint estimation, object detection, and instance segmentation. In addition to traditional task-specific benchmarking, we introduce application-specific benchmarking with biologically relevant metrics (feeding rates, co-occurrence rates) to evaluate the performance of models in real-world use cases. Finally, we present CORVID (COlouR-based Video re-ID), a novel pipeline for individual identification of birds based on the segmentation and classification of colored leg rings, a widespread approach for visual identification of individual birds. CORVID offers a probability-based id tracking method by matching the detected combination of color rings with a database. We use application-specific benchmarking to show that CORVID outperforms state-of-the-art re-id methods. We hope this work offers the community a blueprint for curating real-world datasets from ethically approved biological studies to bridge the gap between computer vision research and biological applications.

  </details>



- **Challenges in Hyperspectral Imaging for Autonomous Driving: The HSI-Drive Case**  
  Koldo Basterretxea, Jon Gutiérrez-Zaballa, Javier Echanobe  
  _2026-03-26_ · https://arxiv.org/abs/2603.25510v1  
  <details><summary>Abstract</summary>

  The use of hyperspectral imaging (HSI) in autonomous driving (AD), while promising, faces many challenges related to the specifics and requirements of this application domain. On the one hand, non-controlled and variable lighting conditions, the wide depth-of-field ranges, and dynamic scenes with fast-moving objects. On the other hand, the requirements for real-time operation and the limited computational resources of embedded platforms. The combination of these factors determines both the criteria for selecting appropriate HSI technologies and the development of custom vision algorithms that leverage the spectral and spatial information obtained from the sensors. In this article, we analyse several techniques explored in the research of HSI-based vision systems with application to AD, using as an example results obtained from experiments using data from the most recent version of the HSI-Drive dataset.

  </details>



- **RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models**  
  Yufeng Yang, Xianfang Zeng, Zhangqi Jiang, Fukun Yin, Jianzhuang Liu, Wei Cheng, jinghong lan, Shiyu Liu, Yuqi Peng, Gang YU, et al.  
  _2026-03-26_ · https://arxiv.org/abs/2603.25502v1  
  <details><summary>Abstract</summary>

  Image restoration under real-world degradations is critical for downstream tasks such as autonomous driving and object detection. However, existing restoration models are often limited by the scale and distribution of their training data, resulting in poor generalization to real-world scenarios. Recently, large-scale image editing models have shown strong generalization ability in restoration tasks, especially for closed-source models like Nano Banana Pro, which can restore images while preserving consistency. Nevertheless, achieving such performance with those large universal models requires substantial data and computational costs. To address this issue, we construct a large-scale dataset covering nine common real-world degradation types and train a state-of-the-art open-source model to narrow the gap with closed-source alternatives. Furthermore, we introduce RealIR-Bench, which contains 464 real-world degraded images and tailored evaluation metrics focusing on degradation removal and consistency preservation. Extensive experiments demonstrate our model ranks first among open-source methods, achieving state-of-the-art performance.

  </details>



- **Temporally Decoupled Diffusion Planning for Autonomous Driving**  
  Xiang Li, Bikun Wang, John Zhang, Jianjun Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25462v1  
  <details><summary>Abstract</summary>

  Motion planning in dynamic urban environments requires balancing immediate safety with long-term goals. While diffusion models effectively capture multi-modal decision-making, existing approaches treat trajectories as monolithic entities, overlooking heterogeneous temporal dependencies where near-term plans are constrained by instantaneous dynamics and far-term plans by navigational goals. To address this, we propose Temporally Decoupled Diffusion Model (TDDM), which reformulates trajectory generation via a noise-as-mask paradigm. By partitioning trajectories into segments with independent noise levels, we implicitly treat high noise as information voids and weak noise as contextual cues. This compels the model to reconstruct corrupted near-term states by leveraging internal correlations with better-preserved temporal contexts. Architecturally, we introduce a Temporally Decoupled Adaptive Layer Normalization (TD-AdaLN) to inject segment-specific timesteps. During inference, our Asymmetric Temporal Classifier-Free Guidance utilizes weakly noised far-term priors to guide immediate path generation. Evaluations on the nuPlan benchmark show TDDM approaches or exceeds state-of-the-art baselines, particularly excelling in the challenging Test14-hard subset.

  </details>



- **DC-Reg: Globally Optimal Point Cloud Registration via Tight Bounding with Difference of Convex Programming**  
  Wei Lian, Fei Ma, Hang Pan, Zhesen Cui, Wangmeng Zuo  
  _2026-03-26_ · https://arxiv.org/abs/2603.25442v1  
  <details><summary>Abstract</summary>

  Achieving globally optimal point cloud registration under partial overlaps and large misalignments remains a fundamental challenge. While simultaneous transformation ($\boldsymbolθ$) and correspondence ($\mathbf{P}$) estimation has the advantage of being robust to nonrigid deformation, its non-convex coupled objective often leads to local minima for heuristic methods and prohibitive convergence times for existing global solvers due to loose lower bounds. To address this, we propose DC-Reg, a robust globally optimal framework that significantly tightens the Branch-and-Bound (BnB) search. Our core innovation is the derivation of a holistic concave underestimator for the coupled transformation-assignment objective, grounded in the Difference of Convex (DC) programming paradigm. Unlike prior works that rely on term-wise relaxations (e.g., McCormick envelopes) which neglect variable interplay, our holistic DC decomposition captures the joint structural interaction between $\boldsymbolθ$ and $\mathbf{P}$. This formulation enables the computation of remarkably tight lower bounds via efficient Linear Assignment Problems (LAP) evaluated at the vertices of the search boxes. We validate our framework on 2D similarity and 3D rigid registration, utilizing rotation-invariant features for the latter to achieve high efficiency without sacrificing optimality. Experimental results on synthetic data and the 3DMatch benchmark demonstrate that DC-Reg achieves significantly faster convergence and superior robustness to extreme noise and outliers compared to state-of-the-art global techniques.

  </details>



- **Multimodal Dataset Distillation via Phased Teacher Models**  
  Shengbin Guo, Hang Zhao, Senqiao Yang, Chenyang Jiang, Yuhang Cheng, Xiangru Peng, Rui Shao, Zhuotao Tian  
  _2026-03-26_ · https://arxiv.org/abs/2603.25388v1  
  <details><summary>Abstract</summary>

  Multimodal dataset distillation aims to construct compact synthetic datasets that enable efficient compression and knowledge transfer from large-scale image-text data. However, existing approaches often fail to capture the complex, dynamically evolving knowledge embedded in the later training stages of teacher models. This limitation leads to degraded student performance and compromises the quality of the distilled data. To address critical challenges such as pronounced cross-stage performance gaps and unstable teacher trajectories, we propose Phased Teacher Model with Shortcut Trajectory (PTM-ST) -- a novel phased distillation framework. PTM-ST leverages stage-aware teacher modeling and a shortcut-based trajectory construction strategy to accurately fit the teacher's learning dynamics across distinct training phases. This enhances both the stability and expressiveness of the distillation process. Through theoretical analysis and comprehensive experiments, we show that PTM-ST significantly mitigates optimization oscillations and inter-phase knowledge gaps, while also reducing storage overhead. Our method consistently surpasses state-of-the-art baselines on Flickr30k and COCO, achieving up to 13.5% absolute improvement and an average gain of 9.53% on Flickr30k. Code: https://github.com/Previsior/PTM-ST.

  </details>



- **Image Rotation Angle Estimation: Comparing Circular-Aware Methods**  
  Maximilian Woehrer  
  _2026-03-26_ · https://arxiv.org/abs/2603.25351v1  
  <details><summary>Abstract</summary>

  Automatic image rotation estimation is a key preprocessing step in many vision pipelines. This task is challenging because angles have circular topology, creating boundary discontinuities that hinder standard regression methods. We present a comprehensive study of five circular-aware methods for global orientation estimation: direct angle regression with circular loss, classification via angular binning, unit-vector regression, phase-shifting coder, and circular Gaussian distribution. Using transfer learning from ImageNet-pretrained models, we systematically evaluate these methods across sixteen modern architectures by adapting their output heads for rotation-specific predictions. Our results show that probabilistic methods, particularly the circular Gaussian distribution, are the most robust across architectures, while classification achieves the best accuracy on well-matched backbones but suffers training instabilities on others. The best configuration (classification with EfficientViT-B3) achieves a mean absolute error (MAE) of 1.23° (mean across five independent runs) on the DRC-D dataset, while the circular Gaussian distribution with MambaOut Base achieves a virtually identical 1.24° with greater robustness across backbones. Training and evaluating our top-performing method-architecture combinations on COCO 2014, the best configuration reaches 3.71° MAE, improving substantially over prior work, with further improvement to 2.84° on the larger COCO 2017 dataset.

  </details>



- **MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data**  
  Zhekai Chen, Yuqing Wang, Manyuan Zhang, Xihui Liu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25319v1  
  <details><summary>Abstract</summary>

  Generating images conditioned on multiple visual references is critical for real-world applications such as multi-subject composition, narrative illustration, and novel view synthesis, yet current models suffer from severe performance degradation as the number of input references grows. We identify the root cause as a fundamental data bottleneck: existing datasets are dominated by single- or few-reference pairs and lack the structured, long-context supervision needed to learn dense inter-reference dependencies. To address this, we introduce MacroData, a large-scale dataset of 400K samples, each containing up to 10 reference images, systematically organized across four complementary dimensions -- Customization, Illustration, Spatial reasoning, and Temporal dynamics -- to provide comprehensive coverage of the multi-reference generation space. Recognizing the concurrent absence of standardized evaluation protocols, we further propose MacroBench, a benchmark of 4,000 samples that assesses generative coherence across graded task dimensions and input scales. Extensive experiments show that fine-tuning on MacroData yields substantial improvements in multi-reference generation, and ablation studies further reveal synergistic benefits of cross-task co-training and effective strategies for handling long-context complexity. The dataset and benchmark will be publicly released.

  </details>



- **Towards Controllable Low-Light Image Enhancement: A Continuous Multi-illumination Dataset and Efficient State Space Framework**  
  Hongru Han, Tingrui Guo, Liming Zhang, Yan Su, Qiwen Xu, Zhuohua Ye  
  _2026-03-26_ · https://arxiv.org/abs/2603.25296v1  
  <details><summary>Abstract</summary>

  Low-light image enhancement (LLIE) has traditionally been formulated as a deterministic mapping. However, this paradigm often struggles to account for the ill-posed nature of the task, where unknown ambient conditions and sensor parameters create a multimodal solution space. Consequently, state-of-the-art methods frequently encounter luminance discrepancies between predictions and labels, often necessitating "gt-mean" post-processing to align output luminance for evaluation. To address this fundamental limitation, we propose a transition toward Controllable Low-light Enhancement (CLE), explicitly reformulating the task as a well-posed conditional problem. To this end, we introduce CLE-RWKV, a holistic framework supported by Light100, a new benchmark featuring continuous real-world illumination transitions. To resolve the conflict between luminance control and chromatic fidelity, a noise-decoupled supervision strategy in the HVI color space is employed, effectively separating illumination modulation from texture restoration. Architecturally, to adapt efficient State Space Models (SSMs) for dense prediction, we leverage a Space-to-Depth (S2D) strategy. By folding spatial neighborhoods into channel dimensions, this design allows the model to recover local inductive biases and effectively bridge the "scanning gap" inherent in flattened visual sequences without sacrificing linear complexity. Experiments across seven benchmarks demonstrate that our approach achieves competitive performance and robust controllability, providing a real-world multi-illumination alternative that significantly reduces the reliance on gt-mean post-processing.

  </details>



- **V2U4Real: A Real-world Large-scale Dataset for Vehicle-to-UAV Cooperative Perception**  
  Weijia Li, Haoen Xiang, Tianxu Wang, Shuaibing Wu, Qiming Xia, Cheng Wang, Chenglu Wen  
  _2026-03-26_ · https://arxiv.org/abs/2603.25275v1  
  <details><summary>Abstract</summary>

  Modern autonomous vehicle perception systems are often constrained by occlusions, blind spots, and limited sensing range. While existing cooperative perception paradigms, such as Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I), have demonstrated their effectiveness in mitigating these challenges, they remain limited to ground-level collaboration and cannot fully address large-scale occlusions or long-range perception in complex environments. To advance research in cross-view cooperative perception, we present V2U4Real, the first large-scale real-world multi-modal dataset for Vehicle-to-UAV (V2U) cooperative object perception. V2U4Real is collected by a ground vehicle and a UAV equipped with multi-view LiDARs and RGB cameras. The dataset covers urban streets, university campuses, and rural roads under diverse traffic scenarios, comprising over 56K LiDAR frames, 56K multi-view camera images, and 700K annotated 3D bounding boxes across four classes. To support a wide range of research tasks, we establish benchmarks for single-agent 3D object detection, cooperative 3D object detection, and object tracking. Comprehensive evaluations of several state-of-the-art models demonstrate the effectiveness of V2U cooperation in enhancing perception robustness and long-range awareness. The V2U4Real dataset and codebase is available at https://github.com/VjiaLi/V2U4Real.

  </details>



- **A Minimum-Energy Control Approach for Redundant Mobile Manipulators in Physical Human-Robot Interaction Applications**  
  Davide Tebaldi, Niccolò Paradisi, Fabio Pini, Luigi Biagiotti  
  _2026-03-26_ · https://arxiv.org/abs/2603.25259v1  
  <details><summary>Abstract</summary>

  Research on mobile manipulation systems that physically interact with humans has expanded rapidly in recent years, opening the way to tasks which could not be performed using fixed-base manipulators. Within this context, developing suitable control methodologies is essential since mobile manipulators introduce additional degrees of freedom, making the design of control approaches more challenging and more prone to performance optimization. This paper proposes a control approach for a mobile manipulator, composed of a mobile base equipped with a robotic arm mounted on the top, with the objective of minimizing the overall kinetic energy stored in the whole-body mobile manipulator in physical human-robot interaction applications. The approach is experimentally tested with reference to a peg-in-hole task, and the results demonstrate that the proposed approach reduces the overall kinetic energy stored in the whole-body robotic system and improves the system performance compared with the benchmark method.

  </details>



- **Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models**  
  Yabin Zhang, Maya Varma, Yunhe Gao, Jean-Benoit Delbrouck, Jiaming Liu, Chong Wang, Curtis Langlotz  
  _2026-03-26_ · https://arxiv.org/abs/2603.25250v1  
  <details><summary>Abstract</summary>

  Out-of-distribution (OOD) detection aims to identify samples that deviate from in-distribution (ID). One popular pipeline addresses this by introducing negative labels distant from ID classes and detecting OOD based on their distance to these labels. However, such labels may present poor activation on OOD samples, failing to capture the OOD characteristics. To address this, we propose \underline{T}est-time \underline{A}ctivated \underline{N}egative \underline{L}abels (TANL) by dynamically evaluating activation levels across the corpus dataset and mining candidate labels with high activation responses during the testing process. Specifically, TANL identifies high-confidence test images online and accumulates their assignment probabilities over the corpus to construct a label activation metric. Such a metric leverages historical test samples to adaptively align with the test distribution, enabling the selection of distribution-adaptive activated negative labels. By further exploring the activation information within the current testing batch, we introduce a more fine-grained, batch-adaptive variant. To fully utilize label activation knowledge, we propose an activation-aware score function that emphasizes negative labels with stronger activations, boosting performance and enhancing its robustness to the label number. Our TANL is training-free, test-efficient, and grounded in theoretical justification. Experiments on diverse backbones and wide task settings validate its effectiveness. Notably, on the large-scale ImageNet benchmark, TANL significantly reduces the FPR95 from 17.5\% to 9.8\%. Codes are available at \href{https://github.com/YBZh/OpenOOD-VLM}{YBZh/OpenOOD-VLM}.

  </details>



- **An Image Dataset of Common Skin Diseases of Bangladesh and Benchmarking Performance with Machine Learning Models**  
  Sazzad Hossain, Saiful Islam, Muhammad Ibrahim, Md. Rasel Ahmed, Md Shuayb, Ahmedul Kabir  
  _2026-03-26_ · https://arxiv.org/abs/2603.25229v1  
  <details><summary>Abstract</summary>

  Skin diseases are a major public health concern worldwide, and their detection is often challenging without access to dermatological expertise. In countries like Bangladesh, which is highly populated, the number of qualified skin specialists and diagnostic instruments is insufficient to meet the demand. Due to the lack of proper detection and treatment of skin diseases, that may lead to severe health consequences including death. Common properties of skin diseases are, changing the color, texture, and pattern of skin and in this era of artificial intelligence and machine learning, we are able to detect skin diseases by using image processing and computer vision techniques. In response to this challenge, we develop a publicly available dataset focused on common skin disease detection using machine learning techniques. We focus on five prevalent skin diseases in Bangladesh: Contact Dermatitis, Vitiligo, Eczema, Scabies, and Tinea Ringworm. The dataset consists of 1612 images (of which, 250 are distinct while others are augmented), collected directly from patients at the outpatient department of Faridpur Medical College, Faridpur, Bangladesh. The data comprises of 302, 381, 301, 316, and 312 images of Dermatitis, Eczema, Scabies, Tinea Ringworm, and Vitiligo, respectively. Although the data are collected regionally, the selected diseases are common across many countries especially in South Asia, making the dataset potentially valuable for global applications in machine learning-based dermatology. We also apply several machine learning and deep learning models on the dataset and report classification performance. We expect that this research would garner attention from machine learning and deep learning researchers and practitioners working in the field of automated disease diagnosis.

  </details>



- **Training-free Detection and 6D Pose Estimation of Unseen Surgical Instruments**  
  Jonas Hein, Lilian Calvet, Matthias Seibold, Siyu Tang, Marc Pollefeys, Philipp Fürnstahl  
  _2026-03-26_ · https://arxiv.org/abs/2603.25228v1  
  <details><summary>Abstract</summary>

  Purpose: Accurate detection and 6D pose estimation of surgical instruments are crucial for many computer-assisted interventions. However, supervised methods lack flexibility for new or unseen tools and require extensive annotated data. This work introduces a training-free pipeline for accurate multi-view 6D pose estimation of unseen surgical instruments, which only requires a textured CAD model as prior knowledge. Methods: Our pipeline consists of two main stages. First, for detection, we generate object mask proposals in each view and score their similarity to rendered templates using a pre-trained feature extractor. Detections are matched across views, triangulated into 3D instance candidates, and filtered using multi-view geometric consistency. Second, for pose estimation, a set of pose hypotheses is iteratively refined and scored using feature-metric scores with cross-view attention. The best hypothesis undergoes a final refinement using a novel multi-view, occlusion-aware contour registration, which minimizes reprojection errors of unoccluded contour points. Results: The proposed method was rigorously evaluated on real-world surgical data from the MVPSP dataset. The method achieves millimeter-accurate pose estimates that are on par with supervised methods under controlled conditions, while maintaining full generalization to unseen instruments. These results demonstrate the feasibility of training-free, marker-less detection and tracking in surgical scenes, and highlight the unique challenges in surgical environments. Conclusion: We present a novel and flexible pipeline that effectively combines state-of-the-art foundational models, multi-view geometry, and contour-based refinement for high-accuracy 6D pose estimation of surgical instruments without task-specific training. This approach enables robust instrument tracking and scene understanding in dynamic clinical environments.

  </details>



- **SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment**  
  Pengyu Chen, Haotian Sa, Yiwei Hu, Yuhan Cheng, Junbo Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25218v1  
  <details><summary>Abstract</summary>

  Detecting small unmanned aerial vehicles (UAVs) from a ground-to-air (G2A) perspective presents significant challenges, including extremely low pixel occupancy, cluttered aerial backgrounds, and strict real-time constraints. Existing YOLO-based detectors are primarily optimized for general object detection and often lack adequate feature resolution for sub-pixel targets, while introducing complexities during deployment. In this paper, we propose SDD-YOLO, a small-target detection framework tailored for G2A anti-UAV surveillance. To capture fine-grained spatial details critical for micro-targets, SDD-YOLO introduces a P2 high-resolution detection head operating at 4 times downsampling. Furthermore, we integrate the recent architectural advancements from YOLO26, including a DFL-free, NMS-free architecture for streamlined inference, and the MuSGD hybrid training strategy with ProgLoss and STAL, which substantially mitigates gradient oscillation on sparse small-target signals. To support our evaluation, we construct DroneSOD-30K, a large-scale G2A dataset comprising approximately 30,000 annotated images covering diverse meteorological conditions. Experiments demonstrate that SDD-YOLO-n achieves a mAP@0.5 of 86.0% on DroneSOD-30K, surpassing the YOLOv5n baseline by 7.8 percentage points. Extensive inference analysis shows our model attains 226 FPS on an NVIDIA RTX 5090 and 35 FPS on an Intel Xeon CPU, demonstrating exceptional efficiency for future edge deployment.

  </details>



- **CIV-DG: Conditional Instrumental Variables for Domain Generalization in Medical Imaging**  
  Shaojin Bai, Yuting Su, Weizhi Nie  
  _2026-03-26_ · https://arxiv.org/abs/2603.25202v1  
  <details><summary>Abstract</summary>

  Cross-site generalizability in medical AI is fundamentally compromised by selection bias, a structural mechanism where patient demographics (e.g., age, severity) non-randomly dictate hospital assignment. Conventional Domain Generalization (DG) paradigms, which predominantly target image-level distribution shifts, fail to address the resulting spurious correlations between site-specific variations and diagnostic labels. To surmount this identifiability barrier, we propose CIV-DG, a causal framework that leverages Conditional Instrumental Variables to disentangle pathological semantics from scanner-induced artifacts. By relaxing the strict random assignment assumption of standard IV methods, CIV-DG accommodates complex clinical scenarios where hospital selection is endogenously driven by patient demographics. We instantiate this theory via a Deep Generalized Method of Moments (DeepGMM) architecture, employing a conditional critic to minimize moment violations and enforce instrument-error orthogonality within demographic strata. Extensive experiments on the Camelyon17 benchmark and large-scale Chest X-Ray datasets demonstrate that CIV-DG significantly outperforms leading baselines, validating the efficacy of conditional causal mechanisms in resolving structural confounding for robust medical AI.

  </details>



- **TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation**  
  Peng Wen, Yuting Wang, Qiurui Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25199v1  
  <details><summary>Abstract</summary>

  Current football imitation research primarily aims to opti mize reward-based objectives, such as goals scored or win rate proxies, paying less attention to accurately replicat ing real-world team tactical behaviors. We introduce Tac SIm, a large-scale dataset and benchmark for Tactical Style Imitation in football. TacSIm imitates the acitons of all 11 players in one team in the given broadcast footage of Pre mier League matches under a single broadcast view. Under a offensive or defensive broadcast footage, TacSIm projects the beginning positions and actions of all 22 players from both sides onto a standard pitch coordinate system. Tac SIm offers an explicit style imitation task and evaluation protocols. Tactics style imitation is measured by using spatial occupancy similarity and movement vector similarity in defined time, supporting the evaluation of spatial and tem poral similarities for one team. We run multiple baseline methods in a unified virtual environment to generate full team behaviors, enabling both quantitative and visual as sessment of tactical coordination. By using unified data and metrics from broadcast to simulation, TacSIm estab lishes a rigorous benchmark for measuring and modeling style-aligned tactical imitation task in football.

  </details>



- **The Competence Shadow: Theory and Bounds of AI Assistance in Safety Engineering**  
  Umair Siddique  
  _2026-03-26_ · https://arxiv.org/abs/2603.25197v1  
  <details><summary>Abstract</summary>

  As AI assistants become integrated into safety engineering workflows for Physical AI systems, a critical question emerges: does AI assistance improve safety analysis quality, or introduce systematic blind spots that surface only through post-deployment incidents? This paper develops a formal framework for AI assistance in safety analysis. We first establish why safety engineering resists benchmark-driven evaluation: safety competence is irreducibly multidimensional, constrained by context-dependent correctness, inherent incompleteness, and legitimate expert disagreement. We formalize this through a five-dimensional competence framework capturing domain knowledge, standards expertise, operational experience, contextual understanding, and judgment. We introduce the competence shadow: the systematic narrowing of human reasoning induced by AI-generated safety analysis. The shadow is not what the AI presents, but what it prevents from being considered. We formalize four canonical human-AI collaboration structures and derive closed-form performance bounds, demonstrating that the competence shadow compounds multiplicatively to produce degradation far exceeding naive additive estimates. The central finding is that AI assistance in safety engineering is a collaboration design problem, not a software procurement decision. The same tool degrades or improves analysis quality depending entirely on how it is used. We derive non-degradation conditions for shadow-resistant workflows and call for a shift from tool qualification toward workflow qualification for trustworthy Physical AI.

  </details>



- **AnyID: Ultra-Fidelity Universal Identity-Preserving Video Generation from Any Visual References**  
  Jiahao Wang, Hualian Sheng, Sijia Cai, Yuxiao Yang, Weizhan Zhang, Caixia Yan, Bing Deng, Jieping Ye  
  _2026-03-26_ · https://arxiv.org/abs/2603.25188v1  
  <details><summary>Abstract</summary>

  Identity-preserving video generation offers powerful tools for creative expression, allowing users to customize videos featuring their beloved characters. However, prevailing methods are typically designed and optimized for a single identity reference. This underlying assumption restricts creative flexibility by inadequately accommodating diverse real-world input formats. Relying on a single source also constitutes an ill-posed scenario, causing an inherently ambiguous setting that makes it difficult for the model to faithfully reproduce an identity across novel contexts. To address these issues, we present AnyID, an ultra-fidelity identity-preservation video generation framework that features two core contributions. First, we introduce a scalable omni-referenced architecture that effectively unifies heterogeneous identity inputs (e.g., faces, portraits, and videos) into a cohesive representation. Second, we propose a primary-referenced generation paradigm, which designates one reference as a canonical anchor and uses a novel differential prompt to enable precise, attribute-level controllability. We conduct training on a large-scale, meticulously curated dataset to ensure robustness and high fidelity, and then perform a final fine-tuning stage using reinforcement learning. This process leverages a preference dataset constructed from human evaluations, where annotators performed pairwise comparisons of videos based on two key criteria: identity fidelity and prompt controllability. Extensive evaluations validate that AnyID achieves ultra-high identity fidelity as well as superior attribute-level controllability across different task settings.

  </details>



- **Bilingual Text-to-Motion Generation: A New Benchmark and Baselines**  
  Wanjiang Weng, Xiaofeng Tan, Xiangbo Shu, Guo-Sen Xie, Pan Zhou, Hongsong Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25178v1  
  <details><summary>Abstract</summary>

  Text-to-motion generation holds significant potential for cross-linguistic applications, yet it is hindered by the lack of bilingual datasets and the poor cross-lingual semantic understanding of existing language models. To address these gaps, we introduce BiHumanML3D, the first bilingual text-to-motion benchmark, constructed via LLM-assisted annotation and rigorous manual correction. Furthermore, we propose a simple yet effective baseline, Bilingual Motion Diffusion (BiMD), featuring Cross-Lingual Alignment (CLA). CLA explicitly aligns semantic representations across languages, creating a robust conditional space that enables high-quality motion generation from bilingual inputs, including zero-shot code-switching scenarios. Extensive experiments demonstrate that BiMD with CLA achieves an FID of 0.045 vs. 0.169 and R@3 of 82.8\% vs. 80.8\%, significantly outperforms monolingual diffusion models and translation baselines on BiHumanML3D, underscoring the critical necessity and reliability of our dataset and the effectiveness of our alignment strategy for cross-lingual motion synthesis. The dataset and code are released at \href{https://wengwanjiang.github.io/BilingualT2M-page}{https://wengwanjiang.github.io/BilingualT2M-page}

  </details>



- **AG-EgoPose: Leveraging Action-Guided Motion and Kinematic Joint Encoding for Egocentric 3D Pose Estimation**  
  Md Mushfiqur Azam, John Quarles, Kevin Desai  
  _2026-03-26_ · https://arxiv.org/abs/2603.25175v1  
  <details><summary>Abstract</summary>

  Egocentric 3D human pose estimation remains challenging due to severe perspective distortion, limited body visibility, and complex camera motion inherent in first-person viewpoints. Existing methods typically rely on single-frame analysis or limited temporal fusion, which fails to effectively leverage the rich motion context available in egocentric videos. We introduce AG-EgoPose, a novel dual-stream framework that integrates short- and long-range motion context with fine-grained spatial cues for robust pose estimation from fisheye camera input. Our framework features two parallel streams: A spatial stream uses a weight-sharing ResNet-18 encoder-decoder to generate 2D joint heatmaps and corresponding joint-specific spatial feature tokens. Simultaneously, a temporal stream uses a ResNet-50 backbone to extract visual features, which are then processed by an action recognition backbone to capture the motion dynamics. These complementary representations are fused and refined in a transformer decoder with learnable joint tokens, which allows for the joint-level integration of spatial and temporal evidence while maintaining anatomical constraints. Experiments on real-world datasets demonstrate that AG-EgoPose achieves state-of-the-art performance in both quantitative and qualitative metrics. Code is available at: https://github.com/Mushfiq5647/AG-EgoPose.

  </details>



- **SportSkills: Physical Skill Learning from Sports Instructional Videos**  
  Kumar Ashutosh, Chi Hsuan Wu, Kristen Grauman  
  _2026-03-26_ · https://arxiv.org/abs/2603.25163v1  
  <details><summary>Abstract</summary>

  Current large-scale video datasets focus on general human activity, but lack depth of coverage on fine-grained activities needed to address physical skill learning. We introduce SportSkills, the first large-scale sports dataset geared towards physical skill learning with in-the-wild video. SportSkills has more than 360k instructional videos containing more than 630k visual demonstrations paired with instructional narrations explaining the know-how behind the actions from 55 varied sports. Through a suite of experiments, we show that SportSkills unlocks the ability to understand fine-grained differences between physical actions. Our representation achieves gains of up to 4x with the same model trained on traditional activity-centric datasets. Crucially, building on SportSkills, we introduce the first large-scale task formulation of mistake-conditioned instructional video retrieval, bridging representation learning and actionable feedback generation (e.g., "here's my execution of a skill; which video clip should I watch to improve it?"). Formal evaluations by professional coaches show our retrieval approach significantly advances the ability of video models to personalize visual instructions for a user query.

  </details>



- **FD$^2$: A Dedicated Framework for Fine-Grained Dataset Distillation**  
  Hongxu Ma, Guang Li, Shijie Wang, Dongzhan Zhou, Baoli Sun, Takahiro Ogawa, Miki Haseyama, Zhihui Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25144v1  
  <details><summary>Abstract</summary>

  Dataset distillation (DD) compresses a large training set into a small synthetic set, reducing storage and training cost, and has shown strong results on general benchmarks. Decoupled DD further improves efficiency by splitting the pipeline into pretraining, sample distillation, and soft-label generation. However, existing decoupled methods largely rely on coarse class-label supervision and optimize samples within each class in a nearly identical manner. On fine-grained datasets, this often yields distilled samples that (i) retain large intra-class variation with subtle inter-class differences and (ii) become overly similar within the same class, limiting localized discriminative cues and hurting recognition. To solve the above-mentioned problems, we propose FD$^{2}$, a dedicated framework for Fine-grained Dataset Distillation. FD$^{2}$ localizes discriminative regions and constructs fine-grained representations for distillation. During pretraining, counterfactual attention learning aggregates discriminative representations to update class prototypes. During distillation, a fine-grained characteristic constraint aligns each sample with its class prototype while repelling others, and a similarity constraint diversifies attention across same-class samples. Experiments on multiple fine-grained and general datasets show that FD$^{2}$ integrates seamlessly with decoupled DD and improves performance in most settings, indicating strong transferability.

  </details>



- **SAVe: Self-Supervised Audio-visual Deepfake Detection Exploiting Visual Artifacts and Audio-visual Misalignment**  
  Sahibzada Adil Shahzad, Ammarah Hashmi, Junichi Yamagishi, Yusuke Yasuda, Yu Tsao, Chia-Wen Lin, Yan-Tsung Peng, Hsin-Min Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25140v1  
  <details><summary>Abstract</summary>

  Multimodal deepfakes can exhibit subtle visual artifacts and cross-modal inconsistencies, which remain challenging to detect, especially when detectors are trained primarily on curated synthetic forgeries. Such synthetic dependence can introduce dataset and generator bias, limiting scalability and robustness to unseen manipulations. We propose SAVe, a self-supervised audio-visual deepfake detection framework that learns entirely on authentic videos. SAVe generates on-the-fly, identity-preserving, region-aware self-blended pseudo-manipulations to emulate tampering artifacts, enabling the model to learn complementary visual cues across multiple facial granularities. To capture cross-modal evidence, SAVe also models lip-speech synchronization via an audio-visual alignment component that detects temporal misalignment patterns characteristic of audio-visual forgeries. Experiments on FakeAVCeleb and AV-LipSync-TIMIT demonstrate competitive in-domain performance and strong cross-dataset generalization, highlighting self-supervised learning as a scalable paradigm for multimodal deepfake detection.

  </details>



- **EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions**  
  Taegyoon Yoon, Yegyu Han, Seojin Ji, Jaewoo Park, Sojeong Kim, Taein Kwon, Hyung-Sin Kim  
  _2026-03-26_ · https://arxiv.org/abs/2603.25135v1  
  <details><summary>Abstract</summary>

  Smart glass is emerging as an useful device since it provides plenty of insights under hands-busy, eyes-on-task situations. To understand the context of the wearer, 6D object pose estimation in egocentric view is becoming essential. However, existing 6D object pose estimation benchmarks fail to capture the challenges of real-world egocentric applications, which are often dominated by severe motion blur, dynamic illumination, and visual obstructions. This discrepancy creates a significant gap between controlled lab data and chaotic real-world application. To bridge this gap, we introduce EgoXtreme, a new large-scale 6D pose estimation dataset captured entirely from an egocentric perspective. EgoXtreme features three challenging scenarios - industrial maintenance, sports, and emergency rescue - designed to introduce severe perceptual ambiguities through extreme lighting, heavy motion blur, and smoke. Evaluations of state-of-the-art generalizable pose estimators on EgoXtreme indicate that their generalization fails to hold in extreme conditions, especially under low light. We further demonstrate that simply applying image restoration (e.g., deblurring) offers no positive improvement for extreme conditions. While performance gain has appeared in tracking-based approach, implying using temporal information in fast-motion scenarios is meaningful. We conclude that EgoXtreme is an essential resource for developing and evaluating the next generation of pose estimation models robust enough for real-world egocentric vision. The dataset and code are available at https://taegyoun88.github.io/EgoXtreme/

  </details>



- **Denoise and Align: Towards Source-Free UDA for Robust Panoramic Semantic Segmentation**  
  Yaowen Chang, Zhen Cao, Xu Zheng, Xiaoxin Mi, Zhen Dong  
  _2026-03-26_ · https://arxiv.org/abs/2603.25131v1  
  <details><summary>Abstract</summary>

  Panoramic semantic segmentation is pivotal for comprehensive 360° scene understanding in critical applications like autonomous driving and virtual reality. However, progress in this domain is constrained by two key challenges: the severe geometric distortions inherent in panoramic projections and the prohibitive cost of dense annotation. While Unsupervised Domain Adaptation (UDA) from label-rich pinhole-camera datasets offers a viable alternative, many real-world tasks impose a stricter source-free (SFUDA) constraint where source data is inaccessible for privacy or proprietary reasons. This constraint significantly amplifies the core problems of domain shift, leading to unreliable pseudo-labels and dramatic performance degradation, particularly for minority classes. To overcome these limitations, we propose the DAPASS framework. DAPASS introduces two synergistic modules to robustly transfer knowledge without source data. First, our Panoramic Confidence-Guided Denoising (PCGD) module generates high-fidelity, class-balanced pseudo-labels by enforcing perturbation consistency and incorporating neighborhood-level confidence to filter noise. Second, a Contextual Resolution Adversarial Module (CRAM) explicitly addresses scale variance and distortion by adversarially aligning fine-grained details from high-resolution crops with global semantics from low-resolution contexts. DAPASS achieves state-of-the-art performances on outdoor (Cityscapes-to-DensePASS) and indoor (Stanford2D3D) benchmarks, yielding 55.04% (+2.05%) and 70.38% (+1.54%) mIoU, respectively.

  </details>



- **AnyDoc: Enhancing Document Generation via Large-Scale HTML/CSS Data Synthesis and Height-Aware Reinforcement Optimization**  
  Jiawei Lin, Wanrong Zhu, Vlad I Morariu, Christopher Tensmeyer  
  _2026-03-26_ · https://arxiv.org/abs/2603.25118v1  
  <details><summary>Abstract</summary>

  Document generation has gained growing attention in the field of AI-driven content creation. In this work, we push its boundaries by introducing AnyDoc, a framework capable of handling multiple generation tasks across a wide spectrum of document categories, all represented in a unified HTML/CSS format. To overcome the limited coverage and scale of existing human-crafted document datasets, AnyDoc first establishes a scalable data synthesis pipeline to automatically generate documents in HTML/CSS form. This pipeline yields DocHTML, a large-scale dataset containing 265,206 document samples, while spanning 111 categories and 32 distinct styles. Additionally, all documents are equipped with comprehensive metadata, including design intentions, HTML/CSS source code, visual assets, and rendered screenshots. Building on the curated dataset, AnyDoc fine-tunes multi-modal large language models (MLLMs) to achieve three practical document generation tasks: intention-to-document, document derendering, and element-to-document. To address the content overflow issue observed during fine-tuning, AnyDoc further incorporates a height-aware reinforcement learning (HARL) post-training procedure. By defining a reward function based on the difference between predicted and target document heights, overflow is penalized and gradually mitigated during HARL, thereby enhancing overall performance. Qualitative and quantitative experiments demonstrate that AnyDoc outperforms both general-purpose MLLMs and task-specific baselines across all three tasks.

  </details>



- **THEMIS: Towards Holistic Evaluation of MLLMs for Scientific Paper Fraud Forensics**  
  Tzu-Yen Ma, Bo Zhang, Zichen Tang, Junpeng Ding, Haolin Tian, Yuanze Li, Zhuodi Hao, Zixin Ding, Zirui Wang, Xinyu Yu, et al.  
  _2026-03-26_ · https://arxiv.org/abs/2603.25089v1  
  <details><summary>Abstract</summary>

  We present THEMIS, a novel multi-task benchmark designed to comprehensively evaluate multimodal large language models (MLLMs) on visual fraud reasoning within real-world academic scenarios. Compared to existing benchmarks, THEMIS introduces three major advances. (1) Real-World Scenarios and Complexity: Our benchmark comprises over 4,000 questions spanning seven scenarios, derived from authentic retracted-paper cases and carefully curated multimodal synthetic data. With 60.47% complex-texture images, THEMIS bridges the critical gap between existing benchmarks and the complexity of real-world academic fraud. (2) Fraud-Type Diversity and Granularity: THEMIS systematically covers five challenging fraud types and introduces 16 fine-grained manipulation operations. On average, each sample undergoes multiple stacked manipulation operations, with the diversity and difficulty of these manipulations demanding a high level of visual fraud reasoning from the models. (3) Multi-Dimensional Capability Evaluation: We establish a mapping from fraud types to five core visual fraud reasoning capabilities, thereby enabling an evaluation that reveals the distinct strengths and specific weaknesses of different models across these core capabilities. Experiments on 16 leading MLLMs show that even the best-performing model, GPT-5, achieves an overall performance of only 56.15%, demonstrating that our benchmark presents a stringent test. We expect THEMIS to advance the development of MLLMs for complex, real-world fraud reasoning tasks.

  </details>



- **Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields**  
  Thanh-Hai Le, Hoang-Hau Tran, Trong-Nghia Vu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25008v1  
  <details><summary>Abstract</summary>

  This paper presents Few TensoRF, a 3D reconstruction framework that combines TensorRF's efficient tensor based representation with FreeNeRF's frequency driven few shot regularization. Using TensorRF to significantly accelerate rendering speed and introducing frequency and occlusion masks, the method improves stability and reconstruction quality under sparse input views. Experiments on the Synthesis NeRF benchmark show that Few TensoRF method improves the average PSNR from 21.45 dB (TensorRF) to 23.70 dB, with the fine tuned version reaching 24.52 dB, while maintaining TensorRF's fast \(\approx10-15\) minute training time. Experiments on the THuman 2.0 dataset further demonstrate competitive performance in human body reconstruction, achieving 27.37 - 34.00 dB with only eight input images. These results highlight Few TensoRF as an efficient and data effective solution for real-time 3D reconstruction across diverse scenes.

  </details>


