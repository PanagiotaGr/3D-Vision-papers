# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-02-04 07:08 UTC_

Total papers shown: **50**


---

- **FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation**  
  Zimu Lu, Houxing Ren, Yunqiao Yang, Ke Wang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  
  _2026-02-03_ · https://arxiv.org/abs/2602.03798v1  
  <details><summary>Abstract</summary>

  Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at https://github.com/mnluzimu/FullStack-Agent.

  </details>



- **RAWDet-7: A Multi-Scenario Benchmark for Object Detection and Description on Quantized RAW Images**  
  Mishal Fatima, Shashank Agnihotri, Kanchana Vaishnavi Gandikota, Michael Moeller, Margret Keuper  
  _2026-02-03_ · https://arxiv.org/abs/2602.03760v1  
  <details><summary>Abstract</summary>

  Most vision models are trained on RGB images processed through ISP pipelines optimized for human perception, which can discard sensor-level information useful for machine reasoning. RAW images preserve unprocessed scene data, enabling models to leverage richer cues for both object detection and object description, capturing fine-grained details, spatial relationships, and contextual information often lost in processed images. To support research in this domain, we introduce RAWDet-7, a large-scale dataset of ~25k training and 7.6k test RAW images collected across diverse cameras, lighting conditions, and environments, densely annotated for seven object categories following MS-COCO and LVIS conventions. In addition, we provide object-level descriptions derived from the corresponding high-resolution sRGB images, facilitating the study of object-level information preservation under RAW image processing and low-bit quantization. The dataset allows evaluation under simulated 4-bit, 6-bit, and 8-bit quantization, reflecting realistic sensor constraints, and provides a benchmark for studying detection performance, description quality & detail, and generalization in low-bit RAW image processing. Dataset & code upon acceptance.

  </details>



- **Edge-Optimized Vision-Language Models for Underground Infrastructure Assessment**  
  Johny J. Lopez, Md Meftahul Ferdaus, Mahdi Abdelguerfi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03742v1  
  <details><summary>Abstract</summary>

  Autonomous inspection of underground infrastructure, such as sewer and culvert systems, is critical to public safety and urban sustainability. Although robotic platforms equipped with visual sensors can efficiently detect structural deficiencies, the automated generation of human-readable summaries from these detections remains a significant challenge, especially on resource-constrained edge devices. This paper presents a novel two-stage pipeline for end-to-end summarization of underground deficiencies, combining our lightweight RAPID-SCAN segmentation model with a fine-tuned Vision-Language Model (VLM) deployed on an edge computing platform. The first stage employs RAPID-SCAN (Resource-Aware Pipeline Inspection and Defect Segmentation using Compact Adaptive Network), achieving 0.834 F1-score with only 0.64M parameters for efficient defect segmentation. The second stage utilizes a fine-tuned Phi-3.5 VLM that generates concise, domain-specific summaries in natural language from the segmentation outputs. We introduce a curated dataset of inspection images with manually verified descriptions for VLM fine-tuning and evaluation. To enable real-time performance, we employ post-training quantization with hardware-specific optimization, achieving significant reductions in model size and inference latency without compromising summarization quality. We deploy and evaluate our complete pipeline on a mobile robotic platform, demonstrating its effectiveness in real-world inspection scenarios. Our results show the potential of edge-deployable integrated AI systems to bridge the gap between automated defect detection and actionable insights for infrastructure maintenance, paving the way for more scalable and autonomous inspection solutions.

  </details>



- **RegionReasoner: Region-Grounded Multi-Round Visual Reasoning**  
  Wenfang Sun, Hao Chen, Yingjun Du, Yefeng Zheng, Cees G. M. Snoek  
  _2026-02-03_ · https://arxiv.org/abs/2602.03733v1  
  <details><summary>Abstract</summary>

  Large vision-language models have achieved remarkable progress in visual reasoning, yet most existing systems rely on single-step or text-only reasoning, limiting their ability to iteratively refine understanding across multiple visual contexts. To address this limitation, we introduce a new multi-round visual reasoning benchmark with training and test sets spanning both detection and segmentation tasks, enabling systematic evaluation under iterative reasoning scenarios. We further propose RegionReasoner, a reinforcement learning framework that enforces grounded reasoning by requiring each reasoning trace to explicitly cite the corresponding reference bounding boxes, while maintaining semantic coherence via a global-local consistency reward. This reward extracts key objects and nouns from both global scene captions and region-level captions, aligning them with the reasoning trace to ensure consistency across reasoning steps. RegionReasoner is optimized with structured rewards combining grounding fidelity and global-local semantic alignment. Experiments on detection and segmentation tasks show that RegionReasoner-7B, together with our newly introduced benchmark RegionDial-Bench, considerably improves multi-round reasoning accuracy, spatial grounding precision, and global-local consistency, establishing a strong baseline for this emerging research direction.

  </details>



- **Referring Industrial Anomaly Segmentation**  
  Pengfei Yue, Xiaokang Jiang, Yilin Lu, Jianghang Lin, Shengchuan Zhang, Liujuan Cao  
  _2026-02-03_ · https://arxiv.org/abs/2602.03673v1  
  <details><summary>Abstract</summary>

  Industrial Anomaly Detection (IAD) is vital for manufacturing, yet traditional methods face significant challenges: unsupervised approaches yield rough localizations requiring manual thresholds, while supervised methods overfit due to scarce, imbalanced data. Both suffer from the "One Anomaly Class, One Model" limitation. To address this, we propose Referring Industrial Anomaly Segmentation (RIAS), a paradigm leveraging language to guide detection. RIAS generates precise masks from text descriptions without manual thresholds and uses universal prompts to detect diverse anomalies with a single model. We introduce the MVTec-Ref dataset to support this, designed with diverse referring expressions and focusing on anomaly patterns, notably with 95% small anomalies. We also propose the Dual Query Token with Mask Group Transformer (DQFormer) benchmark, enhanced by Language-Gated Multi-Level Aggregation (LMA) to improve multi-scale segmentation. Unlike traditional methods using redundant queries, DQFormer employs only "Anomaly" and "Background" tokens for efficient visual-textual integration. Experiments demonstrate RIAS's effectiveness in advancing IAD toward open-set capabilities. Code: https://github.com/swagger-coder/RIAS-MVTec-Ref.

  </details>



- **MM-SCALE: Grounded Multimodal Moral Reasoning via Scalar Judgment and Listwise Alignment**  
  Eunkyu Park, Wesley Hanwen Deng, Cheyon Jin, Matheus Kunzler Maldaner, Jordan Wheeler, Jason I. Hong, Hong Shen, Adam Perer, Ken Holstein, Motahhare Eslami, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03665v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) continue to struggle to make morally salient judgments in multimodal and socially ambiguous contexts. Prior works typically rely on binary or pairwise supervision, which often fail to capture the continuous and pluralistic nature of human moral reasoning. We present MM-SCALE (Multimodal Moral Scale), a large-scale dataset for aligning VLMs with human moral preferences through 5-point scalar ratings and explicit modality grounding. Each image-scenario pair is annotated with moral acceptability scores and grounded reasoning labels by humans using an interface we tailored for data collection, enabling listwise preference optimization over ranked scenario sets. By moving from discrete to scalar supervision, our framework provides richer alignment signals and finer calibration of multimodal moral reasoning. Experiments show that VLMs fine-tuned on MM-SCALE achieve higher ranking fidelity and more stable safety calibration than those trained with binary signals.

  </details>



- **SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection**  
  Wei Zhang, Xiang Liu, Ningjing Liu, Mingxin Liu, Wei Liao, Chunyan Xu, Xue Yang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03634v1  
  <details><summary>Abstract</summary>

  A consistent trend throughout the research of oriented object detection has been the pursuit of maintaining comparable performance with fewer and weaker annotations. This is particularly crucial in the remote sensing domain, where the dense object distribution and a wide variety of categories contribute to prohibitively high costs. Based on the supervision level, existing oriented object detection algorithms can be broadly grouped into fully supervised, semi-supervised, and weakly supervised methods. Within the scope of this work, we further categorize them to include sparsely supervised and partially weakly-supervised methods. To address the challenges of large-scale labeling, we introduce the first Sparse Partial Weakly-Supervised Oriented Object Detection framework, designed to efficiently leverage only a few sparse weakly-labeled data and plenty of unlabeled data. Our framework incorporates three key innovations: (1) We design a Sparse-annotation-Orientation-and-Scale-aware Student (SOS-Student) model to separate unlabeled objects from the background in a sparsely-labeled setting, and learn orientation and scale information from orientation-agnostic or scale-agnostic weak annotations. (2) We construct a novel Multi-level Pseudo-label Filtering strategy that leverages the distribution of model predictions, which is informed by the model's multi-layer predictions. (3) We propose a unique sparse partitioning approach, ensuring equal treatment for each category. Extensive experiments on the DOTA and DIOR datasets show that our framework achieves a significant performance gain over traditional oriented object detection methods mentioned above, offering a highly cost-effective solution. Our code is publicly available at https://github.com/VisionXLab/SPWOOD.

  </details>



- **KTV: Keyframes and Key Tokens Selection for Efficient Training-Free Video LLMs**  
  Baiyang Song, Jun Peng, Yuxin Zhang, Guangyao Chen, Feidiao Yang, Jianyuan Guo  
  _2026-02-03_ · https://arxiv.org/abs/2602.03615v1  
  <details><summary>Abstract</summary>

  Training-free video understanding leverages the strong image comprehension capabilities of pre-trained vision language models (VLMs) by treating a video as a sequence of static frames, thus obviating the need for costly video-specific training. However, this paradigm often suffers from severe visual redundancy and high computational overhead, especially when processing long videos. Crucially, existing keyframe selection strategies, especially those based on CLIP similarity, are prone to biases and may inadvertently overlook critical frames, resulting in suboptimal video comprehension. To address these significant challenges, we propose \textbf{KTV}, a novel two-stage framework for efficient and effective training-free video understanding. In the first stage, KTV performs question-agnostic keyframe selection by clustering frame-level visual features, yielding a compact, diverse, and representative subset of frames that mitigates temporal redundancy. In the second stage, KTV applies key visual token selection, pruning redundant or less informative tokens from each selected keyframe based on token importance and redundancy, which significantly reduces the number of tokens fed into the LLM. Extensive experiments on the Multiple-Choice VideoQA task demonstrate that KTV outperforms state-of-the-art training-free baselines while using significantly fewer visual tokens, \emph{e.g.}, only 504 visual tokens for a 60-min video with 10800 frames, achieving $44.8\%$ accuracy on the MLVU-Test benchmark. In particular, KTV also exceeds several training-based approaches on certain benchmarks.

  </details>



- **High-Resolution Underwater Camouflaged Object Detection: GBU-UCOD Dataset and Topology-Aware and Frequency-Decoupled Networks**  
  Wenji Wu, Shuo Ye, Yiyu Liu, Jiguang He, Zhuo Wang, Zitong Yu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03591v1  
  <details><summary>Abstract</summary>

  Underwater Camouflaged Object Detection (UCOD) is a challenging task due to the extreme visual similarity between targets and backgrounds across varying marine depths. Existing methods often struggle with topological fragmentation of slender creatures in the deep sea and the subtle feature extraction of transparent organisms. In this paper, we propose DeepTopo-Net, a novel framework that integrates topology-aware modeling with frequency-decoupled perception. To address physical degradation, we design the Water-Conditioned Adaptive Perceptor (WCAP), which employs Riemannian metric tensors to dynamically deform convolutional sampling fields. Furthermore, the Abyssal-Topology Refinement Module (ATRM) is developed to maintain the structural connectivity of spindly targets through skeletal priors. Specifically, we first introduce GBU-UCOD, the first high-resolution (2K) benchmark tailored for marine vertical zonation, filling the data gap for hadal and abyssal zones. Extensive experiments on MAS3K, RMAS, and our proposed GBU-UCOD datasets demonstrate that DeepTopo-Net achieves state-of-the-art performance, particularly in preserving the morphological integrity of complex underwater patterns. The datasets and codes will be released at https://github.com/Wuwenji18/GBU-UCOD.

  </details>



- **SlowFocus: Enhancing Fine-grained Temporal Understanding in Video LLM**  
  Ming Nie, Dan Ding, Chunwei Wang, Yuanfan Guo, Jianhua Han, Hang Xu, Li Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03589v1  
  <details><summary>Abstract</summary>

  Large language models (LLMs) have demonstrated exceptional capabilities in text understanding, which has paved the way for their expansion into video LLMs (Vid-LLMs) to analyze video data. However, current Vid-LLMs struggle to simultaneously retain high-quality frame-level semantic information (i.e., a sufficient number of tokens per frame) and comprehensive video-level temporal information (i.e., an adequate number of sampled frames per video). This limitation hinders the advancement of Vid-LLMs towards fine-grained video understanding. To address this issue, we introduce the SlowFocus mechanism, which significantly enhances the equivalent sampling frequency without compromising the quality of frame-level visual tokens. SlowFocus begins by identifying the query-related temporal segment based on the posed question, then performs dense sampling on this segment to extract local high-frequency features. A multi-frequency mixing attention module is further leveraged to aggregate these local high-frequency details with global low-frequency contexts for enhanced temporal comprehension. Additionally, to tailor Vid-LLMs to this innovative mechanism, we introduce a set of training strategies aimed at bolstering both temporal grounding and detailed temporal reasoning capabilities. Furthermore, we establish FineAction-CGR, a benchmark specifically devised to assess the ability of Vid-LLMs to process fine-grained temporal understanding tasks. Comprehensive experiments demonstrate the superiority of our mechanism across both existing public video understanding benchmarks and our proposed FineAction-CGR.

  </details>



- **AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping**  
  Dingyi Zhou, Mu He, Zhuowei Fang, Xiangtong Yao, Yinlong Liu, Alois Knoll, Hu Cao  
  _2026-02-03_ · https://arxiv.org/abs/2602.03547v1  
  <details><summary>Abstract</summary>

  We introduce AffordanceGrasp-R1, a reasoning-driven affordance segmentation framework for robotic grasping that combines a chain-of-thought (CoT) cold-start strategy with reinforcement learning to enhance deduction and spatial grounding. In addition, we redesign the grasping pipeline to be more context-aware by generating grasp candidates from the global scene point cloud and subsequently filtering them using instruction-conditioned affordance masks. Extensive experiments demonstrate that AffordanceGrasp-R1 consistently outperforms state-of-the-art (SOTA) methods on benchmark datasets, and real-world robotic grasping evaluations further validate its robustness and generalization under complex language-conditioned manipulation scenarios.

  </details>



- **Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance**  
  Yingjie Zhu, Xuefeng Bai, Kehai Chen, Yang Xiang, Youcheng Pan, Xiaoqiang Zhou, Min Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03491v1  
  <details><summary>Abstract</summary>

  Reasoning over table images remains challenging for Large Vision-Language Models (LVLMs) due to complex layouts and tightly coupled structure-content information. Existing solutions often depend on expensive supervised training, reinforcement learning, or external tools, limiting efficiency and scalability. This work addresses a key question: how to adapt LVLMs to table reasoning with minimal annotation and no external tools? Specifically, we first introduce DiSCo, a Disentangled Structure-Content alignment framework that explicitly separates structural abstraction from semantic grounding during multimodal alignment, efficiently adapting LVLMs to tables structures. Building on DiSCo, we further present Table-GLS, a Global-to-Local Structure-guided reasoning framework that performs table reasoning via structured exploration and evidence-grounded inference. Extensive experiments across diverse benchmarks demonstrate that our framework efficiently enhances LVLM's table understanding and reasoning capabilities, particularly generalizing to unseen table structures.

  </details>



- **Scaling Continual Learning with Bi-Level Routing Mixture-of-Experts**  
  Meng Lou, Yunxiang Fu, Yizhou Yu  
  _2026-02-03_ · https://arxiv.org/abs/2602.03473v1  
  <details><summary>Abstract</summary>

  Continual learning, especially class-incremental learning (CIL), on the basis of a pre-trained model (PTM) has garnered substantial research interest in recent years. However, how to effectively learn both discriminative and comprehensive feature representations while maintaining stability and plasticity over very long task sequences remains an open problem. We propose CaRE, a scalable {C}ontinual Le{a}rner with efficient Bi-Level {R}outing Mixture-of-{E}xperts (BR-MoE). The core idea of BR-MoE is a bi-level routing mechanism: a router selection stage that dynamically activates relevant task-specific routers, followed by an expert routing phase that dynamically activates and aggregates experts, aiming to inject discriminative and comprehensive representations into every intermediate network layer. On the other hand, we introduce a challenging evaluation protocol for comprehensively assessing CIL methods across very long task sequences spanning hundreds of tasks. Extensive experiments show that CaRE demonstrates leading performance across a variety of datasets and task settings, including commonly used CIL datasets with classical CIL settings (e.g., 5-20 tasks). To the best of our knowledge, CaRE is the first continual learner that scales to very long task sequences (ranging from 100 to over 300 non-overlapping tasks), while outperforming all baselines by a large margin on such task sequences. Code will be publicly released at https://github.com/LMMMEng/CaRE.git.

  </details>



- **HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic**  
  Yu-Hsiang Chen, Wei-Jer Chang, Christian Kotulla, Thomas Keutgens, Steffen Runde, Tobias Moers, Christoph Klas, Wei Zhan, Masayoshi Tomizuka, Yi-Ting Chen  
  _2026-02-03_ · https://arxiv.org/abs/2602.03447v1  
  <details><summary>Abstract</summary>

  We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: https://hetroddata.github.io/HetroD/

  </details>



- **CRL-VLA: Continual Vision-Language-Action Learning**  
  Qixin Zeng, Shuo Zhang, Hongyin Zhang, Renjie Wang, Han Zhao, Libang Zhao, Runze Li, Donglin Wang, Chao Huang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03445v1  
  <details><summary>Abstract</summary>

  Lifelong learning is critical for embodied agents in open-world environments, where reinforcement learning fine-tuning has emerged as an important paradigm to enable Vision-Language-Action (VLA) models to master dexterous manipulation through environmental interaction. Thus, Continual Reinforcement Learning (CRL) is a promising pathway for deploying VLA models in lifelong robotic scenarios, yet balancing stability (retaining old skills) and plasticity (learning new ones) remains a formidable challenge for existing methods. We introduce CRL-VLA, a framework for continual post-training of VLA models with rigorous theoretical bounds. We derive a unified performance bound linking the stability-plasticity trade-off to goal-conditioned advantage magnitude, scaled by policy divergence. CRL-VLA resolves this dilemma via asymmetric regulation: constraining advantage magnitudes on prior tasks while enabling controlled growth on new tasks. This is realized through a simple but effective dual-critic architecture with novel Goal-Conditioned Value Formulation (GCVF), where a frozen critic anchors semantic consistency and a trainable estimator drives adaptation. Experiments on the LIBERO benchmark demonstrate that CRL-VLA effectively harmonizes these conflicting objectives, outperforming baselines in both anti-forgetting and forward adaptation.

  </details>



- **Model-based Optimal Control for Rigid-Soft Underactuated Systems**  
  Daniele Caradonna, Nikhil Nair, Anup Teejo Mathew, Daniel Feliu Talegón, Imran Afgan, Egidio Falotico, Cosimo Della Santina, Federico Renda  
  _2026-02-03_ · https://arxiv.org/abs/2602.03435v1  
  <details><summary>Abstract</summary>

  Continuum soft robots are inherently underactuated and subject to intrinsic input constraints, making dynamic control particularly challenging, especially in hybrid rigid-soft robots. While most existing methods focus on quasi-static behaviors, dynamic tasks such as swing-up require accurate exploitation of continuum dynamics. This has led to studies on simple low-order template systems that often fail to capture the complexity of real continuum deformations. Model-based optimal control offers a systematic solution; however, its application to rigid-soft robots is often limited by the computational cost and inaccuracy of numerical differentiation for high-dimensional models. Building on recent advances in the Geometric Variable Strain model that enable analytical derivatives, this work investigates three optimal control strategies for underactuated soft systems-Direct Collocation, Differential Dynamic Programming, and Nonlinear Model Predictive Control-to perform dynamic swing-up tasks. To address stiff continuum dynamics and constrained actuation, implicit integration schemes and warm-start strategies are employed to improve numerical robustness and computational efficiency. The methods are evaluated in simulation on three Rigid-Soft and high-order soft benchmark systems-the Soft Cart-Pole, the Soft Pendubot, and the Soft Furuta Pendulum- highlighting their performance and computational trade-offs.

  </details>



- **ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response**  
  Xiaomeng Zhu, Fengming Zhu, Weijie Zhou, Ye Tian, Zhenlin Hu, Yufei Huang, Yuchun Guo, Xinyu Wu, Zhengyou Zhang, Fangzhen Lin, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03430v1  
  <details><summary>Abstract</summary>

  While passive agents merely follow instructions, proactive agents align with higher-level objectives, such as assistance and safety by continuously monitoring the environment to determine when and how to act. However, developing proactive agents is hindered by the lack of specialized resources. To address this, we introduce ProAct-75, a benchmark designed to train and evaluate proactive agents across diverse domains, including assistance, maintenance, and safety monitoring. Spanning 75 tasks, our dataset features 91,581 step-level annotations enriched with explicit task graphs. These graphs encode step dependencies and parallel execution possibilities, providing the structural grounding necessary for complex decision-making. Building on this benchmark, we propose ProAct-Helper, a reference baseline powered by a Multimodal Large Language Model (MLLM) that grounds decision-making in state detection, and leveraging task graphs to enable entropy-driven heuristic search for action selection, allowing agents to execute parallel threads independently rather than mirroring the human's next step. Extensive experiments demonstrate that ProAct-Helper outperforms strong closed-source models, improving trigger detection mF1 by 6.21%, saving 0.25 more steps in online one-step decision, and increasing the rate of parallel actions by 15.58%.

  </details>



- **Socratic-Geo: Synthetic Data Generation and Geometric Reasoning via Multi-Agent Interaction**  
  Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Wei Wang, Bing Zhao, Hu Wei, Linfeng Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03414v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have significantly advanced vision-language understanding. However, even state-of-the-art models struggle with geometric reasoning, revealing a critical bottleneck: the extreme scarcity of high-quality image-text pairs. Human annotation is prohibitively expensive, while automated methods fail to ensure fidelity and training effectiveness. Existing approaches either passively adapt to available images or employ inefficient random exploration with filtering, decoupling generation from learning needs. We propose Socratic-Geo, a fully autonomous framework that dynamically couples data synthesis with model learning through multi-agent interaction. The Teacher agent generates parameterized Python scripts with reflective feedback (Reflect for solvability, RePI for visual validity), ensuring image-text pair purity. The Solver agent optimizes reasoning through preference learning, with failure paths guiding Teacher's targeted augmentation. Independently, the Generator learns image generation capabilities on accumulated "image-code-instruction" triplets, distilling programmatic drawing intelligence into visual generation. Starting from only 108 seed problems, Socratic-Solver achieves 49.11 on six benchmarks using one-quarter of baseline data, surpassing strong baselines by 2.43 points. Socratic-Generator achieves 42.4% on GenExam, establishing new state-of-the-art for open-source models, surpassing Seedream-4.0 (39.8%) and approaching Gemini-2.5-Flash-Image (43.1%).

  </details>



- **Symbol-Aware Reasoning with Masked Discrete Diffusion for Handwritten Mathematical Expression Recognition**  
  Takaya Kawakatsu, Ryo Ishiyama  
  _2026-02-03_ · https://arxiv.org/abs/2602.03370v1  
  <details><summary>Abstract</summary>

  Handwritten Mathematical Expression Recognition (HMER) requires reasoning over diverse symbols and 2D structural layouts, yet autoregressive models struggle with exposure bias and syntactic inconsistency. We present a discrete diffusion framework that reformulates HMER as iterative symbolic refinement instead of sequential generation. Through multi-step remasking, the proposal progressively refines both symbols and structural relations, removing causal dependencies and improving structural consistency. A symbol-aware tokenization and Random-Masking Mutual Learning further enhance syntactic alignment and robustness to handwriting diversity. On the MathWriting benchmark, the proposal achieves 5.56\% CER and 60.42\% EM, outperforming strong Transformer and commercial baselines. Consistent gains on CROHME 2014--2023 demonstrate that discrete diffusion provides a new paradigm for structure-aware visual recognition beyond generative modeling.

  </details>



- **Global Geometry Is Not Enough for Vision Representations**  
  Jiwan Chung, Seon Joo Kim  
  _2026-02-03_ · https://arxiv.org/abs/2602.03282v1  
  <details><summary>Abstract</summary>

  A common assumption in representation learning is that globally well-distributed embeddings support robust and generalizable representations. This focus has shaped both training objectives and evaluation protocols, implicitly treating global geometry as a proxy for representational competence. While global geometry effectively encodes which elements are present, it is often insensitive to how they are composed. We investigate this limitation by testing the ability of geometric metrics to predict compositional binding across 21 vision encoders. We find that standard geometry-based statistics exhibit near-zero correlation with compositional binding. In contrast, functional sensitivity, as measured by the input-output Jacobian, reliably tracks this capability. We further provide an analytic account showing that this disparity arises from objective design, as existing losses explicitly constrain embedding geometry but leave the local input-output mapping unconstrained. These results suggest that global embedding geometry captures only a partial view of representational competence and establish functional sensitivity as a critical complementary axis for modeling composite structure.

  </details>



- **HypCBC: Domain-Invariant Hyperbolic Cross-Branch Consistency for Generalizable Medical Image Analysis**  
  Francesco Di Salvo, Sebastian Doerrich, Jonas Alle, Christian Ledig  
  _2026-02-03_ · https://arxiv.org/abs/2602.03264v1  
  <details><summary>Abstract</summary>

  Robust generalization beyond training distributions remains a critical challenge for deep neural networks. This is especially pronounced in medical image analysis, where data is often scarce and covariate shifts arise from different hardware devices, imaging protocols, and heterogeneous patient populations. These factors collectively hinder reliable performance and slow down clinical adoption. Despite recent progress, existing learning paradigms primarily rely on the Euclidean manifold, whose flat geometry fails to capture the complex, hierarchical structures present in clinical data. In this work, we exploit the advantages of hyperbolic manifolds to model complex data characteristics. We present the first comprehensive validation of hyperbolic representation learning for medical image analysis and demonstrate statistically significant gains across eleven in-distribution datasets and three ViT models. We further propose an unsupervised, domain-invariant hyperbolic cross-branch consistency constraint. Extensive experiments confirm that our proposed method promotes domain-invariant features and outperforms state-of-the-art Euclidean methods by an average of $+2.1\%$ AUC on three domain generalization benchmarks: Fitzpatrick17k, Camelyon17-WILDS, and a cross-dataset setup for retinal imaging. These datasets span different imaging modalities, data sizes, and label granularities, confirming generalization capabilities across substantially different conditions. The code is available at https://github.com/francescodisalvo05/hyperbolic-cross-branch-consistency .

  </details>



- **LaVPR: Benchmarking Language and Vision for Place Recognition**  
  Ofer Idan, Dan Badur, Yosi Keller, Yoli Shavit  
  _2026-02-03_ · https://arxiv.org/abs/2602.03253v1  
  <details><summary>Abstract</summary>

  Visual Place Recognition (VPR) often fails under extreme environmental changes and perceptual aliasing. Furthermore, standard systems cannot perform "blind" localization from verbal descriptions alone, a capability needed for applications such as emergency response. To address these challenges, we introduce LaVPR, a large-scale benchmark that extends existing VPR datasets with over 650,000 rich natural-language descriptions. Using LaVPR, we investigate two paradigms: Multi-Modal Fusion for enhanced robustness and Cross-Modal Retrieval for language-based localization. Our results show that language descriptions yield consistent gains in visually degraded conditions, with the most significant impact on smaller backbones. Notably, adding language allows compact models to rival the performance of much larger vision-only architectures. For cross-modal retrieval, we establish a baseline using Low-Rank Adaptation (LoRA) and Multi-Similarity loss, which substantially outperforms standard contrastive methods across vision-language models. Ultimately, LaVPR enables a new class of localization systems that are both resilient to real-world stochasticity and practical for resource-constrained deployment. Our dataset and code are available at https://github.com/oferidan1/LaVPR.

  </details>



- **InstaDrive: Instance-Aware Driving World Models for Realistic and Consistent Video Generation**  
  Zhuoran Yang, Xi Guo, Chenjing Ding, Chiyu Wang, Wei Wu, Yanyong Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03242v1  
  <details><summary>Abstract</summary>

  Autonomous driving relies on robust models trained on high-quality, large-scale multi-view driving videos. While world models offer a cost-effective solution for generating realistic driving videos, they struggle to maintain instance-level temporal consistency and spatial geometric fidelity. To address these challenges, we propose InstaDrive, a novel framework that enhances driving video realism through two key advancements: (1) Instance Flow Guider, which extracts and propagates instance features across frames to enforce temporal consistency, preserving instance identity over time. (2) Spatial Geometric Aligner, which improves spatial reasoning, ensures precise instance positioning, and explicitly models occlusion hierarchies. By incorporating these instance-aware mechanisms, InstaDrive achieves state-of-the-art video generation quality and enhances downstream autonomous driving tasks on the nuScenes dataset. Additionally, we utilize CARLA's autopilot to procedurally and stochastically simulate rare but safety-critical driving scenarios across diverse maps and regions, enabling rigorous safety evaluation for autonomous systems. Our project page is https://shanpoyang654.github.io/InstaDrive/page.html.

  </details>



- **EventFlash: Towards Efficient MLLMs for Event-Based Vision**  
  Shaoyu Liu, Jianing Li, Guanghui Zhao, Yunjian Zhang, Wen Jiang, Ming Li, Xiangyang Ji  
  _2026-02-03_ · https://arxiv.org/abs/2602.03230v1  
  <details><summary>Abstract</summary>

  Event-based multimodal large language models (MLLMs) enable robust perception in high-speed and low-light scenarios, addressing key limitations of frame-based MLLMs. However, current event-based MLLMs often rely on dense image-like processing paradigms, overlooking the spatiotemporal sparsity of event streams and resulting in high computational cost. In this paper, we propose EventFlash, a novel and efficient MLLM to explore spatiotemporal token sparsification for reducing data redundancy and accelerating inference. Technically, we build EventMind, a large-scale and scene-diverse dataset with over 500k instruction sets, providing both short and long event stream sequences to support our curriculum training strategy. We then present an adaptive temporal window aggregation module for efficient temporal sampling, which adaptively compresses temporal tokens while retaining key temporal cues. Finally, a sparse density-guided attention module is designed to improve spatial token efficiency by selecting informative regions and suppressing empty or sparse areas. Experimental results show that EventFlash achieves a $12.4\times$ throughput improvement over the baseline (EventFlash-Zero) while maintaining comparable performance. It supports long-range event stream processing with up to 1,000 bins, significantly outperforming the 5-bin limit of EventGPT. We believe EventFlash serves as an efficient foundation model for event-based vision.

  </details>



- **PokeFusion Attention: Enhancing Reference-Free Style-Conditioned Generation**  
  Jingbang Tang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03220v1  
  <details><summary>Abstract</summary>

  This paper studies reference-free style-conditioned character generation in text-to-image diffusion models, where high-quality synthesis requires both stable character structure and consistent, fine-grained style expression across diverse prompts. Existing approaches primarily rely on text-only prompting, which is often under-specified for visual style and tends to produce noticeable style drift and geometric inconsistency, or introduce reference-based adapters that depend on external images at inference time, increasing architectural complexity and limiting deployment flexibility.We propose PokeFusion Attention, a lightweight decoder-level cross-attention mechanism that fuses textual semantics with learned style embeddings directly inside the diffusion decoder. By decoupling text and style conditioning at the attention level, our method enables effective reference-free stylized generation while keeping the pretrained diffusion backbone fully frozen.PokeFusion Attention trains only decoder cross-attention layers together with a compact style projection module, resulting in a parameter-efficient and plug-and-play control component that can be easily integrated into existing diffusion pipelines and transferred across different backbones.Experiments on a stylized character generation benchmark (Pokemon-style) demonstrate that our method consistently improves style fidelity, semantic alignment, and character shape consistency compared with representative adapter-based baselines, while maintaining low parameter overhead and inference-time simplicity.

  </details>



- **ConsisDrive: Identity-Preserving Driving World Models for Video Generation by Instance Mask**  
  Zhuoran Yang, Yanyong Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03213v1  
  <details><summary>Abstract</summary>

  Autonomous driving relies on robust models trained on large-scale, high-quality multi-view driving videos. Although world models provide a cost-effective solution for generating realistic driving data, they often suffer from identity drift, where the same object changes its appearance or category across frames due to the absence of instance-level temporal constraints. We introduce ConsisDrive, an identity-preserving driving world model designed to enforce temporal consistency at the instance level. Our framework incorporates two key components: (1) Instance-Masked Attention, which applies instance identity masks and trajectory masks within attention blocks to ensure that visual tokens interact only with their corresponding instance features across spatial and temporal dimensions, thereby preserving object identity consistency; and (2) Instance-Masked Loss, which adaptively emphasizes foreground regions with probabilistic instance masking, reducing background noise while maintaining overall scene fidelity. By integrating these mechanisms, ConsisDrive achieves state-of-the-art driving video generation quality and demonstrates significant improvements in downstream autonomous driving tasks on the nuScenes dataset. Our project page is https://shanpoyang654.github.io/ConsisDrive/page.html.

  </details>



- **VIRAL: Visual In-Context Reasoning via Analogy in Diffusion Transformers**  
  Zhiwen Li, Zhongjie Duan, Jinyan Ye, Cen Chen, Daoyuan Chen, Yaliang Li, Yingda Chen  
  _2026-02-03_ · https://arxiv.org/abs/2602.03210v1  
  <details><summary>Abstract</summary>

  Replicating In-Context Learning (ICL) in computer vision remains challenging due to task heterogeneity. We propose \textbf{VIRAL}, a framework that elicits visual reasoning from a pre-trained image editing model by formulating ICL as conditional generation via visual analogy ($x_s : x_t :: x_q : y_q$). We adapt a frozen Diffusion Transformer (DiT) using role-aware multi-image conditioning and introduce a Mixture-of-Experts LoRA to mitigate gradient interference across diverse tasks. Additionally, to bridge the gaps in current visual context datasets, we curate a large-scale dataset spanning perception, restoration, and editing. Experiments demonstrate that VIRAL outperforms existing methods, validating that a unified V-ICL paradigm can handle the majority of visual tasks, including open-domain editing. Our code is available at https://anonymous.4open.science/r/VIRAL-744A

  </details>



- **Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements**  
  Marco Job, Thomas Stastny, Eleni Kelasidi, Roland Siegwart, Michael Pantic  
  _2026-02-03_ · https://arxiv.org/abs/2602.03209v1  
  <details><summary>Abstract</summary>

  Autonomous field robots operating in unstructured environments require robust perception to ensure safe and reliable operations. Recent advances in monocular depth estimation have demonstrated the potential of low-cost cameras as depth sensors; however, their adoption in field robotics remains limited due to the absence of reliable scale cues, ambiguous or low-texture conditions, and the scarcity of large-scale datasets. To address these challenges, we propose a depth completion model that trains on synthetic data and uses extremely sparse measurements from depth sensors to predict dense metric depth in unseen field robotics environments. A synthetic dataset generation pipeline tailored to field robotics enables the creation of multiple realistic datasets for training purposes. This dataset generation approach utilizes textured 3D meshes from Structure from Motion and photorealistic rendering with novel viewpoint synthesis to simulate diverse field robotics scenarios. Our approach achieves an end-to-end latency of 53 ms per frame on a Nvidia Jetson AGX Orin, enabling real-time deployment on embedded platforms. Extensive evaluation demonstrates competitive performance across diverse real-world field robotics scenarios.

  </details>



- **FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion**  
  Chen-Bin Feng, Youyang Sha, Longfei Liu, Yongjun Yu, Chi Man Vong, Xuanlong Yu, Xi Shen  
  _2026-02-03_ · https://arxiv.org/abs/2602.03137v1  
  <details><summary>Abstract</summary>

  In this paper, we present FSOD-VFM: Few-Shot Object Detectors with Vision Foundation Models, a framework that leverages vision foundation models to tackle the challenge of few-shot object detection. FSOD-VFM integrates three key components: a universal proposal network (UPN) for category-agnostic bounding box generation, SAM2 for accurate mask extraction, and DINOv2 features for efficient adaptation to new object categories. Despite the strong generalization capabilities of foundation models, the bounding boxes generated by UPN often suffer from overfragmentation, covering only partial object regions and leading to numerous small, false-positive proposals rather than accurate, complete object detections. To address this issue, we introduce a novel graph-based confidence reweighting method. In our approach, predicted bounding boxes are modeled as nodes in a directed graph, with graph diffusion operations applied to propagate confidence scores across the network. This reweighting process refines the scores of proposals, assigning higher confidence to whole objects and lower confidence to local, fragmented parts. This strategy improves detection granularity and effectively reduces the occurrence of false-positive bounding box proposals. Through extensive experiments on Pascal-5$^i$, COCO-20$^i$, and CD-FSOD datasets, we demonstrate that our method substantially outperforms existing approaches, achieving superior performance without requiring additional training. Notably, on the challenging CD-FSOD dataset, which spans multiple datasets and domains, our FSOD-VFM achieves 31.6 AP in the 10-shot setting, substantially outperforming previous training-free methods that reach only 21.4 AP. Code is available at: https://intellindust-ai-lab.github.io/projects/FSOD-VFM.

  </details>



- **FinMTM: A Multi-Turn Multimodal Benchmark for Financial Reasoning and Agent Evaluation**  
  Chenxi Zhang, Ziliang Gan, Liyun Zhu, Youwei Pang, Qing Zhang, Rongjunchen Zhang  
  _2026-02-03_ · https://arxiv.org/abs/2602.03130v1  
  <details><summary>Abstract</summary>

  The financial domain poses substantial challenges for vision-language models (VLMs) due to specialized chart formats and knowledge-intensive reasoning requirements. However, existing financial benchmarks are largely single-turn and rely on a narrow set of question formats, limiting comprehensive evaluation in realistic application scenarios. To address this gap, we propose FinMTM, a multi-turn multimodal benchmark that expands diversity along both data and task dimensions. On the data side, we curate and annotate 11{,}133 bilingual (Chinese and English) financial QA pairs grounded in financial visuals, including candlestick charts, statistical plots, and report figures. On the task side, FinMTM covers single- and multiple-choice questions, multi-turn open-ended dialogues, and agent-based tasks. We further design task-specific evaluation protocols, including a set-overlap scoring rule for multiple-choice questions, a weighted combination of turn-level and session-level scores for multi-turn dialogues, and a composite metric that integrates planning quality with final outcomes for agent tasks. Extensive experimental evaluation of 22 VLMs reveal their limitations in fine-grained visual perception, long-context reasoning, and complex agent workflows.

  </details>



- **Flexible Geometric Guidance for Probabilistic Human Pose Estimation with Diffusion Models**  
  Francis Snelgar, Ming Xu, Stephen Gould, Liang Zheng, Akshay Asthana  
  _2026-02-03_ · https://arxiv.org/abs/2602.03126v1  
  <details><summary>Abstract</summary>

  3D human pose estimation from 2D images is a challenging problem due to depth ambiguity and occlusion. Because of these challenges the task is underdetermined, where there exists multiple -- possibly infinite -- poses that are plausible given the image. Despite this, many prior works assume the existence of a deterministic mapping and estimate a single pose given an image. Furthermore, methods based on machine learning require a large amount of paired 2D-3D data to train and suffer from generalization issues to unseen scenarios. To address both of these issues, we propose a framework for pose estimation using diffusion models, which enables sampling from a probability distribution over plausible poses which are consistent with a 2D image. Our approach falls under the guidance framework for conditional generation, and guides samples from an unconditional diffusion model, trained only on 3D data, using the gradients of the heatmaps from a 2D keypoint detector. We evaluate our method on the Human 3.6M dataset under best-of-$m$ multiple hypothesis evaluation, showing state-of-the-art performance among methods which do not require paired 2D-3D data for training. We additionally evaluate the generalization ability using the MPI-INF-3DHP and 3DPW datasets and demonstrate competitive performance. Finally, we demonstrate the flexibility of our framework by using it for novel tasks including pose generation and pose completion, without the need to train bespoke conditional models. We make code available at https://github.com/fsnelgar/diffusion_pose .

  </details>



- **A generalizable large-scale foundation model for musculoskeletal radiographs**  
  Shinn Kim, Soobin Lee, Kyoungseob Shin, Han-Soo Kim, Yongsung Kim, Minsu Kim, Juhong Nam, Somang Ko, Daeheon Kwon, Wook Huh, et al.  
  _2026-02-03_ · https://arxiv.org/abs/2602.03076v1  
  <details><summary>Abstract</summary>

  Artificial intelligence (AI) has shown promise in detecting and characterizing musculoskeletal diseases from radiographs. However, most existing models remain task-specific, annotation-dependent, and limited in generalizability across diseases and anatomical regions. Although a generalizable foundation model trained on large-scale musculoskeletal radiographs is clinically needed, publicly available datasets remain limited in size and lack sufficient diversity to enable training across a wide range of musculoskeletal conditions and anatomical sites. Here, we present SKELEX, a large-scale foundation model for musculoskeletal radiographs, trained using self-supervised learning on 1.2 million diverse, condition-rich images. The model was evaluated on 12 downstream diagnostic tasks and generally outperformed baselines in fracture detection, osteoarthritis grading, and bone tumor classification. Furthermore, SKELEX demonstrated zero-shot abnormality localization, producing error maps that identified pathologic regions without task-specific training. Building on this capability, we developed an interpretable, region-guided model for predicting bone tumors, which maintained robust performance on independent external datasets and was deployed as a publicly accessible web application. Overall, SKELEX provides a scalable, label-efficient, and generalizable AI framework for musculoskeletal imaging, establishing a foundation for both clinical translation and data-efficient research in musculoskeletal radiology.

  </details>



- **JRDB-Pose3D: A Multi-person 3D Human Pose and Shape Estimation Dataset for Robotics**  
  Sandika Biswas, Kian Izadpanah, Hamid Rezatofighi  
  _2026-02-03_ · https://arxiv.org/abs/2602.03064v1  
  <details><summary>Abstract</summary>

  Real-world scenes are inherently crowded. Hence, estimating 3D poses of all nearby humans, tracking their movements over time, and understanding their activities within social and environmental contexts are essential for many applications, such as autonomous driving, robot perception, robot navigation, and human-robot interaction. However, most existing 3D human pose estimation datasets primarily focus on single-person scenes or are collected in controlled laboratory environments, which restricts their relevance to real-world applications. To bridge this gap, we introduce JRDB-Pose3D, which captures multi-human indoor and outdoor environments from a mobile robotic platform. JRDB-Pose3D provides rich 3D human pose annotations for such complex and dynamic scenes, including SMPL-based pose annotations with consistent body-shape parameters and track IDs for each individual over time. JRDB-Pose3D contains, on average, 5-10 human poses per frame, with some scenes featuring up to 35 individuals simultaneously. The proposed dataset presents unique challenges, including frequent occlusions, truncated bodies, and out-of-frame body parts, which closely reflect real-world environments. Moreover, JRDB-Pose3D inherits all available annotations from the JRDB dataset, such as 2D pose, information about social grouping, activities, and interactions, full-scene semantic masks with consistent human- and object-level tracking, and detailed annotations for each individual, such as age, gender, and race, making it a holistic dataset for a wide range of downstream perception and human-centric understanding tasks.

  </details>



- **MUSE: A Multi-agent Framework for Unconstrained Story Envisioning via Closed-Loop Cognitive Orchestration**  
  Wenzhang Sun, Zhenyu Wang, Zhangchi Hu, Chunfeng Wang, Hao Li, Wei Chen  
  _2026-02-03_ · https://arxiv.org/abs/2602.03028v1  
  <details><summary>Abstract</summary>

  Generating long-form audio-visual stories from a short user prompt remains challenging due to an intent-execution gap, where high-level narrative intent must be preserved across coherent, shot-level multimodal generation over long horizons. Existing approaches typically rely on feed-forward pipelines or prompt-only refinement, which often leads to semantic drift and identity inconsistency as sequences grow longer. We address this challenge by formulating storytelling as a closed-loop constraint enforcement problem and propose MUSE, a multi-agent framework that coordinates generation through an iterative plan-execute-verify-revise loop. MUSE translates narrative intent into explicit, machine-executable controls over identity, spatial composition, and temporal continuity, and applies targeted multimodal feedback to correct violations during generation. To evaluate open-ended storytelling without ground-truth references, we introduce MUSEBench, a reference-free evaluation protocol validated by human judgments. Experiments demonstrate that MUSE substantially improves long-horizon narrative coherence, cross-modal identity consistency, and cinematic quality compared with representative baselines.

  </details>



- **A Reproducible Framework for Bias-Resistant Machine Learning on Small-Sample Neuroimaging Data**  
  Jagan Mohan Reddy Dwarampudi, Jennifer L Purks, Joshua Wong, Renjie Hu, Tania Banerjee  
  _2026-02-02_ · https://arxiv.org/abs/2602.02920v1  
  <details><summary>Abstract</summary>

  We introduce a reproducible, bias-resistant machine learning framework that integrates domain-informed feature engineering, nested cross-validation, and calibrated decision-threshold optimization for small-sample neuroimaging data. Conventional cross-validation frameworks that reuse the same folds for both model selection and performance estimation yield optimistically biased results, limiting reproducibility and generalization. Demonstrated on a high-dimensional structural MRI dataset of deep brain stimulation cognitive outcomes, the framework achieved a nested-CV balanced accuracy of 0.660\,$\pm$\,0.068 using a compact, interpretable subset selected via importance-guided ranking. By combining interpretability and unbiased evaluation, this work provides a generalizable computational blueprint for reliable machine learning in data-limited biomedical domains.

  </details>



- **A Random Matrix Theory Perspective on the Consistency of Diffusion Models**  
  Binxu Wang, Jacob Zavatone-Veth, Cengiz Pehlevan  
  _2026-02-02_ · https://arxiv.org/abs/2602.02908v1  
  <details><summary>Abstract</summary>

  Diffusion models trained on different, non-overlapping subsets of a dataset often produce strikingly similar outputs when given the same noise seed. We trace this consistency to a simple linear effect: the shared Gaussian statistics across splits already predict much of the generated images. To formalize this, we develop a random matrix theory (RMT) framework that quantifies how finite datasets shape the expectation and variance of the learned denoiser and sampling map in the linear setting. For expectations, sampling variability acts as a renormalization of the noise level through a self-consistent relation $σ^2 \mapsto κ(σ^2)$, explaining why limited data overshrink low-variance directions and pull samples toward the dataset mean. For fluctuations, our variance formulas reveal three key factors behind cross-split disagreement: \textit{anisotropy} across eigenmodes, \textit{inhomogeneity} across inputs, and overall scaling with dataset size. Extending deterministic-equivalence tools to fractional matrix powers further allows us to analyze entire sampling trajectories. The theory sharply predicts the behavior of linear diffusion models, and we validate its predictions on UNet and DiT architectures in their non-memorization regime, identifying where and how samples deviates across training data split. This provides a principled baseline for reproducibility in diffusion training, linking spectral properties of data to the stability of generative outputs.

  </details>



- **DoubleTake: Contrastive Reasoning for Faithful Decision-Making in Medical Imaging**  
  Daivik Patel, Shrenik Patel  
  _2026-02-02_ · https://arxiv.org/abs/2602.02894v1  
  <details><summary>Abstract</summary>

  Accurate decision making in medical imaging requires reasoning over subtle visual differences between confusable conditions, yet most existing approaches rely on nearest neighbor retrieval that returns redundant evidence and reinforces a single hypothesis. We introduce a contrastive, document-aware reference selection framework that constructs compact evidence sets optimized for discrimination rather than similarity by explicitly balancing visual relevance, embedding diversity, and source-level provenance using ROCO embeddings and metadata. While ROCO provides large-scale image-caption pairs, it does not specify how references should be selected for contrastive reasoning, and naive retrieval frequently yields near-duplicate figures from the same document. To address this gap, we release a reproducible reference selection protocol and curated reference bank that enable a systematic study of contrastive retrieval in medical image reasoning. Building on these contrastive evidence sets, we propose Counterfactual-Contrastive Inference, a confidence-aware reasoning framework that performs structured pairwise visual comparisons and aggregates evidence using margin-based decision rules with faithful abstention. On the MediConfusion benchmark, our approach achieves state-of-the-art performance, improving set-level accuracy by nearly 15% relative to prior methods while reducing confusion and improving individual accuracy.

  </details>



- **Self-Supervised Uncalibrated Multi-View Video Anonymization in the Operating Room**  
  Keqi Chen, Vinkle Srivastav, Armine Vardazaryan, Cindy Rolland, Didier Mutter, Nicolas Padoy  
  _2026-02-02_ · https://arxiv.org/abs/2602.02850v1  
  <details><summary>Abstract</summary>

  Privacy preservation is a prerequisite for using video data in Operating Room (OR) research. Effective anonymization relies on the exhaustive localization of every individual; even a single missed detection necessitates extensive manual correction. However, existing approaches face two critical scalability bottlenecks: (1) they usually require manual annotations of each new clinical site for high accuracy; (2) while multi-camera setups have been widely adopted to address single-view ambiguity, camera calibration is typically required whenever cameras are repositioned. To address these problems, we propose a novel self-supervised multi-view video anonymization framework consisting of whole-body person detection and whole-body pose estimation, without annotation or camera calibration. Our core strategy is to enhance the single-view detector by "retrieving" false negatives using temporal and multi-view context, and conducting self-supervised domain adaptation. We first run an off-the-shelf whole-body person detector in each view with a low-score threshold to gather candidate detections. Then, we retrieve the low-score false negatives that exhibit consistency with the high-score detections via tracking and self-supervised uncalibrated multi-view association. These recovered detections serve as pseudo labels to iteratively fine-tune the whole-body detector. Finally, we apply whole-body pose estimation on each detected person, and fine-tune the pose model using its own high-score predictions. Experiments on the 4D-OR dataset of simulated surgeries and our dataset of real surgeries show the effectiveness of our approach achieving over 97% recall. Moreover, we train a real-time whole-body detector using our pseudo labels, achieving comparable performance and highlighting our method's practical applicability. Code is available at https://github.com/CAMMA-public/OR_anonymization.

  </details>



- **LmPT: Conditional Point Transformer for Anatomical Landmark Detection on 3D Point Clouds**  
  Matteo Bastico, Pierre Onghena, David Ryckelynck, Beatriz Marcotegui, Santiago Velasco-Forero, Laurent Corté, Caroline Robine--Decourcelle, Etienne Decencière  
  _2026-02-02_ · https://arxiv.org/abs/2602.02808v1  
  <details><summary>Abstract</summary>

  Accurate identification of anatomical landmarks is crucial for various medical applications. Traditional manual landmarking is time-consuming and prone to inter-observer variability, while rule-based methods are often tailored to specific geometries or limited sets of landmarks. In recent years, anatomical surfaces have been effectively represented as point clouds, which are lightweight structures composed of spatial coordinates. Following this strategy and to overcome the limitations of existing landmarking techniques, we propose Landmark Point Transformer (LmPT), a method for automatic anatomical landmark detection on point clouds that can leverage homologous bones from different species for translational research. The LmPT model incorporates a conditioning mechanism that enables adaptability to different input types to conduct cross-species learning. We focus the evaluation of our approach on femoral landmarking using both human and newly annotated dog femurs, demonstrating its generalization and effectiveness across species. The code and dog femur dataset will be publicly available at: https://github.com/Pierreoo/LandmarkPointTransformer.

  </details>



- **Real-time topology-aware M-mode OCT segmentation for robotic deep anterior lamellar keratoplasty (DALK) guidance**  
  Rosalinda Xiong, Jinglun Yu, Yaning Wang, Ziyi Huang, Jin U. Kang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02798v1  
  <details><summary>Abstract</summary>

  Robotic deep anterior lamellar keratoplasty (DALK) requires accurate real time depth feedback to approach Descemet's membrane (DM) without perforation. M-mode intraoperative optical coherence tomography (OCT) provides high temporal resolution depth traces, but speckle noise, attenuation, and instrument induced shadowing often result in discontinuous or ambiguous layer interfaces that challenge anatomically consistent segmentation at deployment frame rates. We present a lightweight, topology aware M-mode segmentation pipeline based on UNeXt that incorporates anatomical topology regularization to stabilize boundary continuity and layer ordering under low signal to noise ratio conditions. The proposed system achieves end to end throughput exceeding 80 Hz measured over the complete preprocessing inference overlay pipeline on a single GPU, demonstrating practical real time guidance beyond model only timing. This operating margin provides temporal headroom to reject low quality or dropout frames while maintaining a stable effective depth update rate. Evaluation on a standard rabbit eye M-mode dataset using an established baseline protocol shows improved qualitative boundary stability compared with topology agnostic controls, while preserving deployable real time performance.

  </details>



- **Physics-based generation of multilayer corneal OCT data via Gaussian modeling and MCML for AI-driven diagnostic and surgical guidance applications**  
  Jinglun Yu, Yaning Wang, Rosalinda Xiong, Ziyi Huang, Kristina Irsch, Jin U. Kang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02755v1  
  <details><summary>Abstract</summary>

  Training deep learning models for corneal optical coherence tomography (OCT) imaging is limited by the availability of large, well-annotated datasets. We present a configurable Monte Carlo simulation framework that generates synthetic corneal B-scan optical OCT images with pixel-level five-layer segmentation labels derived directly from the simulation geometry. A five-layer corneal model with Gaussian surfaces captures curvature and thickness variability in healthy and keratoconic eyes. Each layer is assigned optical properties from the literature and light transport is simulated using Monte Carlo modeling of light transport in multi-layered tissues (MCML), while incorporating system features such as the confocal PSF and sensitivity roll-off. This approach produces over 10,000 high-resolution (1024x1024) image-label pairs and supports customization of geometry, photon count, noise, and system parameters. The resulting dataset enables systematic training, validation, and benchmarking of AI models under controlled, ground-truth conditions, providing a reproducible and scalable resource to support the development of diagnostic and surgical guidance applications in image-guided ophthalmology.

  </details>



- **Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion**  
  Dan Haramati, Carl Qi, Tal Daniel, Amy Zhang, Aviv Tamar, George Konidaris  
  _2026-02-02_ · https://arxiv.org/abs/2602.02722v1  
  <details><summary>Abstract</summary>

  We propose a hierarchical entity-centric framework for offline Goal-Conditioned Reinforcement Learning (GCRL) that combines subgoal decomposition with factored structure to solve long-horizon tasks in domains with multiple entities. Achieving long-horizon goals in complex environments remains a core challenge in Reinforcement Learning (RL). Domains with multiple entities are particularly difficult due to their combinatorial complexity. GCRL facilitates generalization across goals and the use of subgoal structure, but struggles with high-dimensional observations and combinatorial state-spaces, especially under sparse reward. We employ a two-level hierarchy composed of a value-based GCRL agent and a factored subgoal-generating conditional diffusion model. The RL agent and subgoal generator are trained independently and composed post hoc through selective subgoal generation based on the value function, making the approach modular and compatible with existing GCRL algorithms. We introduce new variations to benchmark tasks that highlight the challenges of multi-entity domains, and show that our method consistently boosts performance of the underlying RL agent on image-based long-horizon tasks with sparse rewards, achieving over 150% higher success rates on the hardest task in our suite and generalizing to increasing horizons and numbers of entities. Rollout videos are provided at: https://sites.google.com/view/hecrl

  </details>



- **End-to-end reconstruction of OCT optical properties and speckle-reduced structural intensity via physics-based learning**  
  Jinglun Yu, Yaning Wang, Wenhan Guo, Yuan Gao, Yu Sun, Jin U. Kang  
  _2026-02-02_ · https://arxiv.org/abs/2602.02721v1  
  <details><summary>Abstract</summary>

  Inverse scattering in optical coherence tomography (OCT) seeks to recover both structural images and intrinsic tissue optical properties, including refractive index, scattering coefficient, and anisotropy. This inverse problem is challenging due to attenuation, speckle noise, and strong coupling among parameters. We propose a regularized end-to-end deep learning framework that jointly reconstructs optical parameter maps and speckle-reduced OCT structural intensity for layer visualization. Trained with Monte Carlo-simulated ground truth, our network incorporates a physics-based OCT forward model that generates predicted signals from the estimated parameters, providing physics-consistent supervision for parameter recovery and artifact suppression. Experiments on the synthetic corneal OCT dataset demonstrate robust optical map recovery under noise, improved resolution, and enhanced structural fidelity. This approach enables quantitative multi-parameter tissue characterization and highlights the benefit of combining physics-informed modeling with deep learning for computational OCT.

  </details>



- **AdaptMMBench: Benchmarking Adaptive Multimodal Reasoning for Mode Selection and Reasoning Process**  
  Xintong Zhang, Xiaowen Zhang, Jongrong Wu, Zhi Gao, Shilin Yan, Zhenxin Diao, Kunpeng Gao, Xuanyan Chen, Yuwei Wu, Yunde Jia, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02676v1  
  <details><summary>Abstract</summary>

  Adaptive multimodal reasoning has emerged as a promising frontier in Vision-Language Models (VLMs), aiming to dynamically modulate between tool-augmented visual reasoning and text reasoning to enhance both effectiveness and efficiency. However, existing evaluations rely on static difficulty labels and simplistic metrics, which fail to capture the dynamic nature of difficulty relative to varying model capacities. Consequently, they obscure the distinction between adaptive mode selection and general performance while neglecting fine-grained process analyses. In this paper, we propose AdaptMMBench, a comprehensive benchmark for adaptive multimodal reasoning across five domains: real-world, OCR, GUI, knowledge, and math, encompassing both direct perception and complex reasoning tasks. AdaptMMBench utilizes a Matthews Correlation Coefficient (MCC) metric to evaluate the selection rationality of different reasoning modes, isolating this meta-cognition ability by dynamically identifying task difficulties based on models' capability boundaries. Moreover, AdaptMMBench facilitates multi-dimensional process evaluation across key step coverage, tool effectiveness, and computational efficiency. Our evaluation reveals that while adaptive mode selection scales with model capacity, it notably decouples from final accuracy. Conversely, key step coverage aligns with performance, though tool effectiveness remains highly inconsistent across model architectures.

  </details>



- **Multi-head automated segmentation by incorporating detection head into the contextual layer neural network**  
  Edwin Kys, Febian Febian  
  _2026-02-02_ · https://arxiv.org/abs/2602.02471v1  
  <details><summary>Abstract</summary>

  Deep learning based auto segmentation is increasingly used in radiotherapy, but conventional models often produce anatomically implausible false positives, or hallucinations, in slices lacking target structures. We propose a gated multi-head Transformer architecture based on Swin U-Net, augmented with inter-slice context integration and a parallel detection head, which jointly performs slice-level structure detection via a multi-layer perceptron and pixel-level segmentation through a context-enhanced stream. Detection outputs gate the segmentation predictions to suppress false positives in anatomically invalid slices, and training uses slice-wise Tversky loss to address class imbalance. Experiments on the Prostate-Anatomical-Edge-Cases dataset from The Cancer Imaging Archive demonstrate that the gated model substantially outperforms a non-gated segmentation-only baseline, achieving a mean Dice loss of $0.013 \pm 0.036$ versus $0.732 \pm 0.314$, with detection probabilities strongly correlated with anatomical presence, effectively eliminating spurious segmentations. In contrast, the non-gated model exhibited higher variability and persistent false positives across all slices. These results indicate that detection-based gating enhances robustness and anatomical plausibility in automated segmentation applications, reducing hallucinated predictions without compromising segmentation quality in valid slices, and offers a promising approach for improving the reliability of clinical radiotherapy auto-contouring workflows.

  </details>



- **RANKVIDEO: Reasoning Reranking for Text-to-Video Retrieval**  
  Tyler Skow, Alexander Martin, Benjamin Van Durme, Rama Chellappa, Reno Kriz  
  _2026-02-02_ · https://arxiv.org/abs/2602.02444v2  
  <details><summary>Abstract</summary>

  Reranking is a critical component of modern retrieval systems, which typically pair an efficient first-stage retriever with a more expressive model to refine results. While large reasoning models have driven rapid progress in text-centric reranking, reasoning-based reranking for video retrieval remains underexplored. To address this gap, we introduce RANKVIDEO, a reasoning-based reranker for video retrieval that explicitly reasons over query-video pairs using video content to assess relevance. RANKVIDEO is trained using a two-stage curriculum consisting of perception-grounded supervised fine-tuning followed by reranking training that combines pointwise, pairwise, and teacher confidence distillation objectives, and is supported by a data synthesis pipeline for constructing reasoning-intensive query-video pairs. Experiments on the large-scale MultiVENT 2.0 benchmark demonstrate that RANKVIDEO consistently improves retrieval performance within a two-stage framework, yielding an average improvement of 31% on nDCG@10 and outperforming text-only and vision-language reranking alternatives, while more efficient.

  </details>



- **UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing**  
  Dianyi Wang, Chaofan Ma, Feng Han, Size Wu, Wei Song, Yibin Wang, Zhixiong Zhang, Tianhang Wang, Siyuan Wang, Zhongyu Wei, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02437v1  
  <details><summary>Abstract</summary>

  Unified multimodal models often struggle with complex synthesis tasks that demand deep reasoning, and typically treat text-to-image generation and image editing as isolated capabilities rather than interconnected reasoning steps. To address this, we propose UniReason, a unified framework that harmonizes these two tasks through a dual reasoning paradigm. We formulate generation as world knowledge-enhanced planning to inject implicit constraints, and leverage editing capabilities for fine-grained visual refinement to further correct visual errors via self-reflection. This approach unifies generation and editing within a shared representation, mirroring the human cognitive process of planning followed by refinement. We support this framework by systematically constructing a large-scale reasoning-centric dataset (~300k samples) covering five major knowledge domains (e.g., cultural commonsense, physics, etc.) for planning, alongside an agent-generated corpus for visual self-correction. Extensive experiments demonstrate that UniReason achieves advanced performance on reasoning-intensive benchmarks such as WISE, KrisBench and UniREditBench, while maintaining superior general synthesis capabilities.

  </details>



- **SelvaMask: Segmenting Trees in Tropical Forests and Beyond**  
  Simon-Olivier Duguay, Hugo Baudchon, Etienne Laliberté, Helene Muller-Landau, Gonzalo Rivas-Torres, Arthur Ouaknine  
  _2026-02-02_ · https://arxiv.org/abs/2602.02426v1  
  <details><summary>Abstract</summary>

  Tropical forests harbor most of the planet's tree biodiversity and are critical to global ecological balance. Canopy trees in particular play a disproportionate role in carbon storage and functioning of these ecosystems. Studying canopy trees at scale requires accurate delineation of individual tree crowns, typically performed using high-resolution aerial imagery. Despite advances in transformer-based models for individual tree crown segmentation, performance remains low in most forests, especially tropical ones. To this end, we introduce SelvaMask, a new tropical dataset containing over 8,800 manually delineated tree crowns across three Neotropical forest sites in Panama, Brazil, and Ecuador. SelvaMask features comprehensive annotations, including an inter-annotator agreement evaluation, capturing the dense structure of tropical forests and highlighting the difficulty of the task. Leveraging this benchmark, we propose a modular detection-segmentation pipeline that adapts vision foundation models (VFMs), using domain-specific detection-prompter. Our approach reaches state-of-the-art performance, outperforming both zero-shot generalist models and fully supervised end-to-end methods in dense tropical forests. We validate these gains on external tropical and temperate datasets, demonstrating that SelvaMask serves as both a challenging benchmark and a key enabler for generalized forest monitoring. Our code and dataset will be released publicly.

  </details>



- **Infinite-World: Scaling Interactive World Models to 1000-Frame Horizons via Pose-Free Hierarchical Memory**  
  Ruiqi Wu, Xuanhua He, Meng Cheng, Tianyu Yang, Yong Zhang, Zhuoliang Kang, Xunliang Cai, Xiaoming Wei, Chunle Guo, Chongyi Li, et al.  
  _2026-02-02_ · https://arxiv.org/abs/2602.02393v2  
  <details><summary>Abstract</summary>

  We propose Infinite-World, a robust interactive world model capable of maintaining coherent visual memory over 1000+ frames in complex real-world environments. While existing world models can be efficiently optimized on synthetic data with perfect ground-truth, they lack an effective training paradigm for real-world videos due to noisy pose estimations and the scarcity of viewpoint revisits. To bridge this gap, we first introduce a Hierarchical Pose-free Memory Compressor (HPMC) that recursively distills historical latents into a fixed-budget representation. By jointly optimizing the compressor with the generative backbone, HPMC enables the model to autonomously anchor generations in the distant past with bounded computational cost, eliminating the need for explicit geometric priors. Second, we propose an Uncertainty-aware Action Labeling module that discretizes continuous motion into a tri-state logic. This strategy maximizes the utilization of raw video data while shielding the deterministic action space from being corrupted by noisy trajectories, ensuring robust action-response learning. Furthermore, guided by insights from a pilot toy study, we employ a Revisit-Dense Finetuning Strategy using a compact, 30-minute dataset to efficiently activate the model's long-range loop-closure capabilities. Extensive experiments, including objective metrics and user studies, demonstrate that Infinite-World achieves superior performance in visual quality, action controllability, and spatial consistency.

  </details>



- **Enhancing Indoor Occupancy Prediction via Sparse Query-Based Multi-Level Consistent Knowledge Distillation**  
  Xiang Li, Yupeng Zheng, Pengfei Li, Yilun Chen, Ya-Qin Zhang, Wenchao Ding  
  _2026-02-02_ · https://arxiv.org/abs/2602.02318v1  
  <details><summary>Abstract</summary>

  Occupancy prediction provides critical geometric and semantic understanding for robotics but faces efficiency-accuracy trade-offs. Current dense methods suffer computational waste on empty voxels, while sparse query-based approaches lack robustness in diverse and complex indoor scenes. In this paper, we propose DiScene, a novel sparse query-based framework that leverages multi-level distillation to achieve efficient and robust occupancy prediction. In particular, our method incorporates two key innovations: (1) a Multi-level Consistent Knowledge Distillation strategy, which transfers hierarchical representations from large teacher models to lightweight students through coordinated alignment across four levels, including encoder-level feature alignment, query-level feature matching, prior-level spatial guidance, and anchor-level high-confidence knowledge transfer and (2) a Teacher-Guided Initialization policy, employing optimized parameter warm-up to accelerate model convergence. Validated on the Occ-Scannet benchmark, DiScene achieves 23.2 FPS without depth priors while outperforming our baseline method, OPUS, by 36.1% and even better than the depth-enhanced version, OPUS†. With depth integration, DiScene† attains new SOTA performance, surpassing EmbodiedOcc by 3.7% with 1.62$\times$ faster inference speed. Furthermore, experiments on the Occ3D-nuScenes benchmark and in-the-wild scenarios demonstrate the versatility of our approach in various environments. Code and models can be accessed at https://github.com/getterupper/DiScene.

  </details>


