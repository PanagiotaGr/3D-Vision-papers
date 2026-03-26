# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-26 07:34 UTC_

Total papers shown: **50**


---

- **Vision-Language Models vs Human: Perceptual Image Quality Assessment**  
  Imran Mehmood, Imad Ali Shah, Ming Ronnier Luo, Brian Deegan  
  _2026-03-25_ · https://arxiv.org/abs/2603.24578v1  
  <details><summary>Abstract</summary>

  Psychophysical experiments remain the most reliable approach for perceptual image quality assessment (IQA), yet their cost and limited scalability encourage automated approaches. We investigate whether Vision Language Models (VLMs) can approximate human perceptual judgments across three image quality scales: contrast, colorfulness and overall preference. Six VLMs four proprietary and two openweight models are benchmarked against psychophysical data. This work presents a systematic benchmark of VLMs for perceptual IQA through comparison with human psychophysical data. The results reveal strong attribute dependent variability models with high human alignment for colorfulness (ρup to 0.93) underperform on contrast and vice-versa. Attribute weighting analysis further shows that most VLMs assign higher weights to colorfulness compared to contrast when evaluating overall preference similar to the psychophysical data. Intramodel consistency analysis reveals a counterintuitive tradeoff: the most self consistent models are not necessarily the most human aligned suggesting response variability reflects sensitivity to scene dependent perceptual cues. Furthermore, human-VLM agreement is increased with perceptual separability, indicating VLMs are more reliable when stimulus differences are clearly expressed.

  </details>



- **EndoVGGT: GNN-Enhanced Depth Estimation for Surgical 3D Reconstruction**  
  Falong Fan, Yi Xie, Arnis Lektauers, Bo Liu, Jerzy Rozenblit  
  _2026-03-25_ · https://arxiv.org/abs/2603.24577v1  
  <details><summary>Abstract</summary>

  Accurate 3D reconstruction of deformable soft tissues is essential for surgical robotic perception. However, low-texture surfaces, specular highlights, and instrument occlusions often fragment geometric continuity, posing a challenge for existing fixed-topology approaches. To address this, we propose EndoVGGT, a geometry-centric framework equipped with a Deformation-aware Graph Attention (DeGAT) module. Rather than using static spatial neighborhoods, DeGAT dynamically constructs feature-space semantic graphs to capture long-range correlations among coherent tissue regions. This enables robust propagation of structural cues across occlusions, enforcing global consistency and improving non-rigid deformation recovery. Extensive experiments on SCARED show that our method significantly improves fidelity, increasing PSNR by 24.6% and SSIM by 9.1% over prior state-of-the-art. Crucially, EndoVGGT exhibits strong zero-shot cross-dataset generalization to the unseen SCARED and EndoNeRF domains, confirming that DeGAT learns domain-agnostic geometric priors. These results highlight the efficacy of dynamic feature-space modeling for consistent surgical 3D reconstruction.

  </details>



- **Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation**  
  Xinying Guo, Chenxi Jiang, Hyun Bin Kim, Ying Sun, Yang Xiao, Yuhang Han, Jianfei Yang  
  _2026-03-25_ · https://arxiv.org/abs/2603.24576v1  
  <details><summary>Abstract</summary>

  Robotic manipulation often requires memory: occlusion and state changes can make decision-time observations perceptually aliased, making action selection non-Markovian at the observation level because the same observation may arise from different interaction histories. Most embodied agents implement memory via semantically compressed traces and similarity-based retrieval, which discards disambiguating fine-grained perceptual cues and can return perceptually similar but decision-irrelevant episodes. Inspired by human episodic memory, we propose Chameleon, which writes geometry-grounded multimodal tokens to preserve disambiguating context and produces goal-directed recall through a differentiable memory stack. We also introduce Camo-Dataset, a real-robot UR5e dataset spanning episodic recall, spatial tracking, and sequential manipulation under perceptual aliasing. Across tasks, Chameleon consistently improves decision reliability and long-horizon control over strong baselines in perceptually confusable settings.

  </details>



- **VFIG: Vectorizing Complex Figures in SVG with Vision-Language Models**  
  Qijia He, Xunmei Liu, Hammaad Memon, Ziang Li, Zixian Ma, Jaemin Cho, Jason Ren, Daniel S Weld, Ranjay Krishna  
  _2026-03-25_ · https://arxiv.org/abs/2603.24575v1  
  <details><summary>Abstract</summary>

  Scalable Vector Graphics (SVG) are an essential format for technical illustration and digital design, offering precise resolution independence and flexible semantic editability. In practice, however, original vector source files are frequently lost or inaccessible, leaving only "flat" rasterized versions (e.g., PNG or JPEG) that are difficult to modify or scale. Manually reconstructing these figures is a prohibitively labor-intensive process, requiring specialized expertise to recover the original geometric intent. To bridge this gap, we propose VFIG, a family of Vision-Language Models trained for complex and high-fidelity figure-to-SVG conversion. While this task is inherently data-driven, existing datasets are typically small-scale and lack the complexity of professional diagrams. We address this by introducing VFIG-DATA, a large-scale dataset of 66K high-quality figure-SVG pairs, curated from a diverse mix of real-world paper figures and procedurally generated diagrams. Recognizing that SVGs are composed of recurring primitives and hierarchical local structures, we introduce a coarse-to-fine training curriculum that begins with supervised fine-tuning (SFT) to learn atomic primitives and transitions to reinforcement learning (RL) refinement to optimize global diagram fidelity, layout consistency, and topological edge cases. Finally, we introduce VFIG-BENCH, a comprehensive evaluation suite with novel metrics designed to measure the structural integrity of complex figures. VFIG achieves state-of-the-art performance among open-source models and performs on par with GPT-5.2, achieving a VLM-Judge score of 0.829 on VFIG-BENCH.

  </details>



- **POLY-SIM: Polyglot Speaker Identification with Missing Modality Grand Challenge 2026 Evaluation Plan**  
  Marta Moscati, Muhammad Saad Saeed, Marina Zanoni, Mubashir Noman, Rohan Kumar Das, Monorama Swain, Yufang Hou, Elisabeth Andre, Khalid Mahmood Malik, Markus Schedl, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.24569v1  
  <details><summary>Abstract</summary>

  Multimodal speaker identification systems typically assume the availability of complete and homogeneous audio-visual modalities during both training and testing. However, in real-world applications, such assumptions often do not hold. Visual information may be missing due to occlusions, camera failures, or privacy constraints, while multilingual speakers introduce additional complexity due to linguistic variability across languages. These challenges significantly affect the robustness and generalization of multimodal speaker identification systems. The POLY-SIM Grand Challenge 2026 aims to advance research in multimodal speaker identification under missing-modality and cross-lingual conditions. Specifically, the Grand Challenge encourages the development of robust methods that can effectively leverage incomplete multimodal inputs while maintaining strong performance across different languages. This report presents the design and organization of the POLY-SIM Grand Challenge 2026, including the dataset, task formulation, evaluation protocol, and baseline model. By providing a standardized benchmark and evaluation framework, the challenge aims to foster progress toward more robust and practical multimodal speaker identification systems.

  </details>



- **UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience**  
  Zichuan Lin, Feiyu Liu, Yijun Yang, Jiafei Lyu, Yiming Gao, Yicheng Liu, Zhicong Lu, Yangbin Yu, Mingyu Yang, Junyou Li, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.24533v1  
  <details><summary>Abstract</summary>

  Autonomous mobile GUI agents have attracted increasing attention along with the advancement of Multimodal Large Language Models (MLLMs). However, existing methods still suffer from inefficient learning from failed trajectories and ambiguous credit assignment under sparse rewards for long-horizon GUI tasks. To that end, we propose UI-Voyager, a novel two-stage self-evolving mobile GUI agent. In the first stage, we employ Rejection Fine-Tuning (RFT), which enables the continuous co-evolution of data and models in a fully autonomous loop. The second stage introduces Group Relative Self-Distillation (GRSD), which identifies critical fork points in group rollouts and constructs dense step-level supervision from successful trajectories to correct failed ones. Extensive experiments on AndroidWorld show that our 4B model achieves an 81.0% Pass@1 success rate, outperforming numerous recent baselines and exceeding human-level performance. Ablation and case studies further verify the effectiveness of GRSD. Our method represents a significant leap toward efficient, self-evolving, and high-performance mobile GUI automation without expensive manual data annotation.

  </details>



- **Toward Physically Consistent Driving Video World Models under Challenging Trajectories**  
  Jiawei Zhou, Zhenxin Zhu, Lingyi Du, Linye Lyu, Lijun Zhou, Zhanqian Wu, Hongcheng Luo, Zhuotao Tian, Bing Wang, Guang Chen, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.24506v1  
  <details><summary>Abstract</summary>

  Video generation models have shown strong potential as world models for autonomous driving simulation. However, existing approaches are primarily trained on real-world driving datasets, which mostly contain natural and safe driving scenarios. As a result, current models often fail when conditioned on challenging or counterfactual trajectories-such as imperfect trajectories generated by simulators or planning systems-producing videos with severe physical inconsistencies and artifacts. To address this limitation, we propose PhyGenesis, a world model designed to generate driving videos with high visual fidelity and strong physical consistency. Our framework consists of two key components: (1) a physical condition generator that transforms potentially invalid trajectory inputs into physically plausible conditions, and (2) a physics-enhanced video generator that produces high-fidelity multi-view driving videos under these conditions. To effectively train these components, we construct a large-scale, physics-rich heterogeneous dataset. Specifically, in addition to real-world driving videos, we generate diverse challenging driving scenarios using the CARLA simulator, from which we derive supervision signals that guide the model to learn physically grounded dynamics under extreme conditions. This challenging-trajectory learning strategy enables trajectory correction and promotes physically consistent video generation. Extensive experiments demonstrate that PhyGenesis consistently outperforms state-of-the-art methods, especially on challenging trajectories. Our project page is available at: https://wm-research.github.io/PhyGenesis/.

  </details>



- **Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models**  
  Siqi Liu, Xinyang Li, Bochao Zou, Junbao Zhuo, Huimin Ma, Jiansheng Chen  
  _2026-03-25_ · https://arxiv.org/abs/2603.24484v1  
  <details><summary>Abstract</summary>

  As large language models (LLMs) continue to advance, there is increasing interest in their ability to infer human mental states and demonstrate a human-like Theory of Mind (ToM). Most existing ToM evaluations, however, are centered on text-based inputs, while scenarios relying solely on visual information receive far less attention. This leaves a gap, since real-world human-AI interaction typically requires multimodal understanding. In addition, many current methods regard the model as a black box and rarely probe how its internal attention behaves in multiple-choice question answering (QA). The impact of LLM hallucinations on such tasks is also underexplored from an interpretability perspective. To address these issues, we introduce VisionToM, a vision-oriented intervention framework designed to strengthen task-aware reasoning. The core idea is to compute intervention vectors that align visual representations with the correct semantic targets, thereby steering the model's attention through different layers of visual features. This guidance reduces the model's reliance on spurious linguistic priors, leading to more reliable multimodal language model (MLLM) outputs and better QA performance. Experiments on the EgoToM benchmark-an egocentric, real-world video dataset for ToM with three multiple-choice QA settings-demonstrate that our method substantially improves the ToM abilities of MLLMs. Furthermore, results on an additional open-ended generation task show that VisionToM enables MLLMs to produce free-form explanations that more accurately capture agents' mental states, pushing machine-human collaboration toward greater alignment.

  </details>



- **Positive-First Most Ambiguous: A Simple Active Learning Criterion for Interactive Retrieval of Rare Categories**  
  Kawtar Zaher, Olivier Buisson, Alexis Joly  
  _2026-03-25_ · https://arxiv.org/abs/2603.24480v1  
  <details><summary>Abstract</summary>

  Real-world fine-grained visual retrieval often requires discovering a rare concept from large unlabeled collections with minimal supervision. This is especially critical in biodiversity monitoring, ecological studies, and long-tailed visual domains, where the target may represent only a tiny fraction of the data, creating highly imbalanced binary problems. Interactive retrieval with relevance feedback offers a practical solution: starting from a small query, the system selects candidates for binary user annotation and iteratively refines a lightweight classifier. While Active Learning (AL) is commonly used to guide selection, conventional AL assumes symmetric class priors and large annotation budgets, limiting effectiveness in imbalanced, low-budget, low-latency settings. We introduce Positive-First Most Ambiguous (PF-MA), a simple yet effective AL criterion that explicitly addresses the class imbalance asymmetry: it prioritizes near-boundary samples while favoring likely positives, enabling rapid discovery of subtle visual categories while maintaining informativeness. Unlike standard methods that oversample negatives, PF-MA consistently returns small batches with a high proportion of relevant samples, improving early retrieval and user satisfaction. To capture retrieval diversity, we also propose a class coverage metric that measures how well selected positives span the visual variability of the target class. Experiments on long-tailed datasets, including fine-grained botanical data, demonstrate that PF-MA consistently outperforms strong baselines in both coverage and classifier performance, across varying class sizes and descriptors. Our results highlight that aligning AL with the asymmetric and user-centric objectives of interactive fine-grained retrieval enables simple yet powerful solutions for retrieving rare and visually subtle categories in realistic human-in-the-loop settings.

  </details>



- **OmniWeaving: Towards Unified Video Generation with Free-form Composition and Reasoning**  
  Kaihang Pan, Qi Tian, Jianwei Zhang, Weijie Kong, Jiangfeng Xiong, Yanxin Long, Shixue Zhang, Haiyi Qiu, Tan Wang, Zheqi Lv, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.24458v1  
  <details><summary>Abstract</summary>

  While proprietary systems such as Seedance-2.0 have achieved remarkable success in omni-capable video generation, open-source alternatives significantly lag behind. Most academic models remain heavily fragmented, and the few existing efforts toward unified video generation still struggle to seamlessly integrate diverse tasks within a single framework. To bridge this gap, we propose OmniWeaving, an omni-level video generation model featuring powerful multimodal composition and reasoning-informed capabilities. By leveraging a massive-scale pretraining dataset that encompasses diverse compositional and reasoning-augmented scenarios, OmniWeaving learns to temporally bind interleaved text, multi-image, and video inputs while acting as an intelligent agent to infer complex user intentions for sophisticated video creation. Furthermore, we introduce IntelligentVBench, the first comprehensive benchmark designed to rigorously assess next-level intelligent unified video generation. Extensive experiments demonstrate that OmniWeaving achieves SoTA performance among open-source unified models. The codes and model will be made publicly available soon. Project Page: https://omniweaving.github.io.

  </details>



- **CUA-Suite: Massive Human-annotated Video Demonstrations for Computer-Use Agents**  
  Xiangru Jian, Shravan Nayak, Kevin Qinghong Lin, Aarash Feizi, Kaixin Li, Patrice Bechard, Spandana Gella, Sai Rajeswar  
  _2026-03-25_ · https://arxiv.org/abs/2603.24440v1  
  <details><summary>Abstract</summary>

  Computer-use agents (CUAs) hold great promise for automating complex desktop workflows, yet progress toward general-purpose agents is bottlenecked by the scarcity of continuous, high-quality human demonstration videos. Recent work emphasizes that continuous video, not sparse screenshots, is the critical missing ingredient for scaling these agents. However, the largest existing open dataset, ScaleCUA, contains only 2 million screenshots, equating to less than 20 hours of video. To address this bottleneck, we introduce CUA-Suite, a large-scale ecosystem of expert video demonstrations and dense annotations for professional desktop computer-use agents. At its core is VideoCUA, which provides approximately 10,000 human-demonstrated tasks across 87 diverse applications with continuous 30 fps screen recordings, kinematic cursor traces, and multi-layerfed reasoning annotations, totaling approximately 55 hours and 6 million frames of expert video. Unlike sparse datasets that capture only final click coordinates, these continuous video streams preserve the full temporal dynamics of human interaction, forming a superset of information that can be losslessly transformed into the formats required by existing agent frameworks. CUA-Suite further provides two complementary resources: UI-Vision, a rigorous benchmark for evaluating grounding and planning capabilities in CUAs, and GroundCUA, a large-scale grounding dataset with 56K annotated screenshots and over 3.6 million UI element annotations. Preliminary evaluation reveals that current foundation action models struggle substantially with professional desktop applications (~60% task failure rate). Beyond evaluation, CUA-Suite's rich multimodal corpus supports emerging research directions including generalist screen parsing, continuous spatial control, video-based reward modeling, and visual world models. All data and models are publicly released.

  </details>



- **The Gait Signature of Frailty: Transfer Learning based Deep Gait Models for Scalable Frailty Assessment**  
  Laura McDaniel, Basudha Pal, Crystal Szczesny, Yuxiang Guo, Ryan Roemmich, Peter Abadir, Rama Chellappa  
  _2026-03-25_ · https://arxiv.org/abs/2603.24434v1  
  <details><summary>Abstract</summary>

  Frailty is a condition in aging medicine characterized by diminished physiological reserve and increased vulnerability to stressors. However, frailty assessment remains subjective, heterogeneous, and difficult to scale in clinical practice. Gait is a sensitive marker of biological aging, capturing multisystem decline before overt disability. Yet the application of modern computer vision to gait-based frailty assessment has been limited by small, imbalanced datasets and a lack of clinically representative benchmarks. In this work, we introduce a publicly available silhouette-based frailty gait dataset collected in a clinically realistic setting, spanning the full frailty spectrum and including older adults who use walking aids. Using this dataset, we evaluate how pretrained gait recognition models can be adapted for frailty classification under limited data conditions. We study both convolutional and hybrid attention-based architectures and show that predictive performance depends primarily on how pretrained representations are transferred rather than architectural complexity alone. Across models, selectively freezing low-level gait representations while allowing higher-level features to adapt yields more stable and generalizable performance than either full fine-tuning or rigid freezing. Conservative handling of class imbalance further improves training stability, and combining complementary learning objectives enhances discrimination between clinically adjacent frailty states. Interpretability analyses reveal consistent model attention to lower-limb and pelvic regions, aligning with established biomechanical correlates of frailty. Together, these findings establish gait-based representation learning as a scalable, non-invasive, and interpretable framework for frailty assessment and support the integration of modern biometric modeling approaches into aging research and clinical practice.

  </details>



- **Enhancing Drone Light Shows Performances: Optimal Allocation and Trajectories for Swarm Drone Formations**  
  Yunes Alqudsi  
  _2026-03-25_ · https://arxiv.org/abs/2603.24401v1  
  <details><summary>Abstract</summary>

  Drone light shows (DLShows) represent a rapidly growing application of swarm robotics, creating captivating aerial displays through the synchronized flight of hundreds or thousands of unmanned aerial vehicles (UAVs) as environmentally friendly and reusable alternatives to traditional pyrotechnics. This domain presents unique challenges in optimally assigning drones to visual waypoints and generating smooth, collision-free trajectories at a very large scale. This article introduces the Unified Assignment and Trajectory Generation (UATG) framework. The proposed approach concurrently solves two core problems: the optimal assignment of drones to designated goal locations and the generation of dynamically feasible, collision-free, time-parameterized trajectories. The UATG framework is specifically designed for DLShows, ensuring minimal transition times between formations and guaranteeing inter-drone collision avoidance. A key innovation is its exceptional computational efficiency, enabling the coordination of large-scale in real-time; for instance, it computes the optimal assignment and trajectories for 1008 drones in approximately one second on a standard laptop. Extensive simulations in realistic environments validate the framework's performance, demonstrating its capability to orchestrate complex formations, from alphanumeric characters to intricate 3D shapes, with precision and visual smoothness. This work provides a critical advancement for the DLShow industry, offering a practical and scalable solution for generating complex aerial choreography and establishing a valuable benchmark for ground control station software designed for the efficient coordination of multiple UAVs. A supplemental animated simulation of this work is available at https://youtu.be/-Fjrhw03594.

  </details>



- **3D-Mix for VLA: A Plug-and-Play Module for Integrating VGGT-based 3D Information into Vision-Language-Action Models**  
  Bin Yu, Shijie Lian, Xiaopeng Lin, Zhaolong Shen, Yuliang Wei, Haishan Liu, Changti Wu, Hang Yuan, Bailing Wang, Cong Huang, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.24393v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models leverage Multimodal Large Language Models (MLLMs) for robotic control, but recent studies reveal that MLLMs exhibit limited spatial intelligence due to training predominantly on 2D data, resulting in inadequate 3D perception for manipulation tasks. While recent approaches incorporate specialized 3D vision models such as VGGT to enhance spatial understanding, they employ diverse integration mechanisms without systematic investigation, leaving the optimal fusion strategy unclear. We conduct a comprehensive pilot study comparing nine VGGT integration schemes on standardized benchmarks and find that semantic-conditioned gated fusion, which adaptively balances 2D semantic and 3D geometric features based on task context, achieved the strongest performance among all nine evaluated fusion schemes in our pilot study. We present 3D-Mix, a plug-and-play module that integrates into diverse VLA architectures (GR00T-style and $π$-style) without modifying existing MLLM or action expert components. Experiments across six MLLM series (nine model variants, 2B--8B parameters) on SIMPLER and LIBERO show that 3D-Mix delivers consistent performance gains, averaging +7.0% on the out-of-domain (OOD) SIMPLER benchmark across all nine GR00T-style variants, establishing a principled approach for enhancing spatial intelligence in VLA systems.

  </details>



- **ViHOI: Human-Object Interaction Synthesis with Visual Priors**  
  Songjin Cai, Linjie Zhong, Ling Guo, Changxing Ding  
  _2026-03-25_ · https://arxiv.org/abs/2603.24383v1  
  <details><summary>Abstract</summary>

  Generating realistic and physically plausible 3D Human-Object Interactions (HOI) remains a key challenge in motion generation. One primary reason is that describing these physical constraints with words alone is difficult. To address this limitation, we propose a new paradigm: extracting rich interaction priors from easily accessible 2D images. Specifically, we introduce ViHOI, a novel framework that enables diffusion-based generative models to leverage rich, task-specific priors from 2D images to enhance generation quality. We utilize a large Vision-Language Model (VLM) as a powerful prior-extraction engine and adopt a layer-decoupled strategy to obtain visual and textual priors. Concurrently, we design a Q-Former-based adapter that compresses the VLM's high-dimensional features into compact prior tokens, which significantly facilitates the conditional training of our diffusion model. Our framework is trained on motion-rendered images from the dataset to ensure strict semantic alignment between visual inputs and motion sequences. During inference, it leverages reference images synthesized by a text-to-image generation model to improve generalization to unseen objects and interaction categories. Experimental results demonstrate that ViHOI achieves state-of-the-art performance, outperforming existing methods across multiple benchmarks and demonstrating superior generalization.

  </details>



- **GeoRouter: Dynamic Paradigm Routing for Worldwide Image Geolocalization**  
  Pengyue Jia, Derong Xu, Yingyi Zhang, Xiaopeng Li, Wenlin Zhang, Yi Wen, Yuanshao Zhu, Xiangyu Zhao  
  _2026-03-25_ · https://arxiv.org/abs/2603.24376v1  
  <details><summary>Abstract</summary>

  Worldwide image geolocalization aims to predict precise GPS coordinates for images captured anywhere on Earth, which is challenging due to the large visual and geographic diversity. Recent methods mainly follow two paradigms: retrieval-based approaches that match queries against a reference database, and generation-based approaches that directly predict coordinates using Large Vision-Language Models (LVLMs). However, we observe distinct error profiles between them: retrieval excels at fine-grained instance matching, while generation offers robust semantic reasoning. This complementary heterogeneity suggests that no single paradigm is universally superior. To harness this potential, we propose GeoRouter, a dynamic routing framework that adaptively assigns each query to the optimal paradigm. GeoRouter leverages an LVLM backbone to analyze visual content and provide routing decisions. To optimize GeoRouter, we introduce a distance-aware preference objective that converts the distance gap between paradigms into a continuous supervision signal, explicitly reflecting relative performance differences. Furthermore, we construct GeoRouting, the first large-scale dataset tailored for training routing policies with independent paradigm predictions. Extensive experiments on IM2GPS3k and YFCC4k demonstrate that GeoRouter significantly outperforms state-of-the-art baselines.

  </details>



- **Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens**  
  Ciem Cornelissen, Sam Leroux, Pieter Simoens  
  _2026-03-25_ · https://arxiv.org/abs/2603.24327v1  
  <details><summary>Abstract</summary>

  Self-supervised learning has emerged as a powerful paradigm for learning visual representations without manual annotations, yet most methods still operate on a single modality and therefore miss the complementary structure available from heterogeneous sensors. We present Le MuMo JEPA, a self-supervised framework that learns unified representations from RGB images and aligned companion modalities. In our driving experiments, the second modality is camera-aligned LiDAR depth; we also evaluate RGB-thermal training and transfer on the Teledyne FLIR ADAS benchmark. Our approach extends LeJEPA to the multi-modal setting by learning fusion tokens that act as a latent bottleneck between modality-specific patch stems inside a shared transformer. Our default model employs a pruned fusion strategy: after an initial cross-modal attention layer, modality-specific tokens are dropped, forcing cross-modal information into the shared fusion-token grid as an efficient latent bottleneck before Sketched Isotropic Gaussian Regularization (SIGReg) is applied to the joint multimodal CLS embedding. On Waymo, Le MuMo JEPA gives the strongest performance-efficiency trade-off on downstream patch probes among the from-scratch multimodal baselines, improving CenterNet detection and dense depth while remaining competitive on segmentation. Under from-scratch training on nuScenes, Le MuMo JEPA remains the strongest model, and it also gives the best FLIR results, especially after Waymo-initialized fine-tuning. It also retains the best overall accuracy-efficiency balance in our study at substantially lower compute, memory, and estimated training time.

  </details>



- **Refining time-space traffic diagrams: A neighborhood-adaptive linear regression method**  
  Zhihong Yao, Yi Yu, Yunxia Wu, Hao Li, Yangsheng Jiang, Zhengbing He  
  _2026-03-25_ · https://arxiv.org/abs/2603.24312v1  
  <details><summary>Abstract</summary>

  The time-space (TS) traffic diagram serves as a crucial tool for characterizing the dynamic evolution of traffic flow, with its resolution directly influencing the effectiveness of traffic theory research and engineering applications. However, constrained by monitoring precision and sampling frequency, existing TS traffic diagrams commonly suffer from low resolution. To address this issue, this paper proposes a refinement method for TS traffic diagrams based on neighborhood-adaptive linear regression. Introducing the concept of neighborhood embedding into TS diagram refinement, the method leverages local pattern similarity in TS diagrams, adaptively identifies neighborhoods similar to target cells, and fits the low-to-high resolution mapping within these neighborhoods for refinement. It avoids the over-smoothing tendency of the traditional global linear model, allows the capture of unique traffic wave propagation and congestion evolution characteristics, and outperforms the traditional neighborhood embedding method in terms of local information utilization to achieve target cell refinement. Validation on two real datasets across multiple scales and upscaling factors shows that, compared to benchmark methods, the proposed method achieves improvements of 9.16%, 8.16%, 1.86%, 3.89%, and 5.83% in metrics including MAE, MAPE, CMJS, SSIM, and GMSD, respectively. Furthermore, the proposed method exhibits strong generalization and robustness in cross-day and cross-scenario validations. In summary, requiring only a minimal amount of paired high- and low-resolution training data, the proposed method features a concise formulation, providing a foundation for the low-cost, fine-grained refinement of low-sampling-rate traffic data.

  </details>



- **ScrollScape: Unlocking 32K Image Generation With Video Diffusion Priors**  
  Haodong Yu, Yabo Zhang, Donglin Di, Ruyi Zhang, Wangmeng Zuo  
  _2026-03-25_ · https://arxiv.org/abs/2603.24270v1  
  <details><summary>Abstract</summary>

  While diffusion models excel at generating images with conventional dimensions, pushing them to synthesize ultra-high-resolution imagery at extreme aspect ratios (EAR) often triggers catastrophic structural failures, such as object repetition and spatial fragmentation.This limitation fundamentally stems from a lack of robust spatial priors, as static text-to-image models are primarily trained on image distributions with conventional dimensions.To overcome this bottleneck, we present ScrollScape, a novel framework that reformulates EAR image synthesis into a continuous video generation process through two core innovations.By mapping the spatial expansion of a massive canvas to the temporal evolution of video frames, ScrollScape leverages the inherent temporal consistency of video models as a powerful global constraint to ensure long-range structural integrity.Specifically, Scanning Positional Encoding (ScanPE) distributes global coordinates across frames to act as a flexible moving camera, while Scrolling Super-Resolution (ScrollSR) leverages video super-resolution priors to circumvent memory bottlenecks, efficiently scaling outputs to an unprecedented 32K resolution. Fine-tuned on a curated 3K multi-ratio image dataset, ScrollScape effectively aligns pre-trained video priors with the EAR generation task. Extensive evaluations demonstrate that it significantly outperforms existing image-diffusion baselines by eliminating severe localized artifacts. Consequently, our method overcomes inherent structural bottlenecks to ensure exceptional global coherence and visual fidelity across diverse domains at extreme scales.

  </details>



- **Memory-Augmented Vision-Language Agents for Persistent and Semantically Consistent Object Captioning**  
  Tommaso Galliena, Stefano Rosa, Tommaso Apicella, Pietro Morerio, Alessio Del Bue, Lorenzo Natale  
  _2026-03-25_ · https://arxiv.org/abs/2603.24257v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) often yield inconsistent descriptions of the same object across viewpoints, hindering the ability of embodied agents to construct consistent semantic representations over time. Previous methods resolved inconsistencies using offline multi-view aggregation or multi-stage pipelines that decouple exploration, data association, and caption learning, with limited capacity to reason over previously observed objects. In this paper, we introduce a unified, memory-augmented Vision-Language agent that simultaneously handles data association, object captioning, and exploration policy within a single autoregressive framework. The model processes the current RGB observation, a top-down explored map, and an object-level episodic memory serialized into object-level tokens, ensuring persistent object identity and semantic consistency across extended sequences. To train the model in a self-supervised manner, we collect a dataset in photorealistic 3D environments using a disagreement-based policy and a pseudo-captioning model that enforces consistency across multi-view caption histories. Extensive evaluation on a manually annotated object-level test set, demonstrate improvements of up to +11.86% in standard captioning scores and +7.39% in caption self-similarity over baseline models, while enabling scalable performance through a compact scene representation. Code, model weights, and data are available at https://github.com/hsp-iit/epos-vlm

  </details>



- **RefReward-SR: LR-Conditioned Reward Modeling for Preference-Aligned Super-Resolution**  
  Yushuai Song, Weize Quan, Weining Wang, Jiahui Sun, Jing Liu, Meng Li, Pengbin Yu, Zhentao Chen, Wei Shen, Lunxi Yuan, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.24198v1  
  <details><summary>Abstract</summary>

  Recent advances in generative super-resolution (SR) have greatly improved visual realism, yet existing evaluation and optimization frameworks remain misaligned with human perception. Full-Reference and No-Reference metrics often fail to reflect perceptual preference, either penalizing semantically plausible details due to pixel misalignment or favoring visually sharp but inconsistent artifacts. Moreover, most SR methods rely on ground-truth (GT)-dependent distribution matching, which does not necessarily correspond to human judgments. In this work, we propose RefReward-SR, a low-resolution (LR) reference-aware reward model for preference-aligned SR. Instead of relying on GT supervision or NR evaluation, RefReward-SR assesses high-resolution (HR) reconstructions conditioned on their LR inputs, treating the LR image as a semantic anchor. Leveraging the visual-linguistic priors of a Multimodal Large Language Models (MLLM), it evaluates semantic consistency and plausibility in a reasoning-aware manner. To support this paradigm, we construct RefSR-18K, the first large-scale LR-conditioned preference dataset for SR, providing pairwise rankings based on LR-HR consistency and HR naturalness. We fine-tune the MLLM with Group Relative Policy Optimization (GRPO) using LR-conditioned ranking rewards, and further integrate GRPO into SR model training with RefReward-SR as the core reward signal for preference-aligned generation. Extensive experiments show that our framework achieves substantially better alignment with human judgments, producing reconstructions that preserve semantic consistency while enhancing perceptual plausibility and visual naturalness. Code, models, and datasets will be released upon paper acceptance.

  </details>



- **Modeling Spatiotemporal Neural Frames for High Resolution Brain Dynamic**  
  Wanying Qu, Jianxiong Gao, Wei Wang, Yanwei Fu  
  _2026-03-25_ · https://arxiv.org/abs/2603.24176v1  
  <details><summary>Abstract</summary>

  Capturing dynamic spatiotemporal neural activity is essential for understanding large-scale brain mechanisms. Functional magnetic resonance imaging (fMRI) provides high-resolution cortical representations that form a strong basis for characterizing fine-grained brain activity patterns. The high acquisition cost of fMRI limits large-scale applications, therefore making high-quality fMRI reconstruction a crucial task. Electroencephalography (EEG) offers millisecond-level temporal cues that complement fMRI. Leveraging this complementarity, we present an EEG-conditioned framework for reconstructing dynamic fMRI as continuous neural sequences with high spatial fidelity and strong temporal coherence at the cortical-vertex level. To address sampling irregularities common in real fMRI acquisitions, we incorporate a null-space intermediate-frame reconstruction, enabling measurement-consistent completion of arbitrary intermediate frames and improving sequence continuity and practical applicability. Experiments on the CineBrain dataset demonstrate superior voxel-wise reconstruction quality and robust temporal consistency across whole-brain and functionally specific regions. The reconstructed fMRI also preserves essential functional information, supporting downstream visual decoding tasks. This work provides a new pathway for estimating high-resolution fMRI dynamics from EEG and advances multimodal neuroimaging toward more dynamic brain activity modeling.

  </details>



- **Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection**  
  Xu Zhang, Zhe Chen, Jing Zhang, Dacheng Tao  
  _2026-03-25_ · https://arxiv.org/abs/2603.24166v1  
  <details><summary>Abstract</summary>

  Most referring object detection (ROD) models, especially the modern grounding detectors, are designed for data-rich conditions, yet many practical deployments, such as robotics, augmented reality, and other specialized domains, would face severe label scarcity. In such regimes, end-to-end grounding detectors need to learn spatial and semantic structure from scratch, wasting precious samples. We ask a simple question: Can explicit reasoning priors help models learn more efficiently when data is scarce? To explore this, we first introduce a Data-efficient Referring Object Detection (De-ROD) task, which is a benchmark protocol for measuring ROD performance in low-data and few-shot settings. We then propose the HeROD (Heuristic-inspired ROD), a lightweight, model-agnostic framework that injects explicit, heuristic-inspired spatial and semantic reasoning priors, which are interpretable signals derived based on the referring phrase, into 3 stages of a modern DETR-style pipeline: proposal ranking, prediction fusion, and Hungarian matching. By biasing both training and inference toward plausible candidates, these priors promise to improve label efficiency and convergence performance. On RefCOCO, RefCOCO+, and RefCOCOg, HeROD consistently outperforms strong grounding baselines in scarce-label regimes. More broadly, our results suggest that integrating simple, interpretable reasoning priors provides a practical and extensible path toward better data-efficient vision-language understanding.

  </details>



- **CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare**  
  Akash Ghosh, Tajamul Ashraf, Rishu Kumar Singh, Numan Saeed, Sriparna Saha, Xiuying Chen, Salman Khan  
  _2026-03-25_ · https://arxiv.org/abs/2603.24157v1  
  <details><summary>Abstract</summary>

  Multimodal agentic pipelines are transforming human-computer interaction by enabling efficient and accessible automation of complex, real-world tasks. However, recent efforts have focused on short-horizon or general-purpose applications (e.g., mobile or desktop interfaces), leaving long-horizon automation for domain-specific systems, particularly in healthcare, largely unexplored. To address this, we introduce CareFlow, a high-quality human-annotated benchmark comprising complex, long-horizon software workflows across medical annotation tools, DICOM viewers, EHR systems, and laboratory information systems. On this benchmark, existing vision-language models (VLMs) perform poorly, struggling with long-horizon reasoning and multi-step interactions in medical contexts. To overcome this, we propose CarePilot, a multi-agent framework based on the actor-critic paradigm. The Actor integrates tool grounding with dual-memory mechanisms (long-term and short-term experience) to predict the next semantic action from the visual interface and system state. The Critic evaluates each action, updates memory based on observed effects, and either executes or provides corrective feedback to refine the workflow. Through iterative agentic simulation, the Actor learns to perform more robust and reasoning-aware predictions during inference. Our experiments show that CarePilot achieves state-of-the-art performance, outperforming strong closed-source and open-source multimodal baselines by approximately 15.26% and 3.38%, respectively, on our benchmark and out-of-distribution dataset.

  </details>



- **Retinal Layer Segmentation in OCT Images With 2.5D Cross-slice Feature Fusion Module for Glaucoma Assessment**  
  Hyunwoo Kim, Heesuk Kim, Wungrak Choi, Jae-Sang Hyun  
  _2026-03-25_ · https://arxiv.org/abs/2603.24115v1  
  <details><summary>Abstract</summary>

  For accurate glaucoma diagnosis and monitoring, reliable retinal layer segmentation in OCT images is essential. However, existing 2D segmentation methods often suffer from slice-to-slice inconsistencies due to the lack of contextual information across adjacent B-scans. 3D segmentation methods are better for capturing slice-to-slice context, but they require expensive computational resources. To address these limitations, we propose a 2.5D segmentation framework that incorporates a novel cross-slice feature fusion (CFF) module into a U-Net-like architecture. The CFF module fuses inter-slice features to effectively capture contextual information, enabling consistent boundary detection across slices and improved robustness in noisy regions. The framework was validated on both a clinical dataset and the publicly available DUKE DME dataset. Compared to other segmentation methods without the CFF module, the proposed method achieved an 8.56% reduction in mean absolute distance and a 13.92% reduction in root mean square error, demonstrating improved segmentation accuracy and robustness. Overall, the proposed 2.5D framework balances contextual awareness and computational efficiency, enabling anatomically reliable retinal layer delineation for automated glaucoma evaluation and potential clinical applications.

  </details>



- **When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm**  
  Ye Leng, Junjie Chu, Mingjie Li, Chenhao Lin, Chao Shen, Michael Backes, Yun Shen, Yang Zhang  
  _2026-03-25_ · https://arxiv.org/abs/2603.24079v1  
  <details><summary>Abstract</summary>

  Recently, multimodal large language models (MLLMs) have emerged as a unified paradigm for language and image generation. Compared with diffusion models, MLLMs possess a much stronger capability for semantic understanding, enabling them to process more complex textual inputs and comprehend richer contextual meanings. However, this enhanced semantic ability may also introduce new and potentially greater safety risks. Taking diffusion models as a reference point, we systematically analyze and compare the safety risks of emerging MLLMs along two dimensions: unsafe content generation and fake image synthesis. Across multiple unsafe generation benchmark datasets, we observe that MLLMs tend to generate more unsafe images than diffusion models. This difference partly arises because diffusion models often fail to interpret abstract prompts, producing corrupted outputs, whereas MLLMs can comprehend these prompts and generate unsafe content. For current advanced fake image detectors, MLLM-generated images are also notably harder to identify. Even when detectors are retrained with MLLMs-specific data, they can still be bypassed by simply providing MLLMs with longer and more descriptive inputs. Our measurements indicate that the emerging safety risks of the cutting-edge generative paradigm, MLLMs, have not been sufficiently recognized, posing new challenges to real-world safety.

  </details>



- **PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation**  
  Yuheng Feng, Wen Zhang, Haodong Duan, Xingxing Zou  
  _2026-03-25_ · https://arxiv.org/abs/2603.24078v1  
  <details><summary>Abstract</summary>

  We present PosterIQ, a design-driven benchmark for poster understanding and generation, annotated across composition structure, typographic hierarchy, and semantic intent. It includes 7,765 image-annotation instances and 822 generation prompts spanning real, professional, and synthetic cases. To bridge visual design cognition and generative modeling, we define tasks for layout parsing, text-image correspondence, typography/readability and font perception, design quality assessment, and controllable, composition-aware generation with metaphor. We evaluate state-of-the-art MLLMs and diffusion-based generators, finding persistent gaps in visual hierarchy, typographic semantics, saliency control, and intention communication; commercial models lead on high-level reasoning but act as insensitive automatic raters, while generators render text well yet struggle with composition-aware synthesis. Extensive analyses show PosterIQ is both a quantitative benchmark and a diagnostic tool for design reasoning, offering reproducible, task-specific metrics. We aim to catalyze models' creativity and integrate human-centred design principles into generative vision-language systems.

  </details>



- **AD-Reasoning: Multimodal Guideline-Guided Reasoning for Alzheimer's Disease Diagnosis**  
  Qiuhui Chen, Yushan Deng, Xuancheng Yao, Yi Hong  
  _2026-03-25_ · https://arxiv.org/abs/2603.24059v1  
  <details><summary>Abstract</summary>

  Alzheimer's disease (AD) diagnosis requires integrating neuroimaging with heterogeneous clinical evidence and reasoning under established criteria, yet most multimodal models remain opaque and weakly guideline-aligned. We present AD-Reasoning, a multimodal framework that couples structural MRI with six clinical modalities and a rule-based verifier to generate structured, NIA-AA-consistent diagnoses. AD-Reasoning combines modality-specific encoders, bidirectional cross-attention fusion, and reinforcement fine-tuning with verifiable rewards that enforce output format, guideline evidence coverage, and reasoning--decision consistency. We also release AD-MultiSense, a 10,378-visit multimodal QA dataset with guideline-validated rationales built from ADNI/AIBL. On AD-MultiSense, AD-Reasoning achieves state-of-the-art diagnostic accuracy and produces structured rationales that improve transparency over recent baselines, while providing transparent rationales.

  </details>



- **LGEST: Dynamic Spatial-Spectral Expert Routing for Hyperspectral Image Classification**  
  Jiawen Wen, Suixuan Qiu, Zihang Luo, Xiaofei Yang, Haotian Shi  
  _2026-03-25_ · https://arxiv.org/abs/2603.24045v1  
  <details><summary>Abstract</summary>

  Deep learning methods, including Convolutional Neural Networks, Transformers and Mamba, have achieved remarkable success in hyperspectral image (HSI) classification. Nevertheless, existing methods exhibit inflexible integration of local-global representations, inadequate handling of spectral-spatial scale disparities across heterogeneous bands, and susceptibility to the Hughes phenomenon under high-dimensional sample heterogeneity. To address these challenges, we propose Local-Global Expert Spatial-Spectral Transformer (LGEST), a novel framework that synergistically combines three key innovations. The LGEST first employs a Deep Spatial-Spectral Autoencoder (DSAE) to generate compact yet discriminative embeddings through hierarchical nonlinear compression, preserving 3D neighborhood coherence while mitigating information loss in high-dimensional spaces. Secondly, a Cross-Interactive Mixed Expert Feature Pyramid (CIEM-FPN) leverages cross-attention mechanisms and residual mixture-of-experts layers to dynamically fuse multi-scale features, adaptively weighting spectral discriminability and spatial saliency through learnable gating functions. Finally, a Local-Global Expert System (LGES) processes decomposed features via sparsely activated expert pairs: convolutional sub-experts capture fine-grained textures, while transformer sub-experts model long-range contextual dependencies, with a routing controller dynamically selecting experts based on real-time feature saliency. Extensive experiments on four benchmark datasets demonstrate that LGEST consistently outperforms state-of-the-art methods.

  </details>



- **A^3: Towards Advertising Aesthetic Assessment**  
  Kaiyuan Ji, Yixuan Gao, Lu Sun, Yushuo Zheng, Zijian Chen, Jianbo Zhang, Xiangyang Zhu, Yuan Tian, Zicheng Zhang, Guangtao Zhai  
  _2026-03-25_ · https://arxiv.org/abs/2603.24037v1  
  <details><summary>Abstract</summary>

  Advertising images significantly impact commercial conversion rates and brand equity, yet current evaluation methods rely on subjective judgments, lacking scalability, standardized criteria, and interpretability. To address these challenges, we present A^3 (Advertising Aesthetic Assessment), a comprehensive framework encompassing four components: a paradigm (A^3-Law), a dataset (A^3-Dataset), a multimodal large language model (A^3-Align), and a benchmark (A^3-Bench). Central to A^3 is a theory-driven paradigm, A^3-Law, comprising three hierarchical stages: (1) Perceptual Attention, evaluating perceptual image signals for their ability to attract attention; (2) Formal Interest, assessing formal composition of image color and spatial layout in evoking interest; and (3) Desire Impact, measuring desire evocation from images and their persuasive impact. Building on A^3-Law, we construct A^3-Dataset with 120K instruction-response pairs from 30K advertising images, each richly annotated with multi-dimensional labels and Chain-of-Thought (CoT) rationales. We further develop A^3-Align, trained under A^3-Law with CoT-guided learning on A^3-Dataset. Extensive experiments on A^3-Bench demonstrate that A^3-Align achieves superior alignment with A^3-Law compared to existing models, and this alignment generalizes well to quality advertisement selection and prescriptive advertisement critique, indicating its potential for broader deployment. Dataset, code, and models can be found at: https://github.com/euleryuan/A3-Align.

  </details>



- **QuadFM: Foundational Text-Driven Quadruped Motion Dataset for Generation and Control**  
  Li Gao, Fuzhi Yang, Jianhui Chen, Liu Liu, Yao Zheng, Yang Cai, Ziqiao Li  
  _2026-03-25_ · https://arxiv.org/abs/2603.24021v1  
  <details><summary>Abstract</summary>

  Despite significant advances in quadrupedal robotics, a critical gap persists in foundational motion resources that holistically integrate diverse locomotion, emotionally expressive behaviors, and rich language semantics-essential for agile, intuitive human-robot interaction. Current quadruped motion datasets are limited to a few mocap primitives (e.g., walk, trot, sit) and lack diverse behaviors with rich language grounding. To bridge this gap, we introduce Quadruped Foundational Motion (QuadFM) , the first large-scale, ultra-high-fidelity dataset designed for text-to-motion generation and general motion control. QuadFM contains 11,784 curated motion clips spanning locomotion, interactive, and emotion-expressive behaviors (e.g., dancing, stretching, peeing), each with three-layer annotation-fine-grained action labels, interaction scenarios, and natural language commands-totaling 35,352 descriptions to support language-conditioned understanding and command execution. We further propose Gen2Control RL, a unified framework that jointly trains a general motion controller and a text-to-motion generator, enabling efficient end-to-end inference on edge hardware. On a real quadruped robot with an NVIDIA Orin, our system achieves real-time motion synthesis (<500 ms latency). Simulation and real-world results show realistic, diverse motions while maintaining robust physical interaction. The dataset will be released at https://github.com/GaoLii/QuadFM.

  </details>



- **COVTrack++: Learning Open-Vocabulary Multi-Object Tracking from Continuous Videos via a Synergistic Paradigm**  
  Zekun Qian, Wei Feng, Ruize Han, Junhui Hou  
  _2026-03-25_ · https://arxiv.org/abs/2603.24016v1  
  <details><summary>Abstract</summary>

  Multi-Object Tracking (MOT) has traditionally focused on a few specific categories, restricting its applicability to real-world scenarios involving diverse objects. Open-Vocabulary Multi-Object Tracking (OVMOT) addresses this by enabling tracking of arbitrary categories, including novel objects unseen during training. However, current progress is constrained by two challenges: the lack of continuously annotated video data for training, and the lack of a customized OVMOT framework to synergistically handle detection and association. We address the data bottleneck by constructing C-TAO, the first continuously annotated training set for OVMOT, which increases annotation density by 26x over the original TAO and captures smooth motion dynamics and intermediate object states. For the framework bottleneck, we propose COVTrack++, a synergistic framework that achieves a bidirectional reciprocal mechanism between detection and association through three modules: (1) Multi-Cue Adaptive Fusion (MCF) dynamically balances appearance, motion, and semantic cues for association feature learning; (2) Multi-Granularity Hierarchical Aggregation (MGA) exploits hierarchical spatial relationships in dense detections, where visible child nodes (e.g., object parts) assist occluded parent objects (e.g., whole body) for association feature enhancement; (3) Temporal Confidence Propagation (TCP) recovers flickering detections through high-confidence tracked objects boosting low-confidence candidates across frames, stabilizing trajectories. Extensive experiments on TAO demonstrate state-of-the-art performance, with novel TETA reaching 35.4% and 30.5% on validation and test sets, improving novel AssocA by 4.8% and novel LocA by 5.8% over previous methods, and show strong zero-shot generalization on BDD100K. The code and dataset will be publicly available.

  </details>



- **UW-VOS: A Large-Scale Dataset for Underwater Video Object Segmentation**  
  Hongshen Zhao, Jingkang Tai, Yuhang Wu, Wenkang Zhang, Xi Lan, Shangyan Wang, Tianyu Zhang, Wankou Yang  
  _2026-03-25_ · https://arxiv.org/abs/2603.24006v1  
  <details><summary>Abstract</summary>

  Underwater Video Object Segmentation (VOS) is essential for marine exploration, yet open-air methods suffer significant degradation due to color distortion, low contrast, and prevalent camouflage. A primary hurdle is the lack of high-quality training data. To bridge this gap, we introduce $\textbf{UW-VOS}$, the first large-scale underwater VOS benchmark comprising 1,431 video sequences across 409 categories with 309,295 mask annotations, constructed via a semi-automatic data engine with rigorous human verification. We further propose $\textbf{SAM-U}$, a parameter-efficient framework that adapts SAM2 to the underwater domain. By inserting lightweight adapters into the image encoder, SAM-U achieves state-of-the-art performance with only $\sim$2$\%$ trainable parameters. Extensive experiments reveal that existing methods experience an average 13-point $\mathcal{J}\&\mathcal{F}$ drop on UW-VOS, while SAM-U effectively bridges this domain gap. Detailed attribute-based analysis further identifies small targets, camouflage, and exit-re-entry as critical bottlenecks, providing a roadmap for future research in robust underwater perception.

  </details>



- **Leave No Stone Unturned: Uncovering Holistic Audio-Visual Intrinsic Coherence for Deepfake Detection**  
  Jielun Peng, Yabin Wang, Yaqi Li, Long Kong, Xiaopeng Hong  
  _2026-03-25_ · https://arxiv.org/abs/2603.23960v1  
  <details><summary>Abstract</summary>

  The rapid progress of generative AI has enabled hyper-realistic audio-visual deepfakes, intensifying threats to personal security and social trust. Most existing deepfake detectors rely either on uni-modal artifacts or audio-visual discrepancies, failing to jointly leverage both sources of information. Moreover, detectors that rely on generator-specific artifacts tend to exhibit degraded generalization when confronted with unseen forgeries. We argue that robust and generalizable detection should be grounded in intrinsic audio-visual coherence within and across modalities. Accordingly, we propose HAVIC, a Holistic Audio-Visual Intrinsic Coherence-based deepfake detector. HAVIC first learns priors of modality-specific structural coherence, inter-modal micro- and macro-coherence by pre-training on authentic videos. Based on the learned priors, HAVIC further performs holistic adaptive aggregation to dynamically fuse audio-visual features for deepfake detection. Additionally, we introduce HiFi-AVDF, a high-fidelity audio-visual deepfake dataset featuring both text-to-video and image-to-video forgeries from state-of-the-art commercial generators. Extensive experiments across several benchmarks demonstrate that HAVIC significantly outperforms existing state-of-the-art methods, achieving improvements of 9.39% AP and 9.37% AUC on the most challenging cross-dataset scenario. Our code and dataset are available at https://github.com/tuffy-studio/HAVIC.

  </details>



- **SynMVCrowd: A Large Synthetic Benchmark for Multi-view Crowd Counting and Localization**  
  Qi Zhang, Daijie Chen, Yunfei Gong, Hui Huang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23956v1  
  <details><summary>Abstract</summary>

  Existing multi-view crowd counting and localization methods are evaluated under relatively small scenes with limited crowd numbers, camera views, and frames. This makes the evaluation and comparison of existing methods impractical, as small datasets are easily overfit by these methods. To avoid these issues, 3DROM proposes a data augmentation method. Instead, in this paper, we propose a large synthetic benchmark, SynMVCrowd, for more practical evaluation and comparison of multi-view crowd counting and localization tasks. The SynMVCrowd benchmark consists of 50 synthetic scenes with a large number of multi-view frames and camera views and a much larger crowd number (up to 1000), which is more suitable for large-scene multi-view crowd vision tasks. Besides, we propose strong multi-view crowd localization and counting baselines that outperform all comparison methods on the new SynMVCrowd benchmark. Moreover, we prove that better domain transferring multi-view and single-image counting performance could be achieved with the aid of the benchmark on novel new real scenes. As a result, the proposed benchmark could advance the research for multi-view and single-image crowd counting and localization to more practical applications. The codes and datasets are here: https://github.com/zqyq/SynMVCrowd.

  </details>



- **Revealing Multi-View Hallucination in Large Vision-Language Models**  
  Wooje Park, Insu Lee, Soohyun Kim, Jaeyun Jang, Minyoung Noh, Kyuhong Shim, Byonghyo Shim  
  _2026-03-25_ · https://arxiv.org/abs/2603.23934v1  
  <details><summary>Abstract</summary>

  Large vision-language models (LVLMs) are increasingly being applied to multi-view image inputs captured from diverse viewpoints. However, despite this growing use, current LVLMs often confuse or mismatch visual information originating from different instances or viewpoints, a phenomenon we term multi-view hallucination. To systematically analyze this problem, we construct MVH-Bench, a benchmark comprising 4.8k question-answer pairs targeting two types of hallucination: cross-instance and cross-view. Empirical results show that recent LVLMs struggle to correctly associate visual evidence with its corresponding instance or viewpoint. To overcome this limitation, we propose Reference Shift Contrastive Decoding (RSCD), a training-free decoding technique that suppresses visual interference by generating negative logits through attention masking. Experiments on MVH-Bench with Qwen2.5-VL and LLaVA-OneVision demonstrate that RSCD consistently improves performance by up to 21.1 and 34.6 points over existing hallucination mitigation methods, highlighting the effectiveness of our approach.

  </details>



- **ORACLE: Orchestrate NPC Daily Activities using Contrastive Learning with Transformer-CVAE**  
  Seong-Eun Hong, JuYeong Hwang, RyunHa Lee, HyeongYeop Kang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23933v1  
  <details><summary>Abstract</summary>

  The integration of Non-player characters (NPCs) within digital environments has been increasingly recognized for its potential to augment user immersion and cognitive engagement. The sophisticated orchestration of their daily activities, reflecting the nuances of human daily routines, contributes significantly to the realism of digital environments. Nevertheless, conventional approaches often produce monotonous repetition, falling short of capturing the intricacies of real human activity plans. In response to this, we introduce ORACLE, a novel generative model for the synthesis of realistic indoor daily activity plans, ensuring NPCs' authentic presence in digital habitats. Exploiting the CASAS smart home dataset's 24-hour indoor activity sequences, ORACLE addresses challenges in the dataset, including its imbalanced sequential data, the scarcity of training samples, and the absence of pre-trained models encapsulating human daily activity patterns. ORACLE's training leverages the sequential data processing prowess of Transformers, the generative controllability of Conditional Variational Autoencoders (CVAE), and the discriminative refinement of contrastive learning. Our experimental results validate the superiority of generating NPC activity plans and the efficacy of our design strategies over existing methods.

  </details>



- **DP^2-VL: Private Photo Dataset Protection by Data Poisoning for Vision-Language Models**  
  Hongyi Miao, Jun Jia, Xincheng Wang, Qianli Ma, Wei Sun, Wangqiu Zhou, Dandan Zhu, Yewen Cao, Zhi Liu, Guangtao Zhai  
  _2026-03-25_ · https://arxiv.org/abs/2603.23925v1  
  <details><summary>Abstract</summary>

  Recent advances in visual-language alignment have endowed vision-language models (VLMs) with fine-grained image understanding capabilities. However, this progress also introduces new privacy risks. This paper first proposes a novel privacy threat model named identity-affiliation learning: an attacker fine-tunes a VLM using only a few private photos of a target individual, thereby embedding associations between the target facial identity and their private property and social relationships into the model's internal representations. Once deployed via public APIs, this model enables unauthorized exposure of the target user's private information upon input of their photos. To benchmark VLMs' susceptibility to such identity-affiliation leakage, we introduce the first identity-affiliation dataset comprising seven typical scenarios appearing in private photos. Each scenario is instantiated with multiple identity-centered photo-description pairs. Experimental results demonstrate that mainstream VLMs like LLaVA, Qwen-VL, and MiniGPT-v2, can recognize facial identities and infer identity-affiliation relationships by fine-tuning on small-scale private photographic dataset, and even on synthetically generated datasets. To mitigate this privacy risk, we propose DP2-VL, the first Dataset Protection framework for private photos that leverages Data Poisoning. Though optimizing imperceptible perturbations by pushing the original representations toward an antithetical region, DP2-VL induces a dataset-level shift in the embedding space of VLMs'encoders. This shift separates protected images from clean inference images, causing fine-tuning on the protected set to overfit. Extensive experiments demonstrate that DP2-VL achieves strong generalization across models, robustness to diverse post-processing operations, and consistent effectiveness across varying protection ratios.

  </details>



- **DepthArb: Training-Free Depth-Arbitrated Generation for Occlusion-Robust Image Synthesis**  
  Hongjin Niu, Jiahao Wang, Xirui Hu, Weizhan Zhang, Lan Ma, Yuan Gao  
  _2026-03-25_ · https://arxiv.org/abs/2603.23924v1  
  <details><summary>Abstract</summary>

  Text-to-image diffusion models frequently exhibit deficiencies in synthesizing accurate occlusion relationships of multiple objects, particularly within dense overlapping regions. Existing training-free layout-guided methods predominantly rely on rigid spatial priors that remain agnostic to depth order, often resulting in concept mixing or illogical occlusion. To address these limitations, we propose DepthArb, a training-free framework that resolves occlusion ambiguities by arbitrating attention competition between interacting objects. Specifically, DepthArb employs two core mechanisms: Attention Arbitration Modulation (AAM), which enforces depth-ordered visibility by suppressing background activations in overlapping regions, and Spatial Compactness Control (SCC), which preserves structural integrity by curbing attention divergence. These mechanisms enable robust occlusion generation without model retraining. To systematically evaluate this capability, we propose OcclBench, a comprehensive benchmark designed to evaluate diverse occlusion scenarios. Extensive evaluations demonstrate that DepthArb consistently outperforms state-of-the-art baselines in both occlusion accuracy and visual fidelity. As a plug-and-play method, DepthArb seamlessly enhances the compositional capabilities of diffusion backbones, offering a novel perspective on spatial layering within generative models.

  </details>



- **Uncertainty-Aware Vision-based Risk Object Identification via Conformal Risk Tube Prediction**  
  Kai-Yu Fu, Yi-Ting Chen  
  _2026-03-25_ · https://arxiv.org/abs/2603.23919v1  
  <details><summary>Abstract</summary>

  We study object importance-based vision risk object identification (Vision-ROI), a key capability for hazard detection in intelligent driving systems. Existing approaches make deterministic decisions and ignore uncertainty, which could lead to safety-critical failures. Specifically, in ambiguous scenarios, fixed decision thresholds may cause premature or delayed risk detection and temporally unstable predictions, especially in complex scenes with multiple interacting risks. Despite these challenges, current methods lack a principled framework to model risk uncertainty jointly across space and time. We propose Conformal Risk Tube Prediction, a unified formulation that captures spatiotemporal risk uncertainty, provides coverage guarantees for true risks, and produces calibrated risk scores with uncertainty estimates. To conduct a systematic evaluation, we present a new dataset and metrics probing diverse scenario configurations with multi-risk coupling effects, which are not supported by existing datasets. We systematically analyze factors affecting uncertainty estimation, including scenario variations, per-risk category behavior, and perception error propagation. Our method delivers substantial improvements over prior approaches, enhancing vision-ROI robustness and downstream performance, such as reducing nuisance braking alerts. For more qualitative results, please visit our project webpage: https://hcis-lab.github.io/CRTP/

  </details>



- **DecepGPT: Schema-Driven Deception Detection with Multicultural Datasets and Robust Multimodal Learning**  
  Jiajian Huang, Dongliang Zhu, Zitong YU, Hui Ma, Jiayu Zhang, Chunmei Zhu, Xiaochun Cao  
  _2026-03-25_ · https://arxiv.org/abs/2603.23916v1  
  <details><summary>Abstract</summary>

  Multimodal deception detection aims to identify deceptive behavior by analyzing audiovisual cues for forensics and security. In these high-stakes settings, investigators need verifiable evidence connecting audiovisual cues to final decisions, along with reliable generalization across domains and cultural contexts. However, existing benchmarks provide only binary labels without intermediate reasoning cues. Datasets are also small with limited scenario coverage, leading to shortcut learning. We address these issues through three contributions. First, we construct reasoning datasets by augmenting existing benchmarks with structured cue-level descriptions and reasoning chains, enabling model output auditable reports. Second, we release T4-Deception, a multicultural dataset based on the unified ``To Tell The Truth'' television format implemented across four countries. With 1695 samples, it is the largest non-laboratory deception detection dataset. Third, we propose two modules for robust learning under small-data conditions. Stabilized Individuality-Commonality Synergy (SICS) refines multimodal representations by synergizing learnable global priors with sample-adaptive residuals, followed by a polarity-aware adjustment that bi-directionally recalibrates representations. Distilled Modality Consistency (DMC) aligns modality-specific predictions with the fused multimodal predictions via knowledge distillation to prevent unimodal shortcut learning. Experiments on three established benchmarks and our novel dataset demonstrate that our method achieves state-of-the-art performance in both in-domain and cross-domain scenarios, while exhibiting superior transferability across diverse cultural contexts. The datasets and codes will be released.

  </details>



- **MMTIT-Bench: A Multilingual and Multi-Scenario Benchmark with Cognition-Perception-Reasoning Guided Text-Image Machine Translation**  
  Gengluo Li, Chengquan Zhang, Yupu Liang, Huawen Shen, Yaping Zhang, Pengyuan Lyu, Weinong Wang, Xingyu Wan, Gangyan Zeng, Han Hu, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.23896v1  
  <details><summary>Abstract</summary>

  End-to-end text-image machine translation (TIMT), which directly translates textual content in images across languages, is crucial for real-world multilingual scene understanding. Despite advances in vision-language large models (VLLMs), robustness across diverse visual scenes and low-resource languages remains underexplored due to limited evaluation resources. We present MMTIT-Bench, a human-verified multilingual and multi-scenario benchmark with 1,400 images spanning fourteen non-English and non-Chinese languages and diverse settings such as documents, scenes, and web images, enabling rigorous assessment of end-to-end TIMT. Beyond benchmarking, we study how reasoning-oriented data design improves translation. Although recent VLLMs have begun to incorporate long Chain-of-Thought (CoT) reasoning, effective thinking paradigms for TIMT are still immature: existing designs either cascade parsing and translation in a sequential manner or focus on language-only reasoning, overlooking the visual cognition central to VLLMs. We propose Cognition-Perception-Reasoning for Translation (CPR-Trans), a data paradigm that integrates scene cognition, text perception, and translation reasoning within a unified reasoning process. Using a VLLM-driven data generation pipeline, CPR-Trans provides structured, interpretable supervision that aligns perception with reasoning. Experiments on 3B and 7B models show consistent gains in accuracy and interpretability. We will release MMTIT-Bench to promote the multilingual and multi-scenario TIMT research upon acceptance.

  </details>



- **FilterGS: Traversal-Free Parallel Filtering and Adaptive Shrinking for Large-Scale LoD 3D Gaussian Splatting**  
  Yixian Wang, Haolin Yu, Jiadong Tang, Yu Gao, Xihan Wang, Yufeng Yue, Yi Yang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23891v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has revolutionized neural rendering with real-time performance. However, scaling this approach to large scenes using Level-of-Detail methods faces critical challenges: inefficient serial traversal consuming over 60\% of rendering time, and redundant Gaussian-tile pairs that incur unnecessary processing overhead. To address these limitations, we introduce FilterGS, featuring a parallel filtering mechanism with two complementary filters that select Gaussian elements efficiently without tree traversal. Additionally, we propose a novel GTC metric that quantifies the redundancy of Gaussian-tile key-value pairs. Based on this metric, we introduce a scene-adaptive Gaussian shrinking strategy that effectively reduces redundant pairs. Extensive experiments demonstrate that FilterGS achieves state-of-the-art rendering speeds while maintaining competitive visual quality across multiple large-scale datasets. Project page: https://github.com/xenon-w/FilterGS

  </details>



- **Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training**  
  Gengluo Li, Chengquan Zhang, Yupu Liang, Huawen Shen, Yaping Zhang, Pengyuan Lyu, Weinong Wang, Xingyu Wan, Gangyan Zeng, Han Hu, et al.  
  _2026-03-25_ · https://arxiv.org/abs/2603.23885v1  
  <details><summary>Abstract</summary>

  Document parsing has recently advanced with multimodal large language models (MLLMs) that directly map document images to structured outputs. Traditional cascaded pipelines depend on precise layout analysis and often fail under casually captured or non-standard conditions. Although end-to-end approaches mitigate this dependency, they still exhibit repetitive, hallucinated, and structurally inconsistent predictions - primarily due to the scarcity of large-scale, high-quality full-page (document-level) end-to-end parsing data and the lack of structure-aware training strategies. To address these challenges, we propose a data-training co-design framework for robust end-to-end document parsing. A Realistic Scene Synthesis strategy constructs large-scale, structurally diverse full-page end-to-end supervision by composing layout templates with rich document elements, while a Document-Aware Training Recipe introduces progressive learning and structure-token optimization to enhance structural fidelity and decoding stability. We further build Wild-OmniDocBench, a benchmark derived from real-world captured documents for robustness evaluation. Integrated into a 1B-parameter MLLM, our method achieves superior accuracy and robustness across both scanned/digital and real-world captured scenarios. All models, data synthesis pipelines, and benchmarks will be publicly released to advance future research in document understanding.

  </details>



- **BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment**  
  Risa Shinoda, Kaede Shiohara, Nakamasa Inoue, Kuniaki Saito, Hiroaki Santo, Fumio Okura  
  _2026-03-25_ · https://arxiv.org/abs/2603.23883v1  
  <details><summary>Abstract</summary>

  Understanding animal species from multimodal data poses an emerging challenge at the intersection of computer vision and ecology. While recent biological models, such as BioCLIP, have demonstrated strong alignment between images and textual taxonomic information for species identification, the integration of the audio modality remains an open problem. We propose BioVITA, a novel visual-textual-acoustic alignment framework for biological applications. BioVITA involves (i) a training dataset, (ii) a representation model, and (iii) a retrieval benchmark. First, we construct a large-scale training dataset comprising 1.3 million audio clips and 2.3 million images, covering 14,133 species annotated with 34 ecological trait labels. Second, building upon BioCLIP2, we introduce a two-stage training framework to effectively align audio representations with visual and textual representations. Third, we develop a cross-modal retrieval benchmark that covers all possible directional retrieval across the three modalities (i.e., image-to-audio, audio-to-text, text-to-image, and their reverse directions), with three taxonomic levels: Family, Genus, and Species. Extensive experiments demonstrate that our model learns a unified representation space that captures species-level semantics beyond taxonomy, advancing multimodal biodiversity understanding. The project page is available at: https://dahlian00.github.io/BioVITA_Page/

  </details>



- **EnvSocial-Diff: A Diffusion-Based Crowd Simulation Model with Environmental Conditioning and Individual-Group Interaction**  
  Bingxue Zhao, Qi Zhang, Hui Huang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23874v1  
  <details><summary>Abstract</summary>

  Modeling realistic pedestrian trajectories requires accounting for both social interactions and environmental context, yet most existing approaches largely emphasize social dynamics. We propose \textbf{EnvSocial-Diff}: a diffusion-based crowd simulation model informed by social physics and augmented with environmental conditioning and individual--group interaction. Our structured environmental conditioning module explicitly encodes obstacles, objects of interest, and lighting levels, providing interpretable signals that capture scene constraints and attractors. In parallel, the individual--group interaction module goes beyond individual-level modeling by capturing both fine-grained interpersonal relations and group-level conformity through a graph-based design. Experiments on multiple benchmark datasets demonstrate that EnvSocial-Diff outperforms the latest state-of-the-art methods, underscoring the importance of explicit environmental conditioning and multi-level social interaction for realistic crowd simulation. Code is here: https://github.com/zqyq/EnvSocial-Diff.

  </details>



- **MLE-UVAD: Minimal Latent Entropy Autoencoder for Fully Unsupervised Video Anomaly Detection**  
  Yuang Geng, Junkai Zhou, Kang Yang, Pan He, Zhuoyang Zhou, Jose C. Principe, Joel Harley, Ivan Ruchkin  
  _2026-03-25_ · https://arxiv.org/abs/2603.23868v1  
  <details><summary>Abstract</summary>

  In this paper, we address the challenging problem of single-scene, fully unsupervised video anomaly detection (VAD), where raw videos containing both normal and abnormal events are used directly for training and testing without any labels. This differs sharply from prior work that either requires extensive labeling (fully or weakly supervised) or depends on normal-only videos (one-class classification), which are vulnerable to distribution shifts and contamination. We propose an entropy-guided autoencoder that detects anomalies through reconstruction error by reconstructing normal frames well while making anomalies reconstruct poorly. The key idea is to combine the standard reconstruction loss with a novel Minimal Latent Entropy (MLE) loss in the autoencoder. Reconstruction loss alone maps normal and abnormal inputs to distinct latent clusters due to their inherent differences, but also risks reconstructing anomalies too well to detect. Therefore, MLE loss addresses this by minimizing the entropy of latent embeddings, encouraging them to concentrate around high-density regions. Since normal frames dominate the raw video, sparse anomalous embeddings are pulled into the normal cluster, so the decoder emphasizes normal patterns and produces poor reconstructions for anomalies. This dual-loss design produces a clear reconstruction gap that enables effective anomaly detection. Extensive experiments on two widely used benchmarks and a challenging self-collected driving dataset demonstrate that our method achieves robust and superior performance over baselines.

  </details>



- **See, Remember, Explore: A Benchmark and Baselines for Streaming Spatial Reasoning**  
  Yuxi Wei, Wei Huang, Qirui Chen, Lu Hou, Xiaojuan Qi  
  _2026-03-25_ · https://arxiv.org/abs/2603.23864v1  
  <details><summary>Abstract</summary>

  Spatial understanding is fundamental for embodied agents, yet most spatial VLMs and benchmarks remain offline-evaluating post-hoc QA over pre-recorded inputs and overlooking two crucial deployment-critical requirements: long-horizon streaming inference and active perception when the current view is insufficient. To address this gap, we introduce S3-Bench, a benchmark suite for streaming spatial question answering with active exploration, where queries are temporally grounded to specific timestamps and must be answered using only observations available up to that moment. S3-Bench adopts a dual-domain design, combining a scalable simulator with controllable trajectories and exploration actions, and real-world streaming videos that capture practical sensing artifacts for rigorous generalization evaluation. Overall, it spans 10K+ scenes and 26K+ trajectories, with dedicated training (S3-Train) and evaluation (S3-Eval) splits. We further propose AMF-VLM, which supports streaming spatial reasoning under bounded computing via (i) memory folding, which compresses long-horizon observations into compact structured memory, and (ii) active exploration, which outputs explicit actions (e.g. move/rotate/scan) to acquire missing evidence before answering. Extensive experiments demonstrate that, compared to models using identical training data, our approach yields improvements of 8.8% and 13.3% on the simulated and real splits of S3-Eval, respectively, while maintaining competitive transferability to standard spatial benchmarks.

  </details>



- **Sparse Autoencoders for Interpretable Medical Image Representation Learning**  
  Philipp Wesp, Robbie Holland, Vasiliki Sideri-Lampretsa, Sergios Gatidis  
  _2026-03-24_ · https://arxiv.org/abs/2603.23794v1  
  <details><summary>Abstract</summary>

  Vision foundation models (FMs) achieve state-of-the-art performance in medical imaging. However, they encode information in abstract latent representations that clinicians cannot interrogate or verify. The goal of this study is to investigate Sparse Autoencoders (SAEs) for replacing opaque FM image representations with human-interpretable, sparse features. We train SAEs on embeddings from BiomedParse (biomedical) and DINOv3 (general-purpose) using 909,873 CT and MRI 2D image slices from the TotalSegmentator dataset. We find that learned sparse features: (a) reconstruct original embeddings with high fidelity (R2 up to 0.941) and recover up to 87.8% of downstream performance using only 10 features (99.4% dimensionality reduction), (b) preserve semantic fidelity in image retrieval tasks, (c) correspond to specific concepts that can be expressed in language using large language model (LLM)-based auto-interpretation. (d) bridge clinical language and abstract latent representations in zero-shot language-driven image retrieval. Our work indicates SAEs are a promising pathway towards interpretable, concept-driven medical vision systems. Code repository: https://github.com/pwesp/sail.

  </details>



- **Retinal Disease Classification from Fundus Images using CNN Transfer Learning**  
  Ali Akram  
  _2026-03-24_ · https://arxiv.org/abs/2603.23785v1  
  <details><summary>Abstract</summary>

  Retinal diseases remain among the leading preventable causes of visual impairment worldwide. Automated screening based on fundus image analysis has the potential to expand access to early detection, particularly in underserved populations. This paper presents a reproducible deep learning pipeline for binary retinal disease risk classification from publicly available fundus photographs. We implement and compare a baseline convolutional neural network with a transfer learning approach using a pretrained VGG16 backbone and evaluate generalization on held-out data. To address class imbalance, we apply class weighting and report standard classification metrics including accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC. The VGG16 transfer learning model achieves 90.8% test accuracy with a weighted F1-score of 0.90, substantially outperforming the baseline CNN (83.1% accuracy). Results indicate that transfer learning improves discrimination compared to a baseline CNN, while also revealing remaining challenges in sensitivity to minority disease cases. We discuss practical limitations related to dataset characteristics, class imbalance, and threshold selection, and provide guidance for reproducibility and future improvements for clinically reliable screening

  </details>


