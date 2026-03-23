# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-23 07:38 UTC_

Total papers shown: **31**


---

- **MME-CoF-Pro: Evaluating Reasoning Coherence in Video Generative Models with Text and Visual Hints**  
  Yu Qi, Xinyi Xu, Ziyu Guo, Siyuan Ma, Renrui Zhang, Xinyan Chen, Ruichuan An, Ruofan Xing, Jiayi Zhang, Haojie Huang, et al.  
  _2026-03-20_ · https://arxiv.org/abs/2603.20194v1  
  <details><summary>Abstract</summary>

  Video generative models show emerging reasoning behaviors. It is essential to ensure that generated events remain causally consistent across frames for reliable deployment, a property we define as reasoning coherence. To bridge the gap in literature for missing reasoning coherence evaluation, we propose MME-CoF-Pro, a comprehensive video reasoning benchmark to assess reasoning coherence in video models. Specifically, MME-CoF-Pro contains 303 samples across 16 categories, ranging from visual logical to scientific reasoning. It introduces Reasoning Score as evaluation metric for assessing process-level necessary intermediate reasoning steps, and includes three evaluation settings, (a) no hint (b) text hint and (c) visual hint, enabling a controlled investigation into the underlying mechanisms of reasoning hint guidance. Evaluation results in 7 open and closed-source video models reveals insights including: (1) Video generative models exhibit weak reasoning coherence, decoupled from generation quality. (2) Text hints boost apparent correctness but often cause inconsistency and hallucinated reasoning (3) Visual hints benefit structured perceptual tasks but struggle with fine-grained perception. Website: https://video-reasoning-coherence.github.io/

  </details>



- **From Masks to Pixels and Meaning: A New Taxonomy, Benchmark, and Metrics for VLM Image Tampering**  
  Xinyi Shang, Yi Tang, Jiacheng Cui, Ahmed Elhagry, Salwa K. Al Khatib, Sondos Mahmoud Bsharat, Jiacheng Liu, Xiaohan Zhao, Jing-Hao Xue, Hao Li, et al.  
  _2026-03-20_ · https://arxiv.org/abs/2603.20193v1  
  <details><summary>Abstract</summary>

  Existing tampering detection benchmarks largely rely on object masks, which severely misalign with the true edit signal: many pixels inside a mask are untouched or only trivially modified, while subtle yet consequential edits outside the mask are treated as natural. We reformulate VLM image tampering from coarse region labels to a pixel-grounded, meaning and language-aware task. First, we introduce a taxonomy spanning edit primitives (replace/remove/splice/inpaint/attribute/colorization, etc.) and their semantic class of tampered object, linking low-level changes to high-level understanding. Second, we release a new benchmark with per-pixel tamper maps and paired category supervision to evaluate detection and classification within a unified protocol. Third, we propose a training framework and evaluation metrics that quantify pixel-level correctness with localization to assess confidence or prediction on true edit intensity, and further measure tamper meaning understanding via semantics-aware classification and natural language descriptions for the predicted regions. We also re-evaluate the existing strong segmentation/localization baselines on recent strong tamper detectors and reveal substantial over- and under-scoring using mask-only metrics, and expose failure modes on micro-edits and off-mask changes. Our framework advances the field from masks to pixels, meanings and language descriptions, establishing a rigorous standard for tamper localization, semantic classification and description. Code and benchmark data are available at https://github.com/VILA-Lab/PIXAR.

  </details>



- **LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation**  
  Jiazheng Xing, Fei Du, Hangjie Yuan, Pengwei Liu, Hongbin Xu, Hai Ci, Ruigang Niu, Weihua Chen, Fan Wang, Yong Liu  
  _2026-03-20_ · https://arxiv.org/abs/2603.20192v1  
  <details><summary>Abstract</summary>

  Recent advances in diffusion models have significantly improved text-to-video generation, enabling personalized content creation with fine-grained control over both foreground and background elements. However, precise face-attribute alignment across subjects remains challenging, as existing methods lack explicit mechanisms to ensure intra-group consistency. Addressing this gap requires both explicit modeling strategies and face-attribute-aware data resources. We therefore propose LumosX, a framework that advances both data and model design. On the data side, a tailored collection pipeline orchestrates captions and visual cues from independent videos, while multimodal large language models (MLLMs) infer and assign subject-specific dependencies. These extracted relational priors impose a finer-grained structure that amplifies the expressive control of personalized video generation and enables the construction of a comprehensive benchmark. On the modeling side, Relational Self-Attention and Relational Cross-Attention intertwine position-aware embeddings with refined attention dynamics to inscribe explicit subject-attribute dependencies, enforcing disciplined intra-group cohesion and amplifying the separation between distinct subject clusters. Comprehensive evaluations on our benchmark demonstrate that LumosX achieves state-of-the-art performance in fine-grained, identity-consistent, and semantically aligned personalized multi-subject video generation. Code and models are available at https://jiazheng-xing.github.io/lumosx-home/.

  </details>



- **Deterministic Mode Proposals: An Efficient Alternative to Generative Sampling for Ambiguous Segmentation**  
  Sebastian Gerard, Josephine Sullivan  
  _2026-03-20_ · https://arxiv.org/abs/2603.20191v1  
  <details><summary>Abstract</summary>

  Many segmentation tasks, such as medical image segmentation or future state prediction, are inherently ambiguous, meaning that multiple predictions are equally correct. Current methods typically rely on generative models to capture this uncertainty. However, identifying the underlying modes of the distribution with these methods is computationally expensive, requiring large numbers of samples and post-hoc clustering. In this paper, we shift the focus from stochastic sampling to the direct generation of likely outcomes. We introduce mode proposal models, a deterministic framework that efficiently produces a fixed-size set of proposal masks in a single forward pass. To handle superfluous proposals, we adapt a confidence mechanism, traditionally used in object detection, to the high-dimensional space of segmentation masks. Our approach significantly reduces inference time while achieving higher ground-truth coverage than existing generative models. Furthermore, we demonstrate that our model can be trained without knowing the full distribution of outcomes, making it applicable to real-world datasets. Finally, we show that by decomposing the velocity field of a pre-trained flow model, we can efficiently estimate prior mode probabilities for our proposals.

  </details>



- **CoVR-R:Reason-Aware Composed Video Retrieval**  
  Omkar Thawakar, Dmitry Demidov, Vaishnav Potlapalli, Sai Prasanna Teja Reddy Bogireddy, Viswanatha Reddy Gajjala, Alaa Mostafa Lasheen, Rao Muhammad Anwer, Fahad Khan  
  _2026-03-20_ · https://arxiv.org/abs/2603.20190v1  
  <details><summary>Abstract</summary>

  Composed Video Retrieval (CoVR) aims to find a target video given a reference video and a textual modification. Prior work assumes the modification text fully specifies the visual changes, overlooking after-effects and implicit consequences (e.g., motion, state transitions, viewpoint or duration cues) that emerge from the edit. We argue that successful CoVR requires reasoning about these after-effects. We introduce a reasoning-first, zero-shot approach that leverages large multimodal models to (i) infer causal and temporal consequences implied by the edit, and (ii) align the resulting reasoned queries to candidate videos without task-specific finetuning. To evaluate reasoning in CoVR, we also propose CoVR-Reason, a benchmark that pairs each (reference, edit, target) triplet with structured internal reasoning traces and challenging distractors that require predicting after-effects rather than keyword matching. Experiments show that our zero-shot method outperforms strong retrieval baselines on recall at K and particularly excels on implicit-effect subsets. Our automatic and human analysis confirm higher step consistency and effect factuality in our retrieved results. Our findings show that incorporating reasoning into general-purpose multimodal models enables effective CoVR by explicitly accounting for causal and temporal after-effects. This reduces dependence on task-specific supervision, improves generalization to challenging implicit-effect cases, and enhances interpretability of retrieval outcomes. These results point toward a scalable and principled framework for explainable video search. The model, code, and benchmark are available at https://github.com/mbzuai-oryx/CoVR-R.

  </details>



- **Wildfire Spread Scenarios: Increasing Sample Diversity of Segmentation Diffusion Models with Training-Free Methods**  
  Sebastian Gerard, Josephine Sullivan  
  _2026-03-20_ · https://arxiv.org/abs/2603.20188v1  
  <details><summary>Abstract</summary>

  Predicting future states in uncertain environments, such as wildfire spread, medical diagnosis, or autonomous driving, requires models that can consider multiple plausible outcomes. While diffusion models can effectively learn such multi-modal distributions, naively sampling from these models is computationally inefficient, potentially requiring hundreds of samples to find low-probability modes that may still be operationally relevant. In this work, we address the challenge of sample-efficient ambiguous segmentation by evaluating several training-free sampling methods that encourage diverse predictions. We adapt two techniques, particle guidance and SPELL, originally designed for the generation of diverse natural images, to discrete segmentation tasks, and additionally propose a simple clustering-based technique. We validate these approaches on the LIDC medical dataset, a modified version of the Cityscapes dataset, and MMFire, a new simulation-based wildfire spread dataset introduced in this paper. Compared to naive sampling, these approaches increase the HM IoU* metric by up to 7.5% on MMFire and 16.4% on Cityscapes, demonstrating that training-free methods can be used to efficiently increase the sample diversity of segmentation diffusion models with little cost to image quality and runtime. Code and dataset: https://github.com/SebastianGer/wildfire-spread-scenarios

  </details>



- **IndoorR2X: Indoor Robot-to-Everything Coordination with LLM-Driven Planning**  
  Fan Yang, Soumya Teotia, Shaunak A. Mehta, Prajit KrisshnaKumar, Quanting Xie, Jun Liu, Yueqi Song, Li Wenkai, Atsunori Moteki, Kanji Uchino, et al.  
  _2026-03-20_ · https://arxiv.org/abs/2603.20182v1  
  <details><summary>Abstract</summary>

  Although robot-to-robot (R2R) communication improves indoor scene understanding beyond what a single robot can achieve, R2R alone cannot overcome partial observability without substantial exploration overhead or scaling team size. In contrast, many indoor environments already include low-cost Internet of Things (IoT) sensors (e.g., cameras) that provide persistent, building-wide context beyond onboard perception. We therefore introduce IndoorR2X, the first benchmark and simulation framework for Large Language Model (LLM)-driven multi-robot task planning with Robot-to-Everything (R2X) perception and communication in indoor environments. IndoorR2X integrates observations from mobile robots and static IoT devices to construct a global semantic state that supports scalable scene understanding, reduces redundant exploration, and enables high-level coordination through LLM-based planning. IndoorR2X provides configurable simulation environments, sensor layouts, robot teams, and task suites to systematically evaluate high-level semantic coordination strategies. Extensive experiments across diverse settings demonstrate that IoT-augmented world modeling improves multi-robot efficiency and reliability, and we highlight key insights and failure modes for advancing LLM-based collaboration between robot teams and indoor IoT sensors.

  </details>



- **TinyML Enhances CubeSat Mission Capabilities**  
  Luigi Capogrosso, Michele Magno  
  _2026-03-20_ · https://arxiv.org/abs/2603.20174v1  
  <details><summary>Abstract</summary>

  Earth observation (EO) missions traditionally rely on transmitting raw or minimally processed imagery from satellites to ground stations for computationally intensive analysis. This paradigm is infeasible for CubeSat systems due to stringent constraints on the onboard embedded processors, energy availability, and communication bandwidth. To overcome these limitations, the paper presents a TinyML-based Convolutional Neural Networks (ConvNets) model optimization and deployment pipeline for onboard image classification, enabling accurate, energy-efficient, and hardware-aware inference under CubeSat-class constraints. Our pipeline integrates structured iterative pruning, post-training INT8 quantization, and hardware-aware operator mapping to compress models and align them with the heterogeneous compute architecture of the STM32N6 microcontroller from STMicroelectronics. This Microcontroller Unit (MCU) integrates a novel Arm Cortex-M55 core and a Neural-ART Neural Processing Unit (NPU), providing a realistic proxy for CubeSat onboard computers. The paper evaluates the proposed approach on three EO benchmark datasets (i.e., EuroSAT, RS_C11, MEDIC) and four models (i.e., SqueezeNet, MobileNetV3, EfficientNet, MCUNetV1). We demonstrate an average reduction in RAM usage of 89.55% and Flash memory of 70.09% for the optimized models, significantly decreasing downlink bandwidth requirements while maintaining task-acceptable accuracy (with a drop ranging from 0.4 to 8.6 percentage points compared to the Float32 baseline). The energy consumption per inference ranges from 0.68 mJ to 6.45 mJ, with latency spanning from 3.22 ms to 30.38 ms. These results fully satisfy the stringent energy budgets and real-time constraints required for efficient onboard EO processing.

  </details>



- **Can Large Multimodal Models Inspect Buildings? A Hierarchical Benchmark for Structural Pathology Reasoning**  
  Hui Zhong, Yichun Gao, Luyan Liu, Hai Yang, Wang Wang, Haowei Zhang, Xinhu Zheng  
  _2026-03-20_ · https://arxiv.org/abs/2603.20148v1  
  <details><summary>Abstract</summary>

  Automated building facade inspection is a critical component of urban resilience and smart city maintenance. Traditionally, this field has relied on specialized discriminative models (e.g., YOLO, Mask R-CNN) that excel at pixel-level localization but are constrained to passive perception and worse generization without the visual understandng to interpret structural topology. Large Multimodal Models (LMMs) promise a paradigm shift toward active reasoning, yet their application in such high-stakes engineering domains lacks rigorous evaluation standards. To bridge this gap, we introduce a human-in-the-loop semi-automated annotation framework, leveraging expert-proposal verification to unify 12 fragmented datasets into a standardized, hierarchical ontology. Building on this foundation, we present \textit{DefectBench}, the first multi-dimensional benchmark designed to interrogate LMMs beyond basic semantic recognition. \textit{DefectBench} evaluates 18 state-of-the-art (SOTA) LMMs across three escalating cognitive dimensions: Semantic Perception, Spatial Localization, and Generative Geometry Segmentation. Extensive experiments reveal that while current LMMs demonstrate exceptional topological awareness and semantic understanding (effectively diagnosing "what" and "how"), they exhibit significant deficiencies in metric localization precision ("where"). Crucially, however, we validate the viability of zero-shot generative segmentation, showing that general-purpose foundation models can rival specialized supervised networks without domain-specific training. This work provides both a rigorous benchmarking standard and a high-quality open-source database, establishing a new baseline for the advancement of autonomous AI agents in civil engineering.

  </details>



- **Synergistic Perception and Generative Recomposition: A Multi-Agent Orchestration for Expert-Level Building Inspection**  
  Hui Zhong, Yichun Gao, Luyan Liu, Xusen Guo, Zhaonian Kuang, Qiming Zhang, Xinhu Zheng  
  _2026-03-20_ · https://arxiv.org/abs/2603.20143v1  
  <details><summary>Abstract</summary>

  Building facade defect inspection is fundamental to structural health monitoring and sustainable urban maintenance, yet it remains a formidable challenge due to extreme geometric variability, low contrast against complex backgrounds, and the inherent complexity of composite defects (e.g., cracks co-occurring with spalling). Such characteristics lead to severe pixel imbalance and feature ambiguity, which, coupled with the critical scarcity of high-quality pixel-level annotations, hinder the generalization of existing detection and segmentation models. To address gaps, we propose \textit{FacadeFixer}, a unified multi-agent framework that treats defect perception as a collaborative reasoning task rather than isolated recognition. Specifically,\textit{FacadeFixer} orchestrates specialized agents for detection and segmentation to handle multi-type defect interference, working in tandem with a generative agent to enable semantic recomposition. This process decouples intricate defects from noisy backgrounds and realistically synthesizes them onto diverse clean textures, generating high-fidelity augmented data with precise expert-level masks. To support this, we introduce a comprehensive multi-task dataset covering six primary facade categories with pixel-level annotations. Extensive experiments demonstrate that \textit{FacadeFixer} significantly outperforms state-of-the-art (SOTA) baselines. Specifically, it excels in capturing pixel-level structural anomalies and highlights generative synthesis as a robust solution to data scarcity in infrastructure inspection. Our code and dataset will be made publicly available.

  </details>



- **Uncertainty Matters: Structured Probabilistic Online Mapping for Motion Prediction in Autonomous Driving**  
  Pritom Gogoi, Faris Janjoš, Bin Yang, Andreas Look  
  _2026-03-20_ · https://arxiv.org/abs/2603.20076v1  
  <details><summary>Abstract</summary>

  Online map generation and trajectory prediction are critical components of the autonomous driving perception-prediction-planning pipeline. While modern vectorized mapping models achieve high geometric accuracy, they typically treat map estimation as a deterministic task, discarding structural uncertainty. Existing probabilistic approaches often rely on diagonal covariance matrices, which assume independence between points and fail to capture the strong spatial correlations inherent in road geometry. To address this, we propose a structured probabilistic formulation for online map generation. Our method explicitly models intra-element dependencies by predicting a dense covariance matrix, parameterized via a Low-Rank plus Diagonal (LRPD) covariance decomposition. This formulation represents uncertainty as a combination of a low-rank component, which captures global spatial structure, and a diagonal component representing independent local noise, thereby capturing geometric correlations without the prohibitive computational cost of full covariance matrices. Evaluations on the nuScenes dataset demonstrate that our uncertainty-aware framework yields consistent improvements in online map generation quality compared to deterministic baselines. Furthermore, our approach establishes new state-of-the-art performance for map-based motion prediction, highlighting the critical role of uncertainty in planning tasks. Code is published under link-available-soon.

  </details>



- **MFil-Mamba: Multi-Filter Scanning for Spatial Redundancy-Aware Visual State Space Models**  
  Puskal Khadka, KC Santosh  
  _2026-03-20_ · https://arxiv.org/abs/2603.20074v1  
  <details><summary>Abstract</summary>

  State Space Models (SSMs), especially recent Mamba architecture, have achieved remarkable success in sequence modeling tasks. However, extending SSMs to computer vision remains challenging due to the non-sequential structure of visual data and its complex 2D spatial dependencies. Although several early studies have explored adapting selective SSMs for vision applications, most approaches primarily depend on employing various traversal strategies over the same input. This introduces redundancy and distorts the intricate spatial relationships within images. To address these challenges, we propose MFil-Mamba, a novel visual state space architecture built on a multi-filter scanning backbone. Unlike fixed multi-directional traversal methods, our design enables each scan to capture unique and contextually relevant spatial information while minimizing redundancy. Furthermore, we incorporate an adaptive weighting mechanism to effectively fuse outputs from multiple scans in addition to architectural enhancements. MFil-Mamba achieves superior performance over existing state-of-the-art models across various benchmarks that include image classification, object detection, instance segmentation, and semantic segmentation. For example, our tiny variant attains 83.2% top-1 accuracy on ImageNet-1K, 47.3% box AP and 42.7% mask AP on MS COCO, and 48.5% mIoU on the ADE20K dataset. Code and models are available at https://github.com/puskal-khadka/MFil-Mamba.

  </details>



- **Layered Quantum Architecture Search for 3D Point Cloud Classification**  
  Natacha Kuete Meli, Jovita Lukasik, Vladislav Golyanik, Michael Moeller  
  _2026-03-20_ · https://arxiv.org/abs/2603.20024v1  
  <details><summary>Abstract</summary>

  We introduce layered Quantum Architecture Search (layered-QAS), a strategy inspired by classical network morphism that designs Parametrised Quantum Circuit (PQC) architectures by progressively growing and adapting them. PQCs offer strong expressiveness with relatively few parameters, yet they lack standard architectural layers (e.g., convolution, attention) that encode inductive biases for a given learning task. To assess the effectiveness of our method, we focus on 3D point cloud classification as a challenging yet highly structured problem. Whereas prior work on this task has used PQCs only as feature extractors for classical classifiers, our approach uses the PQC as the main building block of the classification model. Simulations show that our layered-QAS mitigates barren plateau, outperforms quantum-adapted local and evolutionary QAS baselines, and achieves state-of-the-art results among PQC-based methods on the ModelNet dataset.

  </details>



- **NEC-Diff: Noise-Robust Event-RAW Complementary Diffusion for Seeing Motion in Extreme Darkness**  
  Haoyue Liu, Jinghan Xu, Luxin Feng, Hanyu Zhou, Haozhi Zhao, Yi Chang, Luxin Yan  
  _2026-03-20_ · https://arxiv.org/abs/2603.20005v1  
  <details><summary>Abstract</summary>

  High-quality imaging of dynamic scenes in extremely low-light conditions is highly challenging. Photon scarcity induces severe noise and texture loss, causing significant image degradation. Event cameras, featuring a high dynamic range (120 dB) and high sensitivity to motion, serve as powerful complements to conventional cameras by offering crucial cues for preserving subtle textures. However, most existing approaches emphasize texture recovery from events, while paying little attention to image noise or the intrinsic noise of events themselves, which ultimately hinders accurate pixel reconstruction under photon-starved conditions. In this work, we propose NEC-Diff, a novel diffusion-based event-RAW hybrid imaging framework that extracts reliable information from heavily noisy signals to reconstruct fine scene structures. The framework is driven by two key insights: (1) combining the linear light-response property of RAW images with the brightness-change nature of events to establish a physics-driven constraint for robust dual-modal denoising; and (2) dynamically estimating the SNR of both modalities based on denoising results to guide adaptive feature fusion, thereby injecting reliable cues into the diffusion process for high-fidelity visual reconstruction. Furthermore, we construct the REAL (Raw and Event Acquired in Low-light) dataset which provides 47,800 pixel-aligned low-light RAW images, events, and high-quality references under 0.001-0.8 lux illumination. Extensive experiments demonstrate the superiority of NEC-Diff under extreme darkness. The project are available at: https://github.com/jinghan-xu/NEC-Diff.

  </details>



- **Evaluating Test-Time Adaptation For Facial Expression Recognition Under Natural Cross-Dataset Distribution Shifts**  
  John Turnbull, Shivam Grover, Amin Jalali, Ali Etemad  
  _2026-03-20_ · https://arxiv.org/abs/2603.19994v1  
  <details><summary>Abstract</summary>

  Deep learning models often struggle under natural distribution shifts, a common challenge in real-world deployments. Test-Time Adaptation (TTA) addresses this by adapting models during inference without labeled source data. We present the first evaluation of TTA methods for FER under natural domain shifts, performing cross-dataset experiments with widely used FER datasets. This moves beyond synthetic corruptions to examine real-world shifts caused by differing collection protocols, annotation standards, and demographics. Results show TTA can boost FER performance under natural shifts by up to 11.34\%. Entropy minimization methods such as TENT and SAR perform best when the target distribution is clean. In contrast, prototype adjustment methods like T3A excel under larger distributional distance scenarios. Finally, feature alignment methods such as SHOT deliver the largest gains when the target distribution is noisier than our source. Our cross-dataset analysis shows that TTA effectiveness is governed by the distributional distance and the severity of the natural shift across domains.

  </details>



- **MedSPOT: A Workflow-Aware Sequential Grounding Benchmark for Clinical GUI**  
  Rozain Shakeel, Abdul Rahman Mohammad Ali, Muneeb Mushtaq, Tausifa Jan Saleem, Tajamul Ashraf  
  _2026-03-20_ · https://arxiv.org/abs/2603.19993v1  
  <details><summary>Abstract</summary>

  Despite the rapid progress of Multimodal Large Language Models (MLLMs), their ability to perform reliable visual grounding in high-stakes clinical software environments remains underexplored. Existing GUI benchmarks largely focus on isolated, single-step grounding queries, overlooking the sequential, workflow-driven reasoning required in real-world medical interfaces, where tasks evolve across independent steps and dynamic interface states. We introduce MedSPOT, a workflow-aware sequential grounding benchmark for clinical GUI environments. Unlike prior benchmarks that treat grounding as a standalone prediction task, MedSPOT models procedural interaction as a sequence of structured spatial decisions. The benchmark comprises 216 task-driven videos with 597 annotated keyframes, in which each task consists of 2 to 3 interdependent grounding steps within realistic medical workflows. This design captures interface hierarchies, contextual dependencies, and fine-grained spatial precision under evolving conditions. To evaluate procedural robustness, we propose a strict sequential evaluation protocol that terminates task assessment upon the first incorrect grounding prediction, explicitly measuring error propagation in multi-step workflows. We further introduce a comprehensive failure taxonomy, including edge bias, small-target errors, no prediction, near miss, far miss, and toolbar confusion, to enable systematic diagnosis of model behavior in clinical GUI settings. By shifting evaluation from isolated grounding to workflow-aware sequential reasoning, MedSPOT establishes a realistic and safety-critical benchmark for assessing multimodal models in medical software environments. Code and data are available at: https://github.com/Tajamul21/MedSPOT.

  </details>



- **2K Retrofit: Entropy-Guided Efficient Sparse Refinement for High-Resolution 3D Geometry Prediction**  
  Tianbao Zhang, Zhenyu Liang, Zhenbo Song, Nana Wang, Xiaomei Zhang, Xudong Cai, Zheng Zhu, Kejian Wu, Gang Wang, Zhaoxin Fan  
  _2026-03-20_ · https://arxiv.org/abs/2603.19964v1  
  <details><summary>Abstract</summary>

  High-resolution geometric prediction is essential for robust perception in autonomous driving, robotics, and AR/MR, but current foundation models are fundamentally limited by their scalability to real-world, high-resolution scenarios. Direct inference on 2K images with these models incurs prohibitive computational and memory demands, making practical deployment challenging. To tackle the issue, we present 2K Retrofit, a novel framework that enables efficient 2K-resolution inference for any geometric foundation model, without modifying or retraining the backbone. Our approach leverages fast coarse predictions and an entropy-based sparse refinement to selectively enhance high-uncertainty regions, achieving precise and high-fidelity 2K outputs with minimal overhead. Extensive experiments on widely used benchmark demonstrate that 2K Retrofit consistently achieves state-of-the-art accuracy and speed, bridging the gap between research advances and scalable deployment in high-resolution 3D vision applications. Code will be released upon acceptance.

  </details>



- **LIORNet: Self-Supervised LiDAR Snow Removal Framework for Autonomous Driving under Adverse Weather Conditions**  
  Ji-il Park, Inwook Shim  
  _2026-03-20_ · https://arxiv.org/abs/2603.19936v1  
  <details><summary>Abstract</summary>

  LiDAR sensors provide high-resolution 3D perception and long-range detection, making them indispensable for autonomous driving and robotics. However, their performance significantly degrades under adverse weather conditions such as snow, rain, and fog, where spurious noise points dominate the point cloud and lead to false perception. To address this problem, various approaches have been proposed: distance-based filters exploiting spatial sparsity, intensity-based filters leveraging reflectance distributions, and learning-based methods that adapt to complex environments. Nevertheless, distance-based methods struggle to distinguish valid object points from noise, intensity-based methods often rely on fixed thresholds that lack adaptability to changing conditions, and learning-based methods suffer from the high cost of annotation, limited generalization, and computational overhead. In this study, we propose LIORNet, which eliminates these drawbacks and integrates the strengths of all three paradigms. LIORNet is built upon a U-Net++ backbone and employs a self-supervised learning strategy guided by pseudo-labels generated from multiple physical and statistical cues, including range-dependent intensity thresholds, snow reflectivity, point sparsity, and sensing range constraints. This design enables LIORNet to distinguish noise points from environmental structures without requiring manual annotations, thereby overcoming the difficulty of snow labeling and the limitations of single-principle approaches. Extensive experiments on the WADS and CADC datasets demonstrate that LIORNet outperforms state-of-the-art filtering algorithms in both accuracy and runtime while preserving critical environmental features. These results highlight LIORNet as a practical and robust solution for LiDAR perception in extreme weather, with strong potential for real-time deployment in autonomous driving systems.

  </details>



- **MedQ-Engine: A Closed-Loop Data Engine for Evolving MLLMs in Medical Image Quality Assessment**  
  Jiyao Liu, Junzhi Ning, Wanying Qu, Lihao Liu, Chenglong Ma, Junjun He, Ningsheng Xu  
  _2026-03-20_ · https://arxiv.org/abs/2603.19863v1  
  <details><summary>Abstract</summary>

  Medical image quality assessment (Med-IQA) is a prerequisite for clinical AI deployment, yet multimodal large language models (MLLMs) still fall substantially short of human experts, particularly when required to provide descriptive assessments with clinical reasoning beyond simple quality scores. However, improving them is hindered by the high cost of acquiring descriptive annotations and by the inability of one-time data collection to adapt to the model's evolving weaknesses. To address these challenges, we propose MedQ-Engine, a closed-loop data engine that iteratively evaluates the model to discover failure prototypes via data-driven clustering, explores a million-scale image pool using these prototypes as retrieval anchors with progressive human-in-the-loop annotation, and evolves through quality-assured fine-tuning, forming a self-improving cycle. Models are evaluated on complementary perception and description tasks. An entropy-guided routing mechanism triages annotations to minimize labeling cost. Experiments across five medical imaging modalities show that MedQ-Engine elevates an 8B-parameter model to surpass GPT-4o by over 13% and narrow the gap with human experts to only 4.34%, using only 10K annotations with more than 4x sample efficiency over random sampling.

  </details>



- **FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts**  
  You Li, Dewei Zhou, Fan Ma, Fu Li, Dongliang He, Yi Yang  
  _2026-03-20_ · https://arxiv.org/abs/2603.19857v1  
  <details><summary>Abstract</summary>

  Recent Video-to-Audio (V2A) methods have achieved remarkable progress, enabling the synthesis of realistic, high-quality audio. However, they struggle with fine-grained temporal control in multi-event scenarios or when visual cues are insufficient, such as small regions, off-screen sounds, or occluded or partially visible objects. In this paper, we propose FoleyDirector, a framework that, for the first time, enables precise temporal guidance in DiT-based V2A generation while preserving the base model's audio quality and allowing seamless switching between V2A generation and temporally controlled synthesis. FoleyDirector introduces Structured Temporal Scripts (STS), a set of captions corresponding to short temporal segments, to provide richer temporal information. These features are integrated via the Script-Guided Temporal Fusion Module, which employs Temporal Script Attention to fuse STS features coherently. To handle complex multi-event scenarios, we further propose Bi-Frame Sound Synthesis, enabling parallel in-frame and out-of-frame audio generation and improving controllability. To support training and evaluation, we construct the DirectorSound dataset and introduce VGGSoundDirector and DirectorBench. Experiments demonstrate that FoleyDirector substantially enhances temporal controllability while maintaining high audio fidelity, empowering users to act as Foley directors and advancing V2A toward more expressive and controllable generation.

  </details>



- **Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them**  
  Michael Hubbertz, Qi Han, Tobias Meisen  
  _2026-03-20_ · https://arxiv.org/abs/2603.19852v1  
  <details><summary>Abstract</summary>

  Deep learning-based online mapping has emerged as a cornerstone of autonomous driving, yet these models frequently fail to generalize beyond familiar environments. We propose a framework to identify and measure the underlying failure modes by disentangling two effects: Memorization of input features and overfitting to known map geometries. We propose measures based on evaluation subsets that control for geographical proximity and geometric similarity between training and validation scenes. We introduce Fréchet distance-based reconstruction statistics that capture per-element shape fidelity without threshold tuning, and define complementary failure-mode scores: a localization overfitting score quantifying the performance drop when geographic cues disappear, and a map geometry overfitting score measuring degradation as scenes become geometrically novel. Beyond models, we analyze dataset biases and contribute map geometry-aware diagnostics: A minimum-spanning-tree (MST) diversity measure for training sets and a symmetric coverage measure to quantify geometric similarity between splits. Leveraging these, we formulate an MST-based sparsification strategy that reduces redundancy and improves balancing and performance while shrinking training size. Experiments on nuScenes and Argoverse 2 across multiple state-of-the-art models yield more trustworthy assessment of generalization and show that map geometry-diverse and balanced training sets lead to improved performance. Our results motivate failure-mode-aware protocols and map geometry-centric dataset design for deployable online mapping.

  </details>



- **Hyper-Connections for Adaptive Multi-Modal MRI Brain Tumor Segmentation**  
  Lokendra Kumar, Shubham Aggarwal  
  _2026-03-20_ · https://arxiv.org/abs/2603.19844v1  
  <details><summary>Abstract</summary>

  We present the first study of Hyper-Connections (HC) for volumetric multi-modal brain tumor segmentation, integrating them as a drop-in replacement for fixed residual connections across five architectures: nnU-Net, SwinUNETR, VT-UNet, U-Net, and U-Netpp. Dynamic HC consistently improves all 3D models on the BraTS 2021 dataset, yielding up to +1.03 percent mean Dice gain with negligible parameter overhead. Gains are most pronounced in the Enhancing Tumor sub-region, reflecting improved fine-grained boundary delineation. Modality ablation further reveals that HC-equipped models develop sharper sensitivity toward clinically dominant sequences, specifically T1ce for Tumor Core and Enhancing Tumor, and FLAIR for Whole Tumor, a behavior absent in fixed-connection baselines and consistent across all architectures. In 2D settings, improvements are smaller and configuration-sensitive, suggesting that volumetric spatial context amplifies the benefit of adaptive aggregation. These results establish HC as a simple, efficient, and broadly applicable mechanism for multi-modal feature fusion in medical image segmentation.

  </details>



- **HUGE-Bench: A Benchmark for High-Level UAV Vision-Language-Action Tasks**  
  Jingyu Guo, Ziye Chen, Ziwen Li, Zhengqing Gao, Jiaxin Huang, Hanlue Zhang, Fengming Huang, Yu Yao, Tongliang Liu, Mingming Gong  
  _2026-03-20_ · https://arxiv.org/abs/2603.19822v1  
  <details><summary>Abstract</summary>

  Existing UAV vision-language navigation (VLN) benchmarks have enabled language-guided flight, but they largely focus on long, step-wise route descriptions with goal-centric evaluation, making them less diagnostic for real operations where brief, high-level commands must be grounded into safe multi-stage behaviors. We present HUGE-Bench, a benchmark for High-Level UAV Vision-Language-Action (HL-VLA) tasks that tests whether an agent can interpret concise language and execute complex, process-oriented trajectories with safety awareness. HUGE-Bench comprises 4 real-world digital twin scenes, 8 high-level tasks, and 2.56M meters of trajectories, and is built on an aligned 3D Gaussian Splatting (3DGS)-Mesh representation that combines photorealistic rendering with collision-capable geometry for scalable generation and collision-aware evaluation. We introduce process-oriented and collision-aware metrics to assess process fidelity, terminal accuracy, and safety. Experiments on representative state-of-the-art VLA models reveal significant gaps in high-level semantic completion and safe execution, highlighting HUGE-Bench as a diagnostic testbed for high-level UAV autonomy.

  </details>



- **Evaluating Vision Foundation Models for Pixel and Object Classification in Microscopy**  
  Carolin Teuber, Anwai Archit, Tobias Boothe, Peter Ditte, Jochen Rink, Constantin Pape  
  _2026-03-20_ · https://arxiv.org/abs/2603.19802v1  
  <details><summary>Abstract</summary>

  Deep learning underlies most modern approaches and tools in computer vision, including biomedical imaging. However, for interactive semantic segmentation (often called pixel classification in this context) and interactive object-level classification (object classification), feature-based shallow learning remains widely used. This is due to the diversity of data in this domain, the lack of large pretraining datasets, and the need for computational and label efficiency. In contrast, state-of-the-art tools for many other vision tasks in microscopy - most notably cellular instance segmentation - already rely on deep learning and have recently benefited substantially from vision foundation models (VFMs), particularly SAM. Here, we investigate whether VFMs can also improve pixel and object classification compared to current approaches. To this end, we evaluate several VFMs, including general-purpose models (SAM, SAM2, DINOv3) and domain-specific ones ($μ$SAM, PathoSAM), in combination with shallow learning and attentive probing on five diverse and challenging datasets. Our results demonstrate consistent improvements over hand-crafted features and provide a clear pathway toward practical improvements. Furthermore, our study establishes a benchmark for VFMs in microscopy and informs future developments in this area.

  </details>



- **Offshore oil and gas platform dynamics in the North Sea, Gulf of Mexico, and Persian Gulf: Exploiting the Sentinel-1 archive**  
  Robin Spanier, Thorsten Hoeser, John Truckenbrodt, Felix Bachofer, Claudia Kuenzer  
  _2026-03-20_ · https://arxiv.org/abs/2603.19801v1  
  <details><summary>Abstract</summary>

  The increasing use of marine spaces by offshore infrastructure, including oil and gas platforms, underscores the need for consistent, scalable monitoring. Offshore development has economic, environmental, and regulatory implications, yet maritime areas remain difficult to monitor systematically due to their inaccessibility and spatial extent. This study presents an automated approach to the spatiotemporal detection of offshore oil and gas platforms based on freely available Earth observation data. Leveraging Sentinel-1 archive data and deep learning-based object detection, a consistent quarterly time series of platform locations for three major production regions: the North Sea, the Gulf of Mexico, and the Persian Gulf, was created for the period 2017-2025. In addition, platform size, water depth, distance to the coast, national affiliation, and installation and decommissioning dates were derived. 3,728 offshore platforms were identified in 2025, 356 in the North Sea, 1,641 in the Gulf of Mexico, and 1,731 in the Persian Gulf. While expansion was observed in the Persian Gulf until 2024, the Gulf of Mexico and the North Sea saw a decline in platform numbers from 2018-2020. At the same time, a pronounced dynamic was apparent. More than 2,700 platforms were installed or relocated to new sites, while a comparable number were decommissioned or relocated. Furthermore, the increasing number of platforms with short lifespans points to a structural change in the offshore sector associated with the growing importance of mobile offshore units such as jack-ups or drillships. The results highlighted the potential of freely available Earth observation data and deep learning for consistent, long-term monitoring of marine infrastructure. The derived dataset is public and provides a basis for offshore monitoring, maritime planning, and analyses of the transformation of the offshore energy sector.

  </details>



- **From Plausibility to Verifiability: Risk-Controlled Generative OCR for Vision-Language Models**  
  Weile Gong, Yiping Zuo, Zijian Lu, Xin He, Weibei Fan, Chen Dai  
  _2026-03-20_ · https://arxiv.org/abs/2603.19790v1  
  <details><summary>Abstract</summary>

  Modern vision-language models (VLMs) can act as generative OCR engines, yet open-ended decoding can expose rare but consequential failures. We identify a core deployment misalignment in generative OCR. Autoregressive decoding favors semantic plausibility, whereas OCR requires outputs that are visually grounded and geometrically verifiable. This mismatch produces severe errors, especially over-generation and unsupported substitutions, creating deployment risk even when benchmark accuracy remains high. We therefore formulate frozen VLM OCR as a selective accept/abstain problem and propose a model-agnostic Geometric Risk Controller. The controller probes multiple structured views of the same input, applies lightweight structural screening, and accepts a transcription only when cross-view consensus and stability satisfy predefined criteria, yielding a small family of operating points. Experiments on frozen VLM backbones and standard OCR benchmarks show consistent reductions in extreme-error risk and catastrophic over-generation at predictable coverage costs. Reliable deployment of generative OCR with frozen VLMs benefits from explicit system-level risk control rather than unconstrained generation.

  </details>



- **Evaluating Image Editing with LLMs: A Comprehensive Benchmark and Intermediate-Layer Probing Approach**  
  Shiqi Gao, Zitong Xu, Kang Fu, Huiyu Duan, Xiongkuo Min, Jia wang, Guangtao Zhai  
  _2026-03-20_ · https://arxiv.org/abs/2603.19775v1  
  <details><summary>Abstract</summary>

  Evaluating text-guided image editing (TIE) methods remains a challenging problem, as reliable assessment should simultaneously consider perceptual quality, alignment with textual instructions, and preservation of original image content. Despite rapid progress in TIE models, existing evaluation benchmarks remain limited in scale and often show weak correlation with human perceptual judgments. In this work, we introduce TIEdit, a benchmark for systematic evaluation of text-guided image editing methods. TIEdit consists of 512 source images paired with editing prompts across eight representative editing tasks, producing 5,120 edited images generated by ten state-of-the-art TIE models. To obtain reliable subjective ratings, 20 experts are recruited to produce 307,200 raw subjective ratings, which accumulates into 15,360 mean opinion scores (MOSs) across three evaluation dimensions: perceptual quality, editing alignment, and content preservation. Beyond the benchmark itself, we further propose EditProbe, an LLM-based evaluator that estimates editing quality via intermediate-layer probing of hidden representations. Instead of relying solely on final model outputs, EditProbe extracts informative representations from intermediate layers of multimodal large language models to better capture semantic and perceptual relationships between source images, editing instructions, and edited results. Experimental results demonstrate that widely used automatic evaluation metrics show limited correlation with human judgments on editing tasks, while EditProbe achieves substantially stronger alignment with human perception. Together, TIEdit and EditProbe provide a foundation for more reliable and perceptually aligned evaluation of text-guided image editing methods.

  </details>



- **Template-based Object Detection Using a Foundation Model**  
  Valentin Braeutigam, Matthias Stock, Bernhard Egger  
  _2026-03-20_ · https://arxiv.org/abs/2603.19773v1  
  <details><summary>Abstract</summary>

  Most currently used object detection methods are learning-based, and can detect objects under varying appearances. Those models require training and a training dataset. We focus on use cases with less data variation, but the requirement of being free of generation of training data and training. Such a setup is for example desired in automatic testing of graphical interfaces during software development, especially for continuous integration testing. In our approach, we use segments from segmentation foundation models and combine them with a simple feature-based classification method. This saves time and cost when changing the object to be searched or its design, as nothing has to be retrained and no dataset has to be created. We evaluate our method on the task of detecting and classifying icons in navigation maps, which is used to simplify and automate the testing of user interfaces in automotive industry. Our methods achieve results almost on par with learning-based object detection methods like YOLO, without the need for training.

  </details>



- **FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision**  
  Zekai Wu, Shuqi Fan, Mengyin Liu, Yuhua Luo, Xincheng Lin, Ming Yan, Junhao Wu, Xiuhong Lin, Yuexin Ma, Chenglu Wen, et al.  
  _2026-03-20_ · https://arxiv.org/abs/2603.19770v1  
  <details><summary>Abstract</summary>

  Precise motion timing (PMT) is crucial for swift motion analysis. A millisecond difference may determine victory or defeat in sports competitions. Despite substantial progress in human pose estimation (HPE), PMT remains largely overlooked by the HPE community due to the limited availability of high-temporal-resolution labeled datasets. Today, PMT is achieved using high-speed RGB cameras in specialized scenarios such as the Olympic Games; however, their high costs, light sensitivity, bandwidth, and computational complexity limit their feasibility for daily use. We developed FlashCap, the first flashing LED-based MoCap system for PMT. With FlashCap, we collect a millisecond-resolution human motion dataset, FlashMotion, comprising the event, RGB, LiDAR, and IMU modalities, and demonstrate its high quality through rigorous validation. To evaluate the merits of FlashMotion, we perform two tasks: precise motion timing and high-temporal-resolution HPE. For these tasks, we propose ResPose, a simple yet effective baseline that learns residual poses based on events and RGBs. Experimental results show that ResPose reduces pose estimation errors by ~40% and achieves millisecond-level timing accuracy, enabling new research opportunities. The dataset and code will be shared with the community.

  </details>



- **FREAK: A Fine-grained Hallucination Evaluation Benchmark for Advanced MLLMs**  
  Zhihan Yin, Jianxin Liang, Yueqian Wang, Yifeng Yao, Huishuai Zhang, Dongyan Zhao  
  _2026-03-20_ · https://arxiv.org/abs/2603.19765v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) suffer from hallucinations. Existing hallucination evaluation benchmarks are often limited by over-simplified tasks leading to saturated metrics, or insufficient diversity that fails to adequately assess the hallucination extent in state-of-the-art multimodal models. To address this gap, we propose FREAK, a comprehensive multimodal benchmark designed for fine-grained hallucination assessment in MLLMs. Through high-quality photorealistic images featuring fine-grained counter-commonsense edits, FREAK innovatively evaluates hallucination phenomena in detailed visual perception of MLLMs. Extensive experiments on FREAK show severe hallucination issues in SOTA models regarding detailed visual perception. To enable deeper investigation, we curate a controlled subset to indirectly evaluate the model's ability to perceive target detailed information. Through systematic evaluation of prevailing Chain-of-Thought (CoT) prompting techniques within this task, we reveal critical insights regarding hallucination patterns and model reasoning processes.

  </details>



- **PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences**  
  Min Lin, Gangwei Xu, Xianqi Wang, Yuyi Peng, Xin Yang  
  _2026-03-20_ · https://arxiv.org/abs/2603.19762v1  
  <details><summary>Abstract</summary>

  Point cloud scene flow estimation is fundamental to long-term and fine-grained 3D motion analysis. However, existing methods are typically limited to pairwise settings and struggle to maintain temporal consistency over long sequences as geometry evolves, occlusions emerge, and errors accumulate. In this work, we propose PCSTracker, the first end-to-end framework specifically designed for consistent scene flow estimation in point cloud sequences. Specifically, we introduce an iterative geometry motion joint optimization module (IGMO) that explicitly models the temporal evolution of point features to alleviate correspondence inconsistencies caused by dynamic geometric changes. In addition, a spatio-temporal point trajectory update module (STTU) is proposed to leverage broad temporal context to infer plausible positions for occluded points, ensuring coherent motion estimation. To further handle long sequences, we employ an overlapping sliding-window inference strategy that alternates cross-window propagation and in-window refinement, effectively suppressing error accumulation and maintaining stable long-term motion consistency. Extensive experiments on the synthetic PointOdyssey3D and real-world ADT3D datasets show that PCSTracker achieves the best accuracy in long-term scene flow estimation and maintains real-time performance at 32.5 FPS, while demonstrating superior 3D motion understanding compared to RGB-D-based approaches.

  </details>


