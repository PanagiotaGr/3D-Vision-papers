# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-01-29 07:04 UTC_

Total papers shown: **50**


---

- **C3Box: A CLIP-based Class-Incremental Learning Toolbox**  
  Hao Sun, Da-Wei Zhou  
  _2026-01-28_ · https://arxiv.org/abs/2601.20852v1  
  <details><summary>Abstract</summary>

  Traditional machine learning systems are typically designed for static data distributions, which suffer from catastrophic forgetting when learning from evolving data streams. Class-Incremental Learning (CIL) addresses this challenge by enabling learning systems to continuously learn new classes while preserving prior knowledge. With the rise of pre-trained models (PTMs) such as CLIP, leveraging their strong generalization and semantic alignment capabilities has become a promising direction in CIL. However, existing CLIP-based CIL methods are often scattered across disparate codebases, rely on inconsistent configurations, hindering fair comparisons, reproducibility, and practical adoption. Therefore, we propose C3Box (CLIP-based Class-inCremental learning toolBOX), a modular and comprehensive Python toolbox. C3Box integrates representative traditional CIL methods, ViT-based CIL methods, and state-of-the-art CLIP-based CIL methods into a unified CLIP-based framework. By inheriting the streamlined design of PyCIL, C3Box provides a JSON-based configuration and standardized execution pipeline. This design enables reproducible experimentation with low engineering overhead and makes C3Box a reliable benchmark platform for continual learning research. Designed to be user-friendly, C3Box relies only on widely used open-source libraries and supports major operating systems. The code is available at https://github.com/LAMDA-CL/C3Box.

  </details>



- **A New Dataset and Framework for Robust Road Surface Classification via Camera-IMU Fusion**  
  Willams de Lima Costa, Thifany Ketuli Silva de Souza, Jonas Ferreira Silva, Carlos Gabriel Bezerra Pereira, Bruno Reis Vila Nova, Leonardo Silvino Brito, Rafael Raider Leoni, Juliano Silva, Valter Ferreira, Sibele Miguel Soares Neto, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20847v1  
  <details><summary>Abstract</summary>

  Road surface classification (RSC) is a key enabler for environment-aware predictive maintenance systems. However, existing RSC techniques often fail to generalize beyond narrow operational conditions due to limited sensing modalities and datasets that lack environmental diversity. This work addresses these limitations by introducing a multimodal framework that fuses images and inertial measurements using a lightweight bidirectional cross-attention module followed by an adaptive gating layer that adjusts modality contributions under domain shifts. Given the limitations of current benchmarks, especially regarding lack of variability, we introduce ROAD, a new dataset composed of three complementary subsets: (i) real-world multimodal recordings with RGB-IMU streams synchronized using a gold-standard industry datalogger, captured across diverse lighting, weather, and surface conditions; (ii) a large vision-only subset designed to assess robustness under adverse illumination and heterogeneous capture setups; and (iii) a synthetic subset generated to study out-of-distribution generalization in scenarios difficult to obtain in practice. Experiments show that our method achieves a +1.4 pp improvement over the previous state-of-the-art on the PVS benchmark and an +11.6 pp improvement on our multimodal ROAD subset, with consistently higher F1-scores on minority classes. The framework also demonstrates stable performance across challenging visual conditions, including nighttime, heavy rain, and mixed-surface transitions. These findings indicate that combining affordable camera and IMU sensors with multimodal attention mechanisms provides a scalable, robust foundation for road surface understanding, particularly relevant for regions where environmental variability and cost constraints limit the adoption of high-end sensing suites.

  </details>



- **MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents**  
  Vishnu Sashank Dorbala, Dinesh Manocha  
  _2026-01-28_ · https://arxiv.org/abs/2601.20831v1  
  <details><summary>Abstract</summary>

  Foundation models rely on in-context learning for personalized decision making. The limited size of this context window necessitates memory compression and retrieval systems like RAG. These systems however often treat memory as large offline storage spaces, which is unfavorable for embodied agents that are expected to operate under strict memory and compute constraints, online. In this work, we propose MemCtrl, a novel framework that uses Multimodal Large Language Models (MLLMs) for pruning memory online. MemCtrl augments MLLMs with a trainable memory head μthat acts as a gate to determine which observations or reflections to retain, update, or discard during exploration. We evaluate with training two types of μ, 1) via an offline expert, and 2) via online RL, and observe significant improvement in overall embodied task completion ability on μ-augmented MLLMs. In particular, on augmenting two low performing MLLMs with MemCtrl on multiple subsets of the EmbodiedBench benchmark, we observe that μ-augmented MLLMs show an improvement of around 16% on average, with over 20% on specific instruction subsets. Finally, we present a qualitative analysis on the memory fragments collected by μ, noting the superior performance of μaugmented MLLMs on long and complex instruction types.

  </details>



- **FAIRT2V: Training-Free Debiasing for Text-to-Video Diffusion Models**  
  Haonan Zhong, Wei Song, Tingxu Han, Maurice Pagnucco, Jingling Xue, Yang Song  
  _2026-01-28_ · https://arxiv.org/abs/2601.20791v1  
  <details><summary>Abstract</summary>

  Text-to-video (T2V) diffusion models have achieved rapid progress, yet their demographic biases, particularly gender bias, remain largely unexplored. We present FairT2V, a training-free debiasing framework for text-to-video generation that mitigates encoder-induced bias without finetuning. We first analyze demographic bias in T2V models and show that it primarily originates from pretrained text encoders, which encode implicit gender associations even for neutral prompts. We quantify this effect with a gender-leaning score that correlates with bias in generated videos. Based on this insight, FairT2V mitigates demographic bias by neutralizing prompt embeddings via anchor-based spherical geodesic transformations while preserving semantics. To maintain temporal coherence, we apply debiasing only during early identity-forming steps through a dynamic denoising schedule. We further propose a video-level fairness evaluation protocol combining VideoLLM-based reasoning with human verification. Experiments on the modern T2V model Open-Sora show that FairT2V substantially reduces demographic bias across occupations with minimal impact on video quality.

  </details>



- **Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy**  
  Huanyu Tian, Martin Huber, Lingyun Zeng, Zhe Han, Wayne Bennett, Giuseppe Silvestri, Gerardo Mendizabal-Ruiz, Tom Vercauteren, Alejandro Chavez-Badiola, Christos Bergeles  
  _2026-01-28_ · https://arxiv.org/abs/2601.20776v1  
  <details><summary>Abstract</summary>

  This paper rethinks steady-hand robotic manipulation by using a weakly supervised framework that fuses calibration-aware perception with admittance control. Unlike conventional automation that relies on labor-intensive 2D labeling, our framework leverages reusable warm-up trajectories to extract implicit spatial information, thereby achieving calibration-aware, depth-resolved perception without the need for external fiducials or manual depth annotation. By explicitly characterizing residuals from observation and calibration models, the system establishes a task-space error budget from recorded warm-ups. The uncertainty budget yields a lateral closed-loop accuracy of approx. 49 micrometers at 95% confidence (worst-case testing subset) and a depth accuracy of <= 291 micrometers at 95% confidence bound during large in-plane moves. In a within-subject user study (N=8), the learned agent reduces overall NASA-TLX workload by 77.1% relative to the simple steady-hand assistance baseline. These results demonstrate that the weakly supervised agent improves the reliability of microscope-guided biomedical micromanipulation without introducing complex setup requirements, offering a practical framework for microscope-guided intervention.

  </details>



- **Decoupling Perception and Calibration: Label-Efficient Image Quality Assessment Framework**  
  Xinyue Li, Zhichao Zhang, Zhiming Xu, Shubo Xu, Xiongkuo Min, Yitong Chen, Guangtao Zhai  
  _2026-01-28_ · https://arxiv.org/abs/2601.20689v1  
  <details><summary>Abstract</summary>

  Recent multimodal large language models (MLLMs) have demonstrated strong capabilities in image quality assessment (IQA) tasks. However, adapting such large-scale models is computationally expensive and still relies on substantial Mean Opinion Score (MOS) annotations. We argue that for MLLM-based IQA, the core bottleneck lies not in the quality perception capacity of MLLMs, but in MOS scale calibration. Therefore, we propose LEAF, a Label-Efficient Image Quality Assessment Framework that distills perceptual quality priors from an MLLM teacher into a lightweight student regressor, enabling MOS calibration with minimal human supervision. Specifically, the teacher conducts dense supervision through point-wise judgments and pair-wise preferences, with an estimate of decision reliability. Guided by these signals, the student learns the teacher's quality perception patterns through joint distillation and is calibrated on a small MOS subset to align with human annotations. Experiments on both user-generated and AI-generated IQA benchmarks demonstrate that our method significantly reduces the need for human annotations while maintaining strong MOS-aligned correlations, making lightweight IQA practical under limited annotation budgets.

  </details>



- **ProSkill: Segment-Level Skill Assessment in Procedural Videos**  
  Michele Mazzamuto, Daniele Di Mauro, Gianpiero Francesca, Giovanni Maria Farinella, Antonino Furnari  
  _2026-01-28_ · https://arxiv.org/abs/2601.20661v1  
  <details><summary>Abstract</summary>

  Skill assessment in procedural videos is crucial for the objective evaluation of human performance in settings such as manufacturing and procedural daily tasks. Current research on skill assessment has predominantly focused on sports and lacks large-scale datasets for complex procedural activities. Existing studies typically involve only a limited number of actions, focus on either pairwise assessments (e.g., A is better than B) or on binary labels (e.g., good execution vs needs improvement). In response to these shortcomings, we introduce ProSkill, the first benchmark dataset for action-level skill assessment in procedural tasks. ProSkill provides absolute skill assessment annotations, along with pairwise ones. This is enabled by a novel and scalable annotation protocol that allows for the creation of an absolute skill assessment ranking starting from pairwise assessments. This protocol leverages a Swiss Tournament scheme for efficient pairwise comparisons, which are then aggregated into consistent, continuous global scores using an ELO-based rating system. We use our dataset to benchmark the main state-of-the-art skill assessment algorithms, including both ranking-based and pairwise paradigms. The suboptimal results achieved by the current state-of-the-art highlight the challenges and thus the value of ProSkill in the context of skill assessment for procedural videos. All data and code are available at https://fpv-iplab.github.io/ProSkill/

  </details>



- **FD-MAD: Frequency-Domain Residual Analysis for Face Morphing Attack Detection**  
  Diogo J. Paulo, Hugo Proença, João C. Neves  
  _2026-01-28_ · https://arxiv.org/abs/2601.20656v1  
  <details><summary>Abstract</summary>

  Face morphing attacks present a significant threat to face recognition systems used in electronic identity enrolment and border control, particularly in single-image morphing attack detection (S-MAD) scenarios where no trusted reference is available. In spite of the vast amount of research on this problem, morph detection systems struggle in cross-dataset scenarios. To address this problem, we introduce a region-aware frequency-based morph detection strategy that drastically improves over strong baseline methods in challenging cross-dataset and cross-morph settings using a lightweight approach. Having observed the separability of bona fide and morph samples in the frequency domain of different facial parts, our approach 1) introduces the concept of residual frequency domain, where the frequency of the signal is decoupled from the natural spectral decay to easily discriminate between morph and bona fide data; 2) additionally, we reason in a global and local manner by combining the evidence from different facial regions in a Markov Random Field, which infers a globally consistent decision. The proposed method, trained exclusively on the synthetic morphing attack detection development dataset (SMDD), is evaluated in challenging cross-dataset and cross-morph settings on FRLL-Morph and MAD22 sets. Our approach achieves an average equal error rate (EER) of 1.85\% on FRLL-Morph and ranks second on MAD22 with an average EER of 6.12\%, while also obtaining a good bona fide presentation classification error rate (BPCER) at a low attack presentation classification error rate (APCER) using only spectral features. These findings indicate that Fourier-domain residual modeling with structured regional fusion offers a competitive alternative to deep S-MAD architectures.

  </details>



- **OS-Marathon: Benchmarking Computer-Use Agents on Long-Horizon Repetitive Tasks**  
  Jing Wu, Daphne Barretto, Yiye Chen, Nicholas Gydé, Yanan Jian, Yuhang He, Vibhav Vineet  
  _2026-01-28_ · https://arxiv.org/abs/2601.20650v1  
  <details><summary>Abstract</summary>

  Long-horizon, repetitive workflows are common in professional settings, such as processing expense reports from receipts and entering student grades from exam papers. These tasks are often tedious for humans since they can extend to extreme lengths proportional to the size of the data to process. However, they are ideal for Computer-Use Agents (CUAs) due to their structured, recurring sub-workflows with logic that can be systematically learned. Identifying the absence of an evaluation benchmark as a primary bottleneck, we establish OS-Marathon, comprising 242 long-horizon, repetitive tasks across 2 domains to evaluate state-of-the-art (SOTA) agents. We then introduce a cost-effective method to construct a condensed demonstration using only few-shot examples to teach agents the underlying workflow logic, enabling them to execute similar workflows effectively on larger, unseen data collections. Extensive experiments demonstrate both the inherent challenges of these tasks and the effectiveness of our proposed method. Project website: https://os-marathon.github.io/.

  </details>



- **GDCNet: Generative Discrepancy Comparison Network for Multimodal Sarcasm Detection**  
  Shuguang Zhang, Junhong Lian, Guoxin Yu, Baoxun Xu, Xiang Ao  
  _2026-01-28_ · https://arxiv.org/abs/2601.20618v1  
  <details><summary>Abstract</summary>

  Multimodal sarcasm detection (MSD) aims to identify sarcasm within image-text pairs by modeling semantic incongruities across modalities. Existing methods often exploit cross-modal embedding misalignment to detect inconsistency but struggle when visual and textual content are loosely related or semantically indirect. While recent approaches leverage large language models (LLMs) to generate sarcastic cues, the inherent diversity and subjectivity of these generations often introduce noise. To address these limitations, we propose the Generative Discrepancy Comparison Network (GDCNet). This framework captures cross-modal conflicts by utilizing descriptive, factually grounded image captions generated by Multimodal LLMs (MLLMs) as stable semantic anchors. Specifically, GDCNet computes semantic and sentiment discrepancies between the generated objective description and the original text, alongside measuring visual-textual fidelity. These discrepancy features are then fused with visual and textual representations via a gated module to adaptively balance modality contributions. Extensive experiments on MSD benchmarks demonstrate GDCNet's superior accuracy and robustness, establishing a new state-of-the-art on the MMSD2.0 benchmark.

  </details>



- **CLEAR-Mamba:Towards Accurate, Adaptive and Trustworthy Multi-Sequence Ophthalmic Angiography Classification**  
  Zhuonan Wang, Wenjie Yan, Wenqiao Zhang, Xiaohui Song, Jian Ma, Ke Yao, Yibo Yu, Beng Chin Ooi  
  _2026-01-28_ · https://arxiv.org/abs/2601.20601v1  
  <details><summary>Abstract</summary>

  Medical image classification is a core task in computer-aided diagnosis (CAD), playing a pivotal role in early disease detection, treatment planning, and patient prognosis assessment. In ophthalmic practice, fluorescein fundus angiography (FFA) and indocyanine green angiography (ICGA) provide hemodynamic and lesion-structural information that conventional fundus photography cannot capture. However, due to the single-modality nature, subtle lesion patterns, and significant inter-device variability, existing methods still face limitations in generalization and high-confidence prediction. To address these challenges, we propose CLEAR-Mamba, an enhanced framework built upon MedMamba with optimizations in both architecture and training strategy. Architecturally, we introduce HaC, a hypernetwork-based adaptive conditioning layer that dynamically generates parameters according to input feature distributions, thereby improving cross-domain adaptability. From a training perspective, we develop RaP, a reliability-aware prediction scheme built upon evidential uncertainty learning, which encourages the model to emphasize low-confidence samples and improves overall stability and reliability. We further construct a large-scale ophthalmic angiography dataset covering both FFA and ICGA modalities, comprising multiple retinal disease categories for model training and evaluation. Experimental results demonstrate that CLEAR-Mamba consistently outperforms multiple baseline models, including the original MedMamba, across various metrics-showing particular advantages in multi-disease classification and reliability-aware prediction. This study provides an effective solution that balances generalizability and reliability for modality-specific medical image classification tasks.

  </details>



- **Person Re-ID in 2025: Supervised, Self-Supervised, and Language-Aligned. What Works?**  
  Lakshman Balasubramanian  
  _2026-01-28_ · https://arxiv.org/abs/2601.20598v1  
  <details><summary>Abstract</summary>

  Person Re-Identification (ReID) remains a challenging problem in computer vision. This work reviews various training paradigm and evaluates the robustness of state-of-the-art ReID models in cross-domain applications and examines the role of foundation models in improving generalization through richer, more transferable visual representations. We compare three training paradigms, supervised, self-supervised, and language-aligned models. Through the study the aim is to answer the following questions: Can supervised models generalize in cross-domain scenarios? How does foundation models like SigLIP2 perform for the ReID tasks? What are the weaknesses of current supervised and foundational models for ReID? We have conducted the analysis across 11 models and 9 datasets. Our results show a clear split: supervised models dominate their training domain but crumble on cross-domain data. Language-aligned models, however, show surprising robustness cross-domain for ReID tasks, even though they are not explicitly trained to do so. Code and data available at: https://github.com/moiiai-tech/object-reid-benchmark.

  </details>



- **StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval**  
  Shaokun Wang, Weili Guan, Jizhou Han, Jianlong Wu, Yupeng Hu, Liqiang Nie  
  _2026-01-28_ · https://arxiv.org/abs/2601.20597v1  
  <details><summary>Abstract</summary>

  Continual Text-to-Video Retrieval (CTVR) is a challenging multimodal continual learning setting, where models must incrementally learn new semantic categories while maintaining accurate text-video alignment for previously learned ones, thus making it particularly prone to catastrophic forgetting. A key challenge in CTVR is feature drift, which manifests in two forms: intra-modal feature drift caused by continual learning within each modality, and non-cooperative feature drift across modalities that leads to modality misalignment. To mitigate these issues, we propose StructAlign, a structured cross-modal alignment method for CTVR. First, StructAlign introduces a simplex Equiangular Tight Frame (ETF) geometry as a unified geometric prior to mitigate modality misalignment. Building upon this geometric prior, we design a cross-modal ETF alignment loss that aligns text and video features with category-level ETF prototypes, encouraging the learned representations to form an approximate simplex ETF geometry. In addition, to suppress intra-modal feature drift, we design a Cross-modal Relation Preserving loss, which leverages complementary modalities to preserve cross-modal similarity relations, providing stable relational supervision for feature updates. By jointly addressing non-cooperative feature drift across modalities and intra-modal feature drift, StructAlign effectively alleviates catastrophic forgetting in CTVR. Extensive experiments on benchmark datasets demonstrate that our method consistently outperforms state-of-the-art continual retrieval approaches.

  </details>



- **MeCo: Enhancing LLM-Empowered Multi-Robot Collaboration via Similar Task Memoization**  
  Baiqing Wang, Helei Cui, Bo Zhang, Xiaolong Zheng, Bin Guo, Zhiwen Yu  
  _2026-01-28_ · https://arxiv.org/abs/2601.20577v1  
  <details><summary>Abstract</summary>

  Multi-robot systems have been widely deployed in real-world applications, providing significant improvements in efficiency and reductions in labor costs. However, most existing multi-robot collaboration methods rely on extensive task-specific training, which limits their adaptability to new or diverse scenarios. Recent research leverages the language understanding and reasoning capabilities of large language models (LLMs) to enable more flexible collaboration without specialized training. Yet, current LLM-empowered approaches remain inefficient: when confronted with identical or similar tasks, they must replan from scratch because they omit task-level similarities. To address this limitation, we propose MeCo, a similarity-aware multi-robot collaboration framework that applies the principle of ``cache and reuse'' (a.k.a., memoization) to reduce redundant computation. Unlike simple task repetition, identifying and reusing solutions for similar but not identical tasks is far more challenging, particularly in multi-robot settings. To this end, MeCo introduces a new similarity testing method that retrieves previously solved tasks with high relevance, enabling effective plan reuse without re-invoking LLMs. Furthermore, we present MeCoBench, the first benchmark designed to evaluate performance on similar-task collaboration scenarios. Experimental results show that MeCo substantially reduces planning costs and improves success rates compared with state-of-the-art approaches.

  </details>



- **SegRap2025: A Benchmark of Gross Tumor Volume and Lymph Node Clinical Target Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma**  
  Jia Fu, Litingyu Wang, He Li, Zihao Luo, Huamin Wang, Chenyuan Bian, Zijun Gao, Chunbin Gu, Xin Weng, Jianghao Wu, et al.  
  _2026-01-28_ · https://arxiv.org/abs/2601.20575v1  
  <details><summary>Abstract</summary>

  Accurate delineation of Gross Tumor Volume (GTV), Lymph Node Clinical Target Volume (LN CTV), and Organ-at-Risk (OAR) from Computed Tomography (CT) scans is essential for precise radiotherapy planning in Nasopharyngeal Carcinoma (NPC). Building upon SegRap2023, which focused on OAR and GTV segmentation using single-center paired non-contrast CT (ncCT) and contrast-enhanced CT (ceCT) scans, the SegRap2025 challenge aims to enhance the generalizability and robustness of segmentation models across imaging centers and modalities. SegRap2025 comprises two tasks: Task01 addresses GTV segmentation using paired CT from the SegRap2023 dataset, with an additional external testing set to evaluate cross-center generalization, and Task02 focuses on LN CTV segmentation using multi-center training data and an unseen external testing set, where each case contains paired CT scans or a single modality, emphasizing both cross-center and cross-modality robustness. This paper presents the challenge setup and provides a comprehensive analysis of the solutions submitted by ten participating teams. For GTV segmentation task, the top-performing models achieved average Dice Similarity Coefficient (DSC) of 74.61% and 56.79% on the internal and external testing cohorts, respectively. For LN CTV segmentation task, the highest average DSC values reached 60.24%, 60.50%, and 57.23% on paired CT, ceCT-only, and ncCT-only subsets, respectively. SegRap2025 establishes a large-scale multi-center, multi-modality benchmark for evaluating the generalization and robustness in radiotherapy target segmentation, providing valuable insights toward clinically applicable automated radiotherapy planning systems. The benchmark is available at: https://hilab-git.github.io/SegRap2025_Challenge.

  </details>



- **DiffVC-RT: Towards Practical Real-Time Diffusion-based Perceptual Neural Video Compression**  
  Wenzhuo Ma, Zhenzhong Chen  
  _2026-01-28_ · https://arxiv.org/abs/2601.20564v1  
  <details><summary>Abstract</summary>

  The practical deployment of diffusion-based Neural Video Compression (NVC) faces critical challenges, including severe information loss, prohibitive inference latency, and poor temporal consistency. To bridge this gap, we propose DiffVC-RT, the first framework designed to achieve real-time diffusion-based perceptual NVC. First, we introduce an Efficient and Informative Model Architecture. Through strategic module replacements and pruning, this architecture significantly reduces computational complexity while mitigating structural information loss. Second, to address generative flickering artifacts, we propose Explicit and Implicit Consistency Modeling. We enhance temporal consistency by explicitly incorporating a zero-cost Online Temporal Shift Module within the U-Net, complemented by hybrid implicit consistency constraints. Finally, we present an Asynchronous and Parallel Decoding Pipeline incorporating Mixed Half Precision, which enables asynchronous latent decoding and parallel frame reconstruction via a Batch-dimension Temporal Shift design. Experiments show that DiffVC-RT achieves 80.1% bitrate savings in terms of LPIPS over VTM-17.0 on HEVC dataset with real-time encoding and decoding speeds of 206 / 30 fps for 720p videos on an NVIDIA H800 GPU, marking a significant milestone in diffusion-based video compression.

  </details>



- **AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors**  
  Matic Fučka, Vitjan Zavrtanik, Danijel Skočaj  
  _2026-01-28_ · https://arxiv.org/abs/2601.20524v1  
  <details><summary>Abstract</summary>

  Zero-shot anomaly detection aims to detect and localise abnormal regions in the image without access to any in-domain training images. While recent approaches leverage vision-language models (VLMs), such as CLIP, to transfer high-level concept knowledge, methods based on purely vision foundation models (VFMs), like DINOv2, have lagged behind in performance. We argue that this gap stems from two practical issues: (i) limited diversity in existing auxiliary anomaly detection datasets and (ii) overly shallow VFM adaptation strategies. To address both challenges, we propose AnomalyVFM, a general and effective framework that turns any pretrained VFM into a strong zero-shot anomaly detector. Our approach combines a robust three-stage synthetic dataset generation scheme with a parameter-efficient adaptation mechanism, utilising low-rank feature adapters and a confidence-weighted pixel loss. Together, these components enable modern VFMs to substantially outperform current state-of-the-art methods. More specifically, with RADIO as a backbone, AnomalyVFM achieves an average image-level AUROC of 94.1% across 9 diverse datasets, surpassing previous methods by significant 3.3 percentage points. Project Page: https://maticfuc.github.io/anomaly_vfm/

  </details>



- **Say Cheese! Detail-Preserving Portrait Collection Generation via Natural Language Edits**  
  Zelong Sun, Jiahui Wu, Ying Ba, Dong Jing, Zhiwu Lu  
  _2026-01-28_ · https://arxiv.org/abs/2601.20511v1  
  <details><summary>Abstract</summary>

  As social media platforms proliferate, users increasingly demand intuitive ways to create diverse, high-quality portrait collections. In this work, we introduce Portrait Collection Generation (PCG), a novel task that generates coherent portrait collections by editing a reference portrait image through natural language instructions. This task poses two unique challenges to existing methods: (1) complex multi-attribute modifications such as pose, spatial layout, and camera viewpoint; and (2) high-fidelity detail preservation including identity, clothing, and accessories. To address these challenges, we propose CHEESE, the first large-scale PCG dataset containing 24K portrait collections and 573K samples with high-quality modification text annotations, constructed through an Large Vison-Language Model-based pipeline with inversion-based verification. We further propose SCheese, a framework that combines text-guided generation with hierarchical identity and detail preservation. SCheese employs adaptive feature fusion mechanism to maintain identity consistency, and ConsistencyNet to inject fine-grained features for detail consistency. Comprehensive experiments validate the effectiveness of CHEESE in advancing PCG, with SCheese achieving state-of-the-art performance.

  </details>



- **Latent Temporal Discrepancy as Motion Prior: A Loss-Weighting Strategy for Dynamic Fidelity in T2V**  
  Meiqi Wu, Bingze Song, Ruimin Lin, Chen Zhu, Xiaokun Feng, Jiahong Wu, Xiangxiang Chu, Kaiqi Huang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20504v1  
  <details><summary>Abstract</summary>

  Video generation models have achieved notable progress in static scenarios, yet their performance in motion video generation remains limited, with quality degrading under drastic dynamic changes. This is due to noise disrupting temporal coherence and increasing the difficulty of learning dynamic regions. {Unfortunately, existing diffusion models rely on static loss for all scenarios, constraining their ability to capture complex dynamics.} To address this issue, we introduce Latent Temporal Discrepancy (LTD) as a motion prior to guide loss weighting. LTD measures frame-to-frame variation in the latent space, assigning larger penalties to regions with higher discrepancy while maintaining regular optimization for stable regions. This motion-aware strategy stabilizes training and enables the model to better reconstruct high-frequency dynamics. Extensive experiments on the general benchmark VBench and the motion-focused VMBench show consistent gains, with our method outperforming strong baselines by 3.31% on VBench and 3.58% on VMBench, achieving significant improvements in motion quality.

  </details>



- **Let's Roll a BiFTA: Bi-refinement for Fine-grained Text-visual Alignment in Vision-Language Models**  
  Yuhao Sun, Chengyi Cai, Jiacheng Zhang, Zesheng Ye, Xingliang Yuan, Feng Liu  
  _2026-01-28_ · https://arxiv.org/abs/2601.20419v1  
  <details><summary>Abstract</summary>

  Recent research has shown that aligning fine-grained text descriptions with localized image patches can significantly improve the zero-shot performance of pre-trained vision-language models (e.g., CLIP). However, we find that both fine-grained text descriptions and localized image patches often contain redundant information, making text-visual alignment less effective. In this paper, we tackle this issue from two perspectives: \emph{View Refinement} and \emph{Description refinement}, termed as \textit{\textbf{Bi}-refinement for \textbf{F}ine-grained \textbf{T}ext-visual \textbf{A}lignment} (BiFTA). \emph{View refinement} removes redundant image patches with high \emph{Intersection over Union} (IoU) ratios, resulting in more distinctive visual samples. \emph{Description refinement} removes redundant text descriptions with high pairwise cosine similarity, ensuring greater diversity in the remaining descriptions. BiFTA achieves superior zero-shot performance on 6 benchmark datasets for both ViT-based and ResNet-based CLIP, justifying the necessity to remove redundant information in visual-text alignment.

  </details>



- **RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification**  
  Xinyan Chen, Qinchun Li, Ruiqin Ma, Jiaqi Bai, Li Yi, Jianfei Yang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20377v1  
  <details><summary>Abstract</summary>

  Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.

  </details>



- **Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models**  
  Zengbin Wang, Xuecai Hu, Yong Wang, Feng Xiong, Man Zhang, Xiangxiang Chu  
  _2026-01-28_ · https://arxiv.org/abs/2601.20354v1  
  <details><summary>Abstract</summary>

  Text-to-image (T2I) models have achieved remarkable success in generating high-fidelity images, but they often fail in handling complex spatial relationships, e.g., spatial perception, reasoning, or interaction. These critical aspects are largely overlooked by current benchmarks due to their short or information-sparse prompt design. In this paper, we introduce SpatialGenEval, a new benchmark designed to systematically evaluate the spatial intelligence of T2I models, covering two key aspects: (1) SpatialGenEval involves 1,230 long, information-dense prompts across 25 real-world scenes. Each prompt integrates 10 spatial sub-domains and corresponding 10 multi-choice question-answer pairs, ranging from object position and layout to occlusion and causality. Our extensive evaluation of 21 state-of-the-art models reveals that higher-order spatial reasoning remains a primary bottleneck. (2) To demonstrate that the utility of our information-dense design goes beyond simple evaluation, we also construct the SpatialT2I dataset. It contains 15,400 text-image pairs with rewritten prompts to ensure image consistency while preserving information density. Fine-tuned results on current foundation models (i.e., Stable Diffusion-XL, Uniworld-V1, OmniGen2) yield consistent performance gains (+4.2%, +5.7%, +4.4%) and more realistic effects in spatial relations, highlighting a data-centric paradigm to achieve spatial intelligence in T2I models.

  </details>



- **PalmBridge: A Plug-and-Play Feature Alignment Framework for Open-Set Palmprint Verification**  
  Chenke Zhang, Ziyuan Yang, Licheng Yan, Shuyi Li, Andrew Beng Jin Teoh, Bob Zhang, Yi Zhang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20351v1  
  <details><summary>Abstract</summary>

  Palmprint recognition is widely used in biometric systems, yet real-world performance often degrades due to feature distribution shifts caused by heterogeneous deployment conditions. Most deep palmprint models assume a closed and stationary distribution, leading to overfitting to dataset-specific textures rather than learning domain-invariant representations. Although data augmentation is commonly used to mitigate this issue, it assumes augmented samples can approximate the target deployment distribution, an assumption that often fails under significant domain mismatch. To address this limitation, we propose PalmBridge, a plug-and-play feature-space alignment framework for open-set palmprint verification based on vector quantization. Rather than relying solely on data-level augmentation, PalmBridge learns a compact set of representative vectors directly from training features. During enrollment and verification, each feature vector is mapped to its nearest representative vector under a minimum-distance criterion, and the mapped vector is then blended with the original vector. This design suppresses nuisance variation induced by domain shifts while retaining discriminative identity cues. The representative vectors are jointly optimized with the backbone network using task supervision, a feature-consistency objective, and an orthogonality regularization term to form a stable and well-structured shared embedding space. Furthermore, we analyze feature-to-representative mappings via assignment consistency and collision rate to assess model's sensitivity to blending weights. Experiments on multiple palmprint datasets and backbone architectures show that PalmBridge consistently reduces EER in intra-dataset open-set evaluation and improves cross-dataset generalization with negligible to modest runtime overhead.

  </details>



- **Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation**  
  Yuzhe Huang, Pei Lin, Wanlin Li, Daohan Li, Jiajun Li, Jiaming Jiang, Chenxi Xiao, Ziyuan Jiao  
  _2026-01-28_ · https://arxiv.org/abs/2601.20321v1  
  <details><summary>Abstract</summary>

  Vision-Language-Action (VLA) models have recently emerged as powerful generalists for robotic manipulation. However, due to their predominant reliance on visual modalities, they fundamentally lack the physical intuition required for contact-rich tasks that require precise force regulation and physical reasoning. Existing attempts to incorporate vision-based tactile sensing into VLA models typically treat tactile inputs as auxiliary visual textures, thereby overlooking the underlying correlation between surface deformation and interaction dynamics. To bridge this gap, we propose a paradigm shift from tactile-vision alignment to tactile-force alignment. Here, we introduce TaF-VLA, a framework that explicitly grounds high-dimensional tactile observations in physical interaction forces. To facilitate this, we develop an automated tactile-force data acquisition device and curate the TaF-Dataset, comprising over 10 million synchronized tactile observations, 6-axis force/torque, and matrix force map. To align sequential tactile observations with interaction forces, the central component of our approach is the Tactile-Force Adapter (TaF-Adapter), a tactile sensor encoder that extracts discretized latent information for encoding tactile observations. This mechanism ensures that the learned representations capture history-dependent, noise-insensitive physical dynamics rather than static visual textures. Finally, we integrate this force-aligned encoder into a VLA backbone. Extensive real-world experiments demonstrate that TaF-VLA policy significantly outperforms state-of-the-art tactile-vision-aligned and vision-only baselines on contact-rich tasks, verifying its ability to achieve robust, force-aware manipulation through cross-modal physical reasoning.

  </details>



- **CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting**  
  Jiyuan Xu, Wenyu Zhang, Xin Jing, Shuai Chen, Shuai Zhang, Jiahao Nie  
  _2026-01-28_ · https://arxiv.org/abs/2601.20318v1  
  <details><summary>Abstract</summary>

  Current methods for multivariate time series forecasting can be classified into channel-dependent and channel-independent models. Channel-dependent models learn cross-channel features but often overfit the channel ordering, which hampers adaptation when channels are added or reordered. Channel-independent models treat each channel in isolation to increase flexibility, yet this neglects inter-channel dependencies and limits performance. To address these limitations, we propose \textbf{CPiRi}, a \textbf{channel permutation invariant (CPI)} framework that infers cross-channel structure from data rather than memorizing a fixed ordering, enabling deployment in settings with structural and distributional co-drift without retraining. CPiRi couples \textbf{spatio-temporal decoupling architecture} with \textbf{permutation-invariant regularization training strategy}: a frozen pretrained temporal encoder extracts high-quality temporal features, a lightweight spatial module learns content-driven inter-channel relations, while a channel shuffling strategy enforces CPI during training. We further \textbf{ground CPiRi in theory} by analyzing permutation equivariance in multivariate time series forecasting. Experiments on multiple benchmarks show state-of-the-art results. CPiRi remains stable when channel orders are shuffled and exhibits strong \textbf{inductive generalization} to unseen channels even when trained on \textbf{only half} of the channels, while maintaining \textbf{practical efficiency} on large-scale datasets. The source code is released at https://github.com/JasonStraka/CPiRi.

  </details>



- **Bridging the Applicator Gap with Data-Doping:Dual-Domain Learning for Precise Bladder Segmentation in CT-Guided Brachytherapy**  
  Suresh Das, Siladittya Manna, Sayantari Ghosh  
  _2026-01-28_ · https://arxiv.org/abs/2601.20302v1  
  <details><summary>Abstract</summary>

  Performance degradation due to covariate shift remains a major challenge for deep learning models in medical image segmentation. An open question is whether samples from a shifted distribution can effectively support learning when combined with limited target domain data. We investigate this problem in the context of bladder segmentation in CT guided gynecological brachytherapy, a critical task for accurate dose optimization and organ at risk sparing. While CT scans without brachytherapy applicators (no applicator: NA) are widely available, scans with applicators inserted (with applicator: WA) are scarce and exhibit substantial anatomical deformation and imaging artifacts, making automated segmentation particularly difficult. We propose a dual domain learning strategy that integrates NA and WA CT data to improve robustness and generalizability under covariate shift. Using a curated assorted dataset, we show that NA data alone fail to capture the anatomical and artifact related characteristics of WA images. However, introducing a modest proportion of WA data into a predominantly NA training set leads to significant performance improvements. Through systematic experiments across axial, coronal, and sagittal planes using multiple deep learning architectures, we demonstrate that doping only 10 to 30 percent WA data achieves segmentation performance comparable to models trained exclusively on WA data. The proposed approach attains Dice similarity coefficients of up to 0.94 and Intersection over Union scores of up to 0.92, indicating effective domain adaptation and improved clinical reliability. This study highlights the value of integrating anatomically similar but distribution shifted datasets to overcome data scarcity and enhance deep learning based segmentation for brachytherapy treatment planning.

  </details>



- **Artifact-Aware Evaluation for High-Quality Video Generation**  
  Chen Zhu, Jiashu Zhu, Yanxun Li, Meiqi Wu, Bingze Song, Chubin Chen, Jiahong Wu, Xiangxiang Chu, Yangang Wang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20297v1  
  <details><summary>Abstract</summary>

  With the rapid advancement of video generation techniques, evaluating and auditing generated videos has become increasingly crucial. Existing approaches typically offer coarse video quality scores, lacking detailed localization and categorization of specific artifacts. In this work, we introduce a comprehensive evaluation protocol focusing on three key aspects affecting human perception: Appearance, Motion, and Camera. We define these axes through a taxonomy of 10 prevalent artifact categories reflecting common generative failures observed in video generation. To enable robust artifact detection and categorization, we introduce GenVID, a large-scale dataset of 80k videos generated by various state-of-the-art video generation models, each carefully annotated for the defined artifact categories. Leveraging GenVID, we develop DVAR, a Dense Video Artifact Recognition framework for fine-grained identification and classification of generative artifacts. Extensive experiments show that our approach significantly improves artifact detection accuracy and enables effective filtering of low-quality content.

  </details>



- **TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement**  
  Wanjun Jia, Kang Li, Fan Yang, Mengfei Duan, Wenrui Chen, Yiming Jiang, Hui Zhang, Kailun Yang, Zhiyong Li, Yaonan Wang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20208v1  
  <details><summary>Abstract</summary>

  The central challenge in robotic manipulation of deformable objects lies in aligning high-level semantic instructions with physical interaction points under complex appearance and texture variations. Due to near-infinite degrees of freedom, complex dynamics, and heterogeneous patterns, existing vision-based affordance prediction methods often suffer from boundary overflow and fragmented functional regions. To address these issues, we propose TRACER, a Texture-Robust Affordance Chain-of-thought with dEformable-object Refinement framework, which establishes a cross-hierarchical mapping from hierarchical semantic reasoning to appearance-robust and physically consistent functional region refinement. Specifically, a Tree-structured Affordance Chain-of-Thought (TA-CoT) is formulated to decompose high-level task intentions into hierarchical sub-task semantics, providing consistent guidance across various execution stages. To ensure spatial integrity, a Spatial-Constrained Boundary Refinement (SCBR) mechanism is introduced to suppress prediction spillover, guiding the perceptual response to converge toward authentic interaction manifolds. Furthermore, an Interactive Convergence Refinement Flow (ICRF) is developed to aggregate discrete pixels corrupted by appearance noise, significantly enhancing the spatial continuity and physical plausibility of the identified functional regions. Extensive experiments conducted on the Fine-AGDDO15 dataset and a real-world robotic platform demonstrate that TRACER significantly improves affordance grounding precision across diverse textures and patterns inherent to deformable objects. More importantly, it enhances the success rate of long-horizon tasks, effectively bridging the gap between high-level semantic reasoning and low-level physical execution. The source code and dataset will be made publicly available at https://github.com/Dikay1/TRACER.

  </details>



- **TeleStyle: Content-Preserving Style Transfer in Images and Videos**  
  Shiwen Zhang, Xiaoyan Yang, Bojia Zi, Haibin Huang, Chi Zhang, Xuelong Li  
  _2026-01-28_ · https://arxiv.org/abs/2601.20175v1  
  <details><summary>Abstract</summary>

  Content-preserving style transfer, generating stylized outputs based on content and style references, remains a significant challenge for Diffusion Transformers (DiTs) due to the inherent entanglement of content and style features in their internal representations. In this technical report, we present TeleStyle, a lightweight yet effective model for both image and video stylization. Built upon Qwen-Image-Edit, TeleStyle leverages the base model's robust capabilities in content preservation and style customization. To facilitate effective training, we curated a high-quality dataset of distinct specific styles and further synthesized triplets using thousands of diverse, in-the-wild style categories. We introduce a Curriculum Continual Learning framework to train TeleStyle on this hybrid dataset of clean (curated) and noisy (synthetic) triplets. This approach enables the model to generalize to unseen styles without compromising precise content fidelity. Additionally, we introduce a video-to-video stylization module to enhance temporal consistency and visual quality. TeleStyle achieves state-of-the-art performance across three core evaluation metrics: style similarity, content consistency, and aesthetic quality. Code and pre-trained models are available at https://github.com/Tele-AI/TeleStyle

  </details>



- **Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing**  
  Zhuchenyang Liu, Ziyu Hu, Yao Zhang, Yu Xiao  
  _2026-01-27_ · https://arxiv.org/abs/2601.20107v1  
  <details><summary>Abstract</summary>

  Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive index vector size overheads. Training-free pruning solutions (e.g., EOS-attention based methods) can reduce index vector size by approximately 60% without model adaptation, but often underperform random selection in high-compression scenarios (> 80%). Prior research (e.g., Light-ColPali) attributes this to the conclusion that visual token importance is inherently query-dependent, thereby questioning the feasibility of training-free pruning. In this work, we propose Structural Anchor Pruning (SAP), a training-free pruning method that identifies key visual patches from middle layers to achieve high performance compression. We also introduce Oracle Score Retention (OSR) protocol to evaluate how layer-wise information affects compression efficiency. Evaluations on the ViDoRe benchmark demonstrate that SAP reduces index vectors by over 90% while maintaining robust retrieval fidelity, providing a highly scalable solution for Visual RAG. Furthermore, our OSR-based analysis reveals that semantic structural anchor patches persist in the middle layers, unlike traditional pruning solutions that focus on the final layer where structural signals dissipate.

  </details>



- **NucFuseRank: Dataset Fusion and Performance Ranking for Nuclei Instance Segmentation**  
  Nima Torbati, Anastasia Meshcheryakova, Ramona Woitek, Sepideh Hatamikia, Diana Mechtcheriakova, Amirreza Mahbod  
  _2026-01-27_ · https://arxiv.org/abs/2601.20104v1  
  <details><summary>Abstract</summary>

  Nuclei instance segmentation in hematoxylin and eosin (H&E)-stained images plays an important role in automated histological image analysis, with various applications in downstream tasks. While several machine learning and deep learning approaches have been proposed for nuclei instance segmentation, most research in this field focuses on developing new segmentation algorithms and benchmarking them on a limited number of arbitrarily selected public datasets. In this work, rather than focusing on model development, we focused on the datasets used for this task. Based on an extensive literature review, we identified manually annotated, publicly available datasets of H&E-stained images for nuclei instance segmentation and standardized them into a unified input and annotation format. Using two state-of-the-art segmentation models, one based on convolutional neural networks (CNNs) and one based on a hybrid CNN and vision transformer architecture, we systematically evaluated and ranked these datasets based on their nuclei instance segmentation performance. Furthermore, we proposed a unified test set (NucFuse-test) for fair cross-dataset evaluation and a unified training set (NucFuse-train) for improved segmentation performance by merging images from multiple datasets. By evaluating and ranking the datasets, performing comprehensive analyses, generating fused datasets, conducting external validation, and making our implementation publicly available, we provided a new benchmark for training, testing, and evaluating nuclei instance segmentation models on H&E-stained histological images.

  </details>



- **Size Matters: Reconstructing Real-Scale 3D Models from Monocular Images for Food Portion Estimation**  
  Gautham Vinod, Bruce Coburn, Siddeshwar Raghavan, Jiangpeng He, Fengqing Zhu  
  _2026-01-27_ · https://arxiv.org/abs/2601.20051v1  
  <details><summary>Abstract</summary>

  The rise of chronic diseases related to diet, such as obesity and diabetes, emphasizes the need for accurate monitoring of food intake. While AI-driven dietary assessment has made strides in recent years, the ill-posed nature of recovering size (portion) information from monocular images for accurate estimation of ``how much did you eat?'' is a pressing challenge. Some 3D reconstruction methods have achieved impressive geometric reconstruction but fail to recover the crucial real-world scale of the reconstructed object, limiting its usage in precision nutrition. In this paper, we bridge the gap between 3D computer vision and digital health by proposing a method that recovers a true-to-scale 3D reconstructed object from a monocular image. Our approach leverages rich visual features extracted from models trained on large-scale datasets to estimate the scale of the reconstructed object. This learned scale enables us to convert single-view 3D reconstructions into true-to-life, physically meaningful models. Extensive experiments and ablation studies on two publicly available datasets show that our method consistently outperforms existing techniques, achieving nearly a 30% reduction in mean absolute volume-estimation error, showcasing its potential to enhance the domain of precision nutrition. Code: https://gitlab.com/viper-purdue/size-matters

  </details>



- **DuwatBench: Bridging Language and Visual Heritage through an Arabic Calligraphy Benchmark for Multimodal Understanding**  
  Shubham Patle, Sara Ghaboura, Hania Tariq, Mohammad Usman Khan, Omkar Thawakar, Rao Muhammad Anwer, Salman Khan  
  _2026-01-27_ · https://arxiv.org/abs/2601.19898v1  
  <details><summary>Abstract</summary>

  Arabic calligraphy represents one of the richest visual traditions of the Arabic language, blending linguistic meaning with artistic form. Although multimodal models have advanced across languages, their ability to process Arabic script, especially in artistic and stylized calligraphic forms, remains largely unexplored. To address this gap, we present DuwatBench, a benchmark of 1,272 curated samples containing about 1,475 unique words across six classical and modern calligraphic styles, each paired with sentence-level detection annotations. The dataset reflects real-world challenges in Arabic writing, such as complex stroke patterns, dense ligatures, and stylistic variations that often challenge standard text recognition systems. Using DuwatBench, we evaluated 13 leading Arabic and multilingual multimodal models and showed that while they perform well on clean text, they struggle with calligraphic variation, artistic distortions, and precise visual-text alignment. By publicly releasing DuwatBench and its annotations, we aim to advance culturally grounded multimodal research, foster fair inclusion of the Arabic language and visual heritage in AI systems, and support continued progress in this area. Our dataset (https://huggingface.co/datasets/MBZUAI/DuwatBench) and evaluation suit (https://github.com/mbzuai-oryx/DuwatBench) are publicly available.

  </details>



- **VGGT-SLAM 2.0: Real time Dense Feed-forward Scene Reconstruction**  
  Dominic Maggio, Luca Carlone  
  _2026-01-27_ · https://arxiv.org/abs/2601.19887v1  
  <details><summary>Abstract</summary>

  We present VGGT-SLAM 2.0, a real time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT. Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics. Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures. Finally, we conduct a suite of experiments which includes showing VGGT-SLAM 2.0 can easily be adapted for open-set object detection and demonstrating real time performance while running online onboard a ground robot using a Jetson Thor. We also test in environments ranging from cluttered indoor apartments and office scenes to a 4,200 square foot barn, and we also demonstrate VGGT-SLAM 2.0 achieves the highest accuracy on the TUM dataset with about 23 percent less pose error than VGGT-SLAM. Code will be released upon publication.

  </details>



- **Information-Theoretic Detection of Bimanual Interactions for Dual-Arm Robot Plan Generation**  
  Elena Merlo, Marta Lagomarsino, Arash Ajoudani  
  _2026-01-27_ · https://arxiv.org/abs/2601.19832v1  
  <details><summary>Abstract</summary>

  Programming by demonstration is a strategy to simplify the robot programming process for non-experts via human demonstrations. However, its adoption for bimanual tasks is an underexplored problem due to the complexity of hand coordination, which also hinders data recording. This paper presents a novel one-shot method for processing a single RGB video of a bimanual task demonstration to generate an execution plan for a dual-arm robotic system. To detect hand coordination policies, we apply Shannon's information theory to analyze the information flow between scene elements and leverage scene graph properties. The generated plan is a modular behavior tree that assumes different structures based on the desired arms coordination. We validated the effectiveness of this framework through multiple subject video demonstrations, which we collected and made open-source, and exploiting data from an external, publicly available dataset. Comparisons with existing methods revealed significant improvements in generating a centralized execution plan for coordinating two-arm systems.

  </details>



- **Diffusion for De-Occlusion: Accessory-Aware Diffusion Inpainting for Robust Ear Biometric Recognition**  
  Deeksha Arun, Kevin W. Bowyer, Patrick Flynn  
  _2026-01-27_ · https://arxiv.org/abs/2601.19795v1  
  <details><summary>Abstract</summary>

  Ear occlusions (arising from the presence of ear accessories such as earrings and earphones) can negatively impact performance in ear-based biometric recognition systems, especially in unconstrained imaging circumstances. In this study, we assess the effectiveness of a diffusion-based ear inpainting technique as a pre-processing aid to mitigate the issues of ear accessory occlusions in transformer-based ear recognition systems. Given an input ear image and an automatically derived accessory mask, the inpainting model reconstructs clean and anatomically plausible ear regions by synthesizing missing pixels while preserving local geometric coherence along key ear structures, including the helix, antihelix, concha, and lobule. We evaluate the effectiveness of this pre-processing aid in transformer-based recognition systems for several vision transformer models and different patch sizes for a range of benchmark datasets. Experiments show that diffusion-based inpainting can be a useful pre-processing aid to alleviate ear accessory occlusions to improve overall recognition performance.

  </details>



- **WaterClear-GS: Optical-Aware Gaussian Splatting for Underwater Reconstruction and Restoration**  
  Xinrui Zhang, Yufeng Wang, Shuangkang Fang, Zesheng Wang, Dacheng Qi, Wenrui Ding  
  _2026-01-27_ · https://arxiv.org/abs/2601.19753v1  
  <details><summary>Abstract</summary>

  Underwater 3D reconstruction and appearance restoration are hindered by the complex optical properties of water, such as wavelength-dependent attenuation and scattering. Existing Neural Radiance Fields (NeRF)-based methods struggle with slow rendering speeds and suboptimal color restoration, while 3D Gaussian Splatting (3DGS) inherently lacks the capability to model complex volumetric scattering effects. To address these issues, we introduce WaterClear-GS, the first pure 3DGS-based framework that explicitly integrates underwater optical properties of local attenuation and scattering into Gaussian primitives, eliminating the need for an auxiliary medium network. Our method employs a dual-branch optimization strategy to ensure underwater photometric consistency while naturally recovering water-free appearances. This strategy is enhanced by depth-guided geometry regularization and perception-driven image loss, together with exposure constraints, spatially-adaptive regularization, and physically guided spectral regularization, which collectively enforce local 3D coherence and maintain natural visual perception. Experiments on standard benchmarks and our newly collected dataset demonstrate that WaterClear-GS achieves outstanding performance on both novel view synthesis (NVS) and underwater image restoration (UIR) tasks, while maintaining real-time rendering. The code will be available at https://buaaxrzhang.github.io/WaterClear-GS/.

  </details>



- **Benchmarking Multimodal Large Language Models for Missing Modality Completion in Product Catalogues**  
  Junchen Fu, Wenhao Deng, Kaiwen Zheng, Ioannis Arapakis, Yu Ye, Yongxin Ni, Joemon M. Jose, Xuri Ge  
  _2026-01-27_ · https://arxiv.org/abs/2601.19750v2  
  <details><summary>Abstract</summary>

  Missing-modality information on e-commerce platforms, such as absent product images or textual descriptions, often arises from annotation errors or incomplete metadata, impairing both product presentation and downstream applications such as recommendation systems. Motivated by the multimodal generative capabilities of recent Multimodal Large Language Models (MLLMs), this work investigates a fundamental yet underexplored question: can MLLMs generate missing modalities for products in e-commerce scenarios? We propose the Missing Modality Product Completion Benchmark (MMPCBench), which consists of two sub-benchmarks: a Content Quality Completion Benchmark and a Recommendation Benchmark. We further evaluate six state-of-the-art MLLMs from the Qwen2.5-VL and Gemma-3 model families across nine real-world e-commerce categories, focusing on image-to-text and text-to-image completion tasks. Experimental results show that while MLLMs can capture high-level semantics, they struggle with fine-grained word-level and pixel- or patch-level alignment. In addition, performance varies substantially across product categories and model scales, and we observe no trivial correlation between model size and performance, in contrast to trends commonly reported in mainstream benchmarks. We also explore Group Relative Policy Optimization (GRPO) to better align MLLMs with this task. GRPO improves image-to-text completion but does not yield gains for text-to-image completion. Overall, these findings expose the limitations of current MLLMs in real-world cross-modal generation and represent an early step toward more effective missing-modality product completion.

  </details>



- **Interpretable and backpropagation-free Green Learning for efficient multi-task echocardiographic segmentation and classification**  
  Jyun-Ping Kao, Jiaxing Yang, C. -C. Jay Kuo, Jonghye Woo  
  _2026-01-27_ · https://arxiv.org/abs/2601.19743v1  
  <details><summary>Abstract</summary>

  Echocardiography is a cornerstone for managing heart failure (HF), with Left Ventricular Ejection Fraction (LVEF) being a critical metric for guiding therapy. However, manual LVEF assessment suffers from high inter-observer variability, while existing Deep Learning (DL) models are often computationally intensive and data-hungry "black boxes" that impede clinical trust and adoption. Here, we propose a backpropagation-free multi-task Green Learning (MTGL) framework that performs simultaneous Left Ventricle (LV) segmentation and LVEF classification. Our framework integrates an unsupervised VoxelHop encoder for hierarchical spatio-temporal feature extraction with a multi-level regression decoder and an XG-Boost classifier. On the EchoNet-Dynamic dataset, our MTGL model achieves state-of-the-art classification and segmentation performance, attaining a classification accuracy of 94.3% and a Dice Similarity Coefficient (DSC) of 0.912, significantly outperforming several advanced 3D DL models. Crucially, our model achieves this with over an order of magnitude fewer parameters, demonstrating exceptional computational efficiency. This work demonstrates that the GL paradigm can deliver highly accurate, efficient, and interpretable solutions for complex medical image analysis, paving the way for more sustainable and trustworthy artificial intelligence in clinical practice.

  </details>



- **A new Image Similarity Metric for a Perceptual and Transparent Geometric and Chromatic Assessment**  
  Antonio Di Marino, Vincenzo Bevilacqua, Emanuel Di Nardo, Angelo Ciaramella, Ivanoe De Falco, Giovanna Sannino  
  _2026-01-27_ · https://arxiv.org/abs/2601.19680v1  
  <details><summary>Abstract</summary>

  In the literature, several studies have shown that state-of-the-art image similarity metrics are not perceptual metrics; moreover, they have difficulty evaluating images, especially when texture distortion is also present. In this work, we propose a new perceptual metric composed of two terms. The first term evaluates the dissimilarity between the textures of two images using Earth Mover's Distance. The second term evaluates the chromatic dissimilarity between two images in the Oklab perceptual color space. We evaluated the performance of our metric on a non-traditional dataset, called Berkeley-Adobe Perceptual Patch Similarity, which contains a wide range of complex distortions in shapes and colors. We have shown that our metric outperforms the state of the art, especially when images contain shape distortions, confirming also its greater perceptiveness. Furthermore, although deep black-box metrics could be very accurate, they only provide similarity scores between two images, without explaining their main differences and similarities. Our metric, on the other hand, provides visual explanations to support the calculated score, making the similarity assessment transparent and justified.

  </details>



- **Towards Governance-Oriented Low-Altitude Intelligence: A Management-Centric Multi-Modal Benchmark With Implicitly Coordinated Vision-Language Reasoning Framework**  
  Hao Chang, Zhihui Wang, Lingxiang Wu, Peijin Wang, Wenhui Diao, Jinqiao Wang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19640v1  
  <details><summary>Abstract</summary>

  Low-altitude vision systems are becoming a critical infrastructure for smart city governance. However, existing object-centric perception paradigms and loosely coupled vision-language pipelines are still difficult to support management-oriented anomaly understanding required in real-world urban governance. To bridge this gap, we introduce GovLA-10K, the first management-oriented multi-modal benchmark for low-altitude intelligence, along with GovLA-Reasoner, a unified vision-language reasoning framework tailored for governance-aware aerial perception. Unlike existing studies that aim to exhaustively annotate all visible objects, GovLA-10K is deliberately designed around functionally salient targets that directly correspond to practical management needs, and further provides actionable management suggestions grounded in these observations. To effectively coordinate the fine-grained visual grounding with high-level contextual language reasoning, GovLA-Reasoner introduces an efficient feature adapter that implicitly coordinates discriminative representation sharing between the visual detector and the large language model (LLM). Extensive experiments show that our method significantly improves performance while avoiding the need of fine-tuning for any task-specific individual components. We believe our work offers a new perspective and foundation for future studies on management-aware low-altitude vision-language systems.

  </details>



- **The role of self-supervised pretraining in differentially private medical image analysis**  
  Soroosh Tayebi Arasteh, Mina Farajiamiri, Mahshad Lotfinia, Behrus Hinrichs-Puladi, Jonas Bienzeisler, Mohamed Alhaskir, Mirabela Rusu, Christiane Kuhl, Sven Nebelung, Daniel Truhn  
  _2026-01-27_ · https://arxiv.org/abs/2601.19618v1  
  <details><summary>Abstract</summary>

  Differential privacy (DP) provides formal protection for sensitive data but typically incurs substantial losses in diagnostic performance. Model initialization has emerged as a critical factor in mitigating this degradation, yet the role of modern self-supervised learning under full-model DP remains poorly understood. Here, we present a large-scale evaluation of initialization strategies for differentially private medical image analysis, using chest radiograph classification as a representative benchmark with more than 800,000 images. Using state-of-the-art ConvNeXt models trained with DP-SGD across realistic privacy regimes, we compare non-domain-specific supervised ImageNet initialization, non-domain-specific self-supervised DINOv3 initialization, and domain-specific supervised pretraining on MIMIC-CXR, the largest publicly available chest radiograph dataset. Evaluations are conducted across five external datasets spanning diverse institutions and acquisition settings. We show that DINOv3 initialization consistently improves diagnostic utility relative to ImageNet initialization under DP, but remains inferior to domain-specific supervised pretraining, which achieves performance closest to non-private baselines. We further demonstrate that initialization choice strongly influences demographic fairness, cross-dataset generalization, and robustness to data scale and model capacity under privacy constraints. The results establish initialization strategy as a central determinant of utility, fairness, and generalization in differentially private medical imaging.

  </details>



- **Localized Latent Editing for Dose-Response Modeling in Botulinum Toxin Injection Planning**  
  Estèphe Arnaud, Mohamed Daoudi, Pierre Guerreschi  
  _2026-01-27_ · https://arxiv.org/abs/2601.19593v1  
  <details><summary>Abstract</summary>

  Botulinum toxin (Botox) injections are the gold standard for managing facial asymmetry and aesthetic rejuvenation, yet determining the optimal dosage remains largely intuitive, often leading to suboptimal outcomes. We propose a localized latent editing framework that simulates Botulinum Toxin injection effects for injection planning through dose-response modeling. Our key contribution is a Region-Specific Latent Axis Discovery method that learns localized muscle relaxation trajectories in StyleGAN2's latent space, enabling precise control over specific facial regions without global side effects. By correlating these localized latent trajectories with injected toxin units, we learn a predictive dose-response model. We rigorously compare two approaches: direct metric regression versus image-based generative simulation on a clinical dataset of N=360 images from 46 patients. On a hold-out test set, our framework demonstrates moderate-to-strong structural correlations for geometric asymmetry metrics, confirming that the generative model correctly captures the direction of morphological changes. While biological variability limits absolute precision, we introduce a hybrid "Human-in-the-Loop" workflow where clinicians interactively refine simulations, bridging the gap between pathological reconstruction and cosmetic planning.

  </details>



- **ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving**  
  Yujin Wang, Yutong Zheng, Wenxian Fan, Tianyi Wang, Hongqing Chu, Daxin Tian, Bingzhao Gao, Jianqiang Wang, Hong Chen  
  _2026-01-27_ · https://arxiv.org/abs/2601.19582v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts.

  </details>



- **The S3LI Vulcano Dataset: A Dataset for Multi-Modal SLAM in Unstructured Planetary Environments**  
  Riccardo Giubilato, Marcus Gerhard Müller, Marco Sewtz, Laura Alejandra Encinar Gonzalez, John Folkesson, Rudolph Triebel  
  _2026-01-27_ · https://arxiv.org/abs/2601.19557v1  
  <details><summary>Abstract</summary>

  We release the S3LI Vulcano dataset, a multi-modal dataset towards development and benchmarking of Simultaneous Localization and Mapping (SLAM) and place recognition algorithms that rely on visual and LiDAR modalities. Several sequences are recorded on the volcanic island of Vulcano, from the Aeolian Islands in Sicily, Italy. The sequences provide users with data from a variety of environments, textures and terrains, including basaltic or iron-rich rocks, geological formations from old lava channels, as well as dry vegetation and water. The data (rmc.dlr.de/s3li_dataset) is accompanied by an open source toolkit (github.com/DLR-RM/s3li-toolkit) providing tools for generating ground truth poses as well as preparation of labelled samples for place recognition tasks.

  </details>



- **A Non-Invasive 3D Gait Analysis Framework for Quantifying Psychomotor Retardation in Major Depressive Disorder**  
  Fouad Boutaleb, Emery Pierson, Mohamed Daoudi, Clémence Nineuil, Ali Amad, Fabien D'Hondt  
  _2026-01-27_ · https://arxiv.org/abs/2601.19526v1  
  <details><summary>Abstract</summary>

  Predicting the status of Major Depressive Disorder (MDD) from objective, non-invasive methods is an active research field. Yet, extracting automatically objective, interpretable features for a detailed analysis of the patient state remains largely unexplored. Among MDD's symptoms, Psychomotor retardation (PMR) is a core item, yet its clinical assessment remains largely subjective. While 3D motion capture offers an objective alternative, its reliance on specialized hardware often precludes routine clinical use. In this paper, we propose a non-invasive computational framework that transforms monocular RGB video into clinically relevant 3D gait kinematics. Our pipeline uses Gravity-View Coordinates along with a novel trajectory-correction algorithm that leverages the closed-loop topology of our adapted Timed Up and Go (TUG) protocol to mitigate monocular depth errors. This novel pipeline enables the extraction of 297 explicit gait biomechanical biomarkers from a single camera capture. To address the challenges of small clinical datasets, we introduce a stability-based machine learning framework that identifies robust motor signatures while preventing overfitting. Validated on the CALYPSO dataset, our method achieves an 83.3% accuracy in detecting PMR and explains 64% of the variance in overall depression severity (R^2=0.64). Notably, our study reveals a strong link between reduced ankle propulsion and restricted pelvic mobility to the depressive motor phenotype. These results demonstrate that physical movement serves as a robust proxy for the cognitive state, offering a transparent and scalable tool for the objective monitoring of depression in standard clinical environments.

  </details>



- **ALRM: Agentic LLM for Robotic Manipulation**  
  Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid  
  _2026-01-27_ · https://arxiv.org/abs/2601.19510v1  
  <details><summary>Abstract</summary>

  Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.

  </details>



- **Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots**  
  Mehdi Heydari Shahna, Seyed Adel Alizadeh Kolagar, Jouni Mattila  
  _2026-01-27_ · https://arxiv.org/abs/2601.19499v1  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) can be highly effective at learning goal-reaching policies, but it typically does not provide formal guarantees that the goal will always be reached. A common approach to provide formal goal-reaching guarantees is to introduce a shielding mechanism that restricts the agent to actions that satisfy predefined safety constraints. The main challenge here is integrating this mechanism with RL so that learning and exploration remain effective without becoming overly conservative. Hence, this paper proposes an RL-based control framework that provides formal goal-reaching guarantees for wheeled mobile robots operating in unstructured environments. We first design a real-time RL policy with a set of 15 carefully defined reward terms. These rewards encourage the robot to reach both static and dynamic goals while generating sufficiently smooth command signals that comply with predefined safety specifications, which is critical in practice. Second, a Lyapunov-like stabilizer layer is integrated into the benchmark RL framework as a policy supervisor to formally strengthen the goal-reaching control while preserving meaningful exploration of the state action space. The proposed framework is suitable for real-time deployment in challenging environments, as it provides a formal guarantee of convergence to the intended goal states and compensates for uncertainties by generating real-time control signals based on the current state, while respecting real-world motion constraints. The experimental results show that the proposed Lyapunov-like stabilizer consistently improves the benchmark RL policies, boosting the goal-reaching rate from 84.6% to 99.0%, sharply reducing failures, and improving efficiency.

  </details>



- **Cortex-Grounded Diffusion Models for Brain Image Generation**  
  Fabian Bongratz, Yitong Li, Sama Elbaroudy, Christian Wachinger  
  _2026-01-27_ · https://arxiv.org/abs/2601.19498v1  
  <details><summary>Abstract</summary>

  Synthetic neuroimaging data can mitigate critical limitations of real-world datasets, including the scarcity of rare phenotypes, domain shifts across scanners, and insufficient longitudinal coverage. However, existing generative models largely rely on weak conditioning signals, such as labels or text, which lack anatomical grounding and often produce biologically implausible outputs. To this end, we introduce Cor2Vox, a cortex-grounded generative framework for brain magnetic resonance image (MRI) synthesis that ties image generation to continuous structural priors of the cerebral cortex. It leverages high-resolution cortical surfaces to guide a 3D shape-to-image Brownian bridge diffusion process, enabling topologically faithful synthesis and precise control over underlying anatomies. To support the generation of new, realistic brain shapes, we developed a large-scale statistical shape model of cortical morphology derived from over 33,000 UK Biobank scans. We validated the fidelity of Cor2Vox based on traditional image quality metrics, advanced cortical surface reconstruction, and whole-brain segmentation quality, outperforming many baseline methods. Across three applications, namely (i) anatomically consistent synthesis, (ii) simulation of progressive gray matter atrophy, and (iii) harmonization of in-house frontotemporal dementia scans with public datasets, Cor2Vox preserved fine-grained cortical morphology at the sub-voxel level, exhibiting remarkable robustness to variations in cortical geometry and disease phenotype without retraining.

  </details>



- **Dynamic Worlds, Dynamic Humans: Generating Virtual Human-Scene Interaction Motion in Dynamic Scenes**  
  Yin Wang, Zhiying Leng, Haitian Liu, Frederick W. B. Li, Mu Li, Xiaohui Liang  
  _2026-01-27_ · https://arxiv.org/abs/2601.19484v1  
  <details><summary>Abstract</summary>

  Scenes are continuously undergoing dynamic changes in the real world. However, existing human-scene interaction generation methods typically treat the scene as static, which deviates from reality. Inspired by world models, we introduce Dyn-HSI, the first cognitive architecture for dynamic human-scene interaction, which endows virtual humans with three humanoid components. (1)Vision (human eyes): we equip the virtual human with a Dynamic Scene-Aware Navigation, which continuously perceives changes in the surrounding environment and adaptively predicts the next waypoint. (2)Memory (human brain): we equip the virtual human with a Hierarchical Experience Memory, which stores and updates experiential data accumulated during training. This allows the model to leverage prior knowledge during inference for context-aware motion priming, thereby enhancing both motion quality and generalization. (3) Control (human body): we equip the virtual human with Human-Scene Interaction Diffusion Model, which generates high-fidelity interaction motions conditioned on multimodal inputs. To evaluate performance in dynamic scenes, we extend the existing static human-scene interaction datasets to construct a dynamic benchmark, Dyn-Scenes. We conduct extensive qualitative and quantitative experiments to validate Dyn-HSI, showing that our method consistently outperforms existing approaches and generates high-quality human-scene interaction motions in both static and dynamic settings.

  </details>


