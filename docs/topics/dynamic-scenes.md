# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-17 07:17 UTC_

Total papers shown: **12**


---

- **Evaluating Time Awareness and Cross-modal Active Perception of Large Models via 4D Escape Room Task**  
  Yurui Dong, Ziyue Wang, Shuyun Lu, Dairu Liu, Xuechen Liu, Fuwen Luo, Peng Li, Yang Liu  
  _2026-03-16_ · https://arxiv.org/abs/2603.15467v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have recently made rapid progress toward unified Omni models that integrate vision, language, and audio. However, existing environments largely focus on 2D or 3D visual context and vision-language tasks, offering limited support for temporally dependent auditory signals and selective cross-modal integration, where different modalities may provide complementary or interfering information, which are essential capabilities for realistic multimodal reasoning. As a result, whether models can actively coordinate modalities and reason under time-varying, irreversible conditions remains underexplored. To this end, we introduce \textbf{EscapeCraft-4D}, a customizable 4D environment for assessing selective cross-modal perception and time awareness in Omni models. It incorporates trigger-based auditory sources, temporally transient evidence, and location-dependent cues, requiring agents to perform spatio-temporal reasoning and proactive multimodal integration under time constraints. Building on this environment, we curate a benchmark to evaluate corresponding abilities across powerful models. Evaluation results suggest that models struggle with modality bias, and reveal significant gaps in current model's ability to integrate multiple modalities under time constraints. Further in-depth analysis uncovers how multiple modalities interact and jointly influence model decisions in complex multimodal reasoning environments.

  </details>



- **AnyCrowd: Instance-Isolated Identity-Pose Binding for Arbitrary Multi-Character Animation**  
  Zhenyu Xie, Ji Xia, Michael Kampffmeyer, Panwen Hu, Zehua Ma, Yujian Zheng, Jing Wang, Zheng Chong, Xujie Zhang, Xianhang Cheng, et al.  
  _2026-03-16_ · https://arxiv.org/abs/2603.15415v1  
  <details><summary>Abstract</summary>

  Controllable character animation has advanced rapidly in recent years, yet multi-character animation remains underexplored. As the number of characters grows, multi-character reference encoding becomes more susceptible to latent identity entanglement, resulting in identity bleeding and reduced controllability. Moreover, learning precise and spatio-temporally consistent correspondences between reference identities and driving pose sequences becomes increasingly challenging, often leading to identity-pose mis-binding and inconsistency in generated videos. To address these challenges, we propose AnyCrowd, a Diffusion Transformer (DiT)-based video generation framework capable of scaling to an arbitrary number of characters. Specifically, we first introduce an Instance-Isolated Latent Representation (IILR), which encodes character instances independently prior to DiT processing to prevent latent identity entanglement. Building on this disentangled representation, we further propose Tri-Stage Decoupled Attention (TSDA) to bind identities to driving poses by decomposing self-attention into: (i) instance-aware foreground attention, (ii) background-centric interaction, and (iii) global foreground-background coordination. Furthermore, to mitigate token ambiguity in overlapping regions, an Adaptive Gated Fusion (AGF) module is integrated within TSDA to predict identity-aware weights, effectively fusing competing token groups into identity-consistent representations...

  </details>



- **Flash-Unified: A Training-Free and Task-Aware Acceleration Framework for Native Unified Models**  
  Junlong Ke, Zichen Wen, Boxue Yang, Yantai Yang, Xuyang Liu, Chenfei Liao, Zhaorun Chen, Shaobo Wang, Linfeng Zhang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15271v1  
  <details><summary>Abstract</summary>

  Native unified multimodal models, which integrate both generative and understanding capabilities, face substantial computational overhead that hinders their real-world deployment. Existing acceleration techniques typically employ a static, monolithic strategy, ignoring the fundamental divergence in computational profiles between iterative generation tasks (e.g., image generation) and single-pass understanding tasks (e.g., VQA). In this work, we present the first systematic analysis of unified models, revealing pronounced parameter specialization, where distinct neuron sets are critical for each task. This implies that, at the parameter level, unified models have implicitly internalized separate inference pathways for generation and understanding within a single architecture. Based on these insights, we introduce a training-free and task-aware acceleration framework, FlashU, that tailors optimization to each task's demands. Across both tasks, we introduce Task-Specific Network Pruning and Dynamic Layer Skipping, aiming to eliminate inter-layer and task-specific redundancy. For visual generation, we implement a time-varying control signal for the guidance scale and a temporal approximation for the diffusion head via Diffusion Head Cache. For multimodal understanding, building upon the pruned model, we introduce Dynamic Token Pruning via a V-Norm Proxy to exploit the spatial redundancy of visual inputs. Extensive experiments on Show-o2 demonstrate that FlashU achieves 1.78$\times$ to 2.01$\times$ inference acceleration across both understanding and generation tasks while maintaining SOTA performance, outperforming competing unified models and validating our task-aware acceleration paradigm. Our code is publicly available at https://github.com/Rirayh/FlashU.

  </details>



- **MER-Bench: A Comprehensive Benchmark for Multimodal Meme Reappraisal**  
  Yiqi Nie, Fei Wang, Junjie Chen, Kun Li, Yudi Cai, Dan Guo, Chenglong Li, Meng Wang  
  _2026-03-16_ · https://arxiv.org/abs/2603.15020v1  
  <details><summary>Abstract</summary>

  Memes represent a tightly coupled, multimodal form of social expression, in which visual context and overlaid text jointly convey nuanced affect and commentary. Inspired by cognitive reappraisal in psychology, we introduce Meme Reappraisal, a novel multimodal generation task that aims to transform negatively framed memes into constructive ones while preserving their underlying scenario, entities, and structural layout. Unlike prior works on meme understanding or generation, Meme Reappraisal requires emotion-controllable, structure-preserving multimodal transformation under multiple semantic and stylistic constraints. To support this task, we construct MER-Bench, a benchmark of real-world memes with fine-grained multimodal annotations, including source and target emotions, positively rewritten meme text, visual editing specifications, and taxonomy labels covering visual type, sentiment polarity, and layout structure. We further propose a structured evaluation framework based on a multimodal large language model (MLLM)-as-a-Judge paradigm, decomposing performance into modality-level generation quality, affect controllability, structural fidelity, and global affective alignment. Extensive experiments across representative image-editing and multimodal-generation systems reveal substantial gaps in satisfying the constraints of structural preservation, semantic consistency, and affective transformation. We believe MER-Bench establishes a foundation for research on controllable meme editing and emotion-aware multimodal generation. Our code is available at: https://github.com/one-seven17/MER-Bench.

  </details>



- **$\text{F}^2\text{HDR}$: Two-Stage HDR Video Reconstruction via Flow Adapter and Physical Motion Modeling**  
  Huanjing Yue, Dawei Li, Shaoxiong Tu, Jingyu Yang  
  _2026-03-16_ · https://arxiv.org/abs/2603.14920v1  
  <details><summary>Abstract</summary>

  Reconstructing High Dynamic Range (HDR) videos from sequences of alternating-exposure Low Dynamic Range (LDR) frames remains highly challenging, especially under dynamic scenes where cross-exposure inconsistencies and complex motion make inter-frame alignment difficult, leading to ghosting and detail loss. Existing methods often suffer from inaccurate alignment, suboptimal feature aggregation, and degraded reconstruction quality in motion-dominated regions. To address these challenges, we propose $\text{F}^2\text{HDR}$, a two-stage HDR video reconstruction framework that robustly perceives inter-frame motion and restores fine details in complex dynamic scenarios. The proposed framework integrates a flow adapter that adapts generic optical flow for robust cross-exposure alignment, a physical motion modeling to identify salient motion regions, and a motion-aware refinement network that aggregates complementary information while removing ghosting and noise. Extensive experiments demonstrate that $\text{F}^2\text{HDR}$ achieves state-of-the-art performance on real-world HDR video benchmarks, producing ghost-free and high-fidelity results under large motion and exposure variations.

  </details>



- **E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction**  
  Yunsoo Kim, Changki Sung, Dasol Hong, Hyun Myung  
  _2026-03-16_ · https://arxiv.org/abs/2603.14684v1  
  <details><summary>Abstract</summary>

  The emergence of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS) has advanced novel view synthesis (NVS). These methods, however, require high-quality RGB inputs and accurate corresponding poses, limiting robustness under real-world conditions such as fast camera motion or adverse lighting. Event cameras, which capture brightness changes at each pixel with high temporal resolution and wide dynamic range, enable precise sensing of dynamic scenes and offer a promising solution. However, existing event-based NVS methods either assume known poses or rely on depth estimation models that are bounded by their initial observations, failing to generalize as the camera traverses previously unseen regions. We present E2EGS, a pose-free framework operating solely on event streams. Our key insight is that edge information provides rich structural cues essential for accurate trajectory estimation and high-quality NVS. To extract edges from noisy event streams, we exploit the distinct spatio-temporal characteristics of edges and non-edge regions. The event camera's movement induces consistent events along edges, while non-edge regions produce sparse noise. We leverage this through a patch-based temporal coherence analysis that measures local variance to extract edges while robustly suppressing noise. The extracted edges guide structure-aware Gaussian initialization and enable edge-weighted losses throughout initialization, tracking, and bundle adjustment. Extensive experiments on both synthetic and real datasets demonstrate that E2EGS achieves superior reconstruction quality and trajectory accuracy, establishing a fully pose-free paradigm for event-based 3D reconstruction.

  </details>



- **VisionCoach: Reinforcing Grounded Video Reasoning via Visual-Perception Prompting**  
  Daeun Lee, Shoubin Yu, Yue Zhang, Mohit Bansal  
  _2026-03-15_ · https://arxiv.org/abs/2603.14659v1  
  <details><summary>Abstract</summary>

  Video reasoning requires models to locate and track question-relevant evidence across frames. While reinforcement learning (RL) with verifiable rewards improves accuracy, it still struggles to achieve reliable spatio-temporal grounding during the reasoning process. Moreover, improving grounding typically relies on scaled training data or inference-time perception tools, which increases annotation cost or computational cost. To address this challenge, we propose VisonCoach, an input-adaptive RL framework that improves spatio-temporal grounding through visual prompting as training-time guidance. During RL training, visual prompts are selectively applied to challenging inputs to amplify question-relevant evidence and suppress distractors. The model then internalizes these improvements through self-distillation, enabling grounded reasoning directly on raw videos without visual prompting at inference. VisonCoach consists of two components: (1) Visual Prompt Selector, which predicts appropriate prompt types conditioned on the video and question, and (2) Spatio-Temporal Reasoner, optimized with RL under visual prompt guidance and object-aware grounding rewards that enforce object identity consistency and multi-region bounding-box overlap. Extensive experiments demonstrate that VisonCoach achieves state-of-the-art performance under comparable settings, across diverse video reasoning, video understanding, and temporal grounding benchmarks (V-STAR, VideoMME, World-Sense, VideoMMMU, PerceptionTest, and Charades-STA), while maintaining a single efficient inference pathway without external tools. Our results show that visual prompting during training improves grounded video reasoning, while self-distillation enables the model to internalize this ability without requiring prompts at inference time.

  </details>



- **Wi-Spike: A Low-power WiFi Human Multi-action Recognition Model with Spiking Neural Networks**  
  Nengbo Zhang, Yao Ying, Lu Wang, Kaishun Wu, Jieming Ma, Fei Luo  
  _2026-03-15_ · https://arxiv.org/abs/2603.14475v1  
  <details><summary>Abstract</summary>

  WiFi-based human action recognition (HAR) has gained significant attention due to its non-intrusive and privacy-preserving nature. However, most existing WiFi sensing models predominantly focus on improving recognition accuracy, while issues of power consumption and energy efficiency remain insufficiently discussed. In this work, we present Wi-Spike, a bio-inspired spiking neural network (SNN) framework for efficient and accurate action recognition using WiFi channel state information (CSI) signals. Specifically, leveraging the event-driven and low-power characteristics of SNNs, Wi-Spike introduces spiking convolutional layers for spatio-temporal feature extraction and a novel temporal attention mechanism to enhance discriminative representation. The extracted features are subsequently encoded and classified through spiking fully connected layers and a voting layer. Comprehensive experiments on three benchmark datasets (NTU-Fi-HAR, NTU-Fi-HumanID, and UT-HAR) demonstrate that Wi-Spike achieves competitive accuracy in single-action recognition and superior performance in multi-action recognition tasks. As for energy consumption, Wi-Spike reduces the energy cost by at least half compared with other methods, while still achieving 95.83% recognition accuracy in human activity recognition. More importantly, Wi-Spike establishes a new state-of-the-art in WiFi-based multi-action HAR, offering a promising solution for real-time, energy-efficient edge sensing applications.

  </details>



- **Uni-MDTrack: Learning Decoupled Memory and Dynamic States for Parameter-Efficient Visual Tracking in All Modality**  
  Wenrui Cai, Zhenyi Lu, Yuzhe Li, Yongchao Feng, Jinqing Zhang, Qingjie Liu, Yunhong Wang  
  _2026-03-15_ · https://arxiv.org/abs/2603.14452v1  
  <details><summary>Abstract</summary>

  With the advent of Transformer-based one-stream trackers that possess strong capability in inter-frame relation modeling, recent research has increasingly focused on how to introduce spatio-temporal context. However, most existing methods rely on a limited number of historical frames, which not only leads to insufficient utilization of the context, but also inevitably increases the length of input and incurs prohibitive computational overhead. Methods that query an external memory bank, on the other hand, suffer from inadequate fusion between the retrieved spatio-temporal features and the backbone. Moreover, using discrete historical frames as context overlooks the rich dynamics of the target. To address the issues, we propose Uni-MDTrack, which consists of two core components: Memory-Aware Compression Prompt (MCP) module and Dynamic State Fusion (DSF) module. MCP effectively compresses rich memory features into memory-aware prompt tokens, which deeply interact with the input throughout the entire backbone, significantly enhancing the performance while maintaining a stable computational load. DSF complements the discrete memory by capturing the continuous dynamic, progressively introducing the updated dynamic state features from shallow to deep layers, while also preserving high efficiency. Uni-MDTrack also supports unified tracking across RGB, RGB-D/T/E, and RGB-Language modalities. Experiments show that in Uni-MDTrack, training only the MCP, DSF, and prediction head, keeping the proportion of trainable parameters around 30%, yields substantial performance gains, achieves state-of-the-art results on 10 datasets spanning five modalities. Furthermore, both MCP and DSF exhibit excellent generality, functioning as plug-and-play components that can boost the performance of various baseline trackers, while significantly outperforming existing parameter-efficient training approaches.

  </details>



- **LoCAtion: Long-time Collaborative Attention Framework for High Dynamic Range Video Reconstruction**  
  Qianyu Zhang, Bolun Zheng, Lingyu Zhu, Aiai Huang, Zongpeng Li, Shiqi Wang  
  _2026-03-15_ · https://arxiv.org/abs/2603.14377v1  
  <details><summary>Abstract</summary>

  Prevailing High Dynamic Range (HDR) video reconstruction methods are fundamentally trapped in a fragile alignment-and-fusion paradigm. While explicit spatial alignment can successfully recover fine details in controlled environments, it becomes a severe bottleneck in unconstrained dynamic scenes. By forcing rigid alignment across unpredictable motions and varying exposures, these methods inevitably translate registration errors into severe ghosting artifacts and temporal flickering. In this paper, we rethink this conventional prerequisite. Recognizing that explicit alignment is inherently vulnerable to real-world complexities, we propose LoCAtion, a Long-time Collaborative Attention framework that reformulates HDR video generation from a fragile spatial warping task into a robust, alignment-free collaborative feature routing problem. Guided by this new formulation, our architecture explicitly decouples the highly entangled reconstruction task. Rather than struggling to rigidly warp neighboring frames, we anchor the scene on a continuous medium-exposure backbone and utilize collaborative attention to dynamically harvest and inject reliable irradiance cues from unaligned exposures. Furthermore, we introduce a learned global sequence solver. By leveraging bidirectional context and long-range temporal modeling, it propagates corrective signals and structural features across the entire sequence, inherently enforcing whole-video coherence and eliminating jitter. Extensive experiments demonstrate that LoCAtion achieves state-of-the-art visual quality and temporal stability, offering a highly competitive balance between accuracy and computational efficiency.

  </details>



- **4D Synchronized Fields: Motion-Language Gaussian Splatting for Temporal Scene Understanding**  
  Mohamed Rayan Barhdadi, Samir Abdaljalil, Rasul Khanbayov, Erchin Serpedin, Hasan Kurban  
  _2026-03-15_ · https://arxiv.org/abs/2603.14301v1  
  <details><summary>Abstract</summary>

  Current 4D representations decouple geometry, motion, and semantics: reconstruction methods discard interpretable motion structure; language-grounded methods attach semantics after motion is learned, blind to how objects move; and motion-aware methods encode dynamics as opaque per-point residuals without object-level organization. We propose 4D Synchronized Fields, a 4D Gaussian representation that learns object-factored motion in-loop during reconstruction and synchronizes language to the resulting kinematics through a per-object conditioned field. Each Gaussian trajectory is decomposed into shared object motion plus an implicit residual, and a kinematic-conditioned ridge map predicts temporal semantic variation, yielding a single representation in which reconstruction, motion, and semantics are structurally coupled and enabling open-vocabulary temporal queries that retrieve both objects and moments. On HyperNeRF, 4D Synchronized Fields achieves 28.52 dB mean PSNR, the highest among all language-grounded and motion-aware baselines, within 1.5 dB of reconstruction-only methods. On targeted temporal-state retrieval, the kinematic-conditioned field attains 0.884 mean accuracy, 0.815 mean vIoU, and 0.733 mean tIoU, surpassing 4D LangSplat (0.620, 0.433, and 0.439 respectively) and LangSplat (0.415, 0.304, and 0.262). Ablation confirms that kinematic conditioning is the primary driver, accounting for +0.45 tIoU over a static-embedding-only baseline. 4D Synchronized Fields is the only method that jointly exposes interpretable motion primitives and temporally grounded language fields from a single trained representation. Code will be released.

  </details>



- **Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions**  
  Zhiyang Lu, Wen Jiang, Tianren Wu, Zhichao Wang, Changwang Zhang, Siqi Shen, Ming Cheng  
  _2026-03-15_ · https://arxiv.org/abs/2603.14189v1  
  <details><summary>Abstract</summary>

  Gait recognition is an emerging biometric technology that enables non-intrusive and hard-to-spoof human identification. However, most existing methods are confined to short-range, unimodal settings and fail to generalize to long-range and cross-distance scenarios under real-world conditions. To address this gap, we present \textbf{LRGait}, the first LiDAR-Camera multimodal benchmark designed for robust long-range gait recognition across diverse outdoor distances and environments. We further propose \textbf{EMGaitNet}, an end-to-end framework tailored for long-range multimodal gait recognition. To bridge the modality gap between RGB images and point clouds, we introduce a semantic-guided fusion pipeline. A CLIP-based Semantic Mining (SeMi) module first extracts human body-part-aware semantic cues, which are then employed to align 2D and 3D features via a Semantic-Guided Alignment (SGA) module within a unified embedding space. A Symmetric Cross-Attention Fusion (SCAF) module hierarchically integrates visual contours and 3D geometric features, and a Spatio-Temporal (ST) module captures global gait dynamics. Extensive experiments on various gait datasets validate the effectiveness of our method.

  </details>


