# Datasets & Benchmarks (3D / Vision)

_Updated: 2026-03-15 07:12 UTC_

Total papers shown: **44**


---

- **MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning**  
  Haozhan Shen, Shilin Yan, Hongwei Xue, Shuaiqi Lu, Xiaojun Tang, Guannan Zhang, Tiancheng Zhao, Jianwei Yin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12266v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) are increasingly used to carry out visual workflows such as navigating GUIs, where the next step depends on verified visual compositional conditions (e.g., "if a permission dialog appears and the color of the interface is green, click Allow") and the process may branch or terminate early. Yet this capability remains under-evaluated: existing benchmarks focus on shallow-compositions or independent-constraints rather than deeply chained compositional conditionals. In this paper, we introduce MM-CondChain, a benchmark for visually grounded deep compositional reasoning. Each benchmark instance is organized as a multi-layer reasoning chain, where every layer contains a non-trivial compositional condition grounded in visual evidence and built from multiple objects, attributes, or relations. To answer correctly, an MLLM must perceive the image in detail, reason over multiple visual elements at each step, and follow the resulting execution path to the final outcome. To scalably construct such workflow-style data, we propose an agentic synthesis pipeline: a Planner orchestrates layer-by-layer generation of compositional conditions, while a Verifiable Programmatic Intermediate Representation (VPIR) ensures each layer's condition is mechanically verifiable. A Composer then assembles these verified layers into complete instructions. Using this pipeline, we construct benchmarks across three visual domains: natural images, data charts, and GUI trajectories. Experiments on a range of MLLMs show that even the strongest model attains only 53.33 Path F1, with sharp drops on hard negatives and as depth or predicate complexity grows, confirming that deep compositional reasoning remains a fundamental challenge.

  </details>



- **OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams**  
  Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, Weidi Xie  
  _2026-03-12_ · https://arxiv.org/abs/2603.12265v1  
  <details><summary>Abstract</summary>

  Modern visual agents require representations that are general, causal, and physically structured to operate in real-time streaming environments. However, current vision foundation models remain fragmented, specializing narrowly in image semantic perception, offline temporal modeling, or spatial geometry. This paper introduces OmniStream, a unified streaming visual backbone that effectively perceives, reconstructs, and acts from diverse visual inputs. By incorporating causal spatiotemporal attention and 3D rotary positional embeddings (3D-RoPE), our model supports efficient, frame-by-frame online processing of video streams via a persistent KV-cache. We pre-train OmniStream using a synergistic multi-task framework coupling static and temporal representation learning, streaming geometric reconstruction, and vision-language alignment on 29 datasets. Extensive evaluations show that, even with a strictly frozen backbone, OmniStream achieves consistently competitive performance with specialized experts across image and video probing, streaming geometric reconstruction, complex video and spatial reasoning, as well as robotic manipulation (unseen at training). Rather than pursuing benchmark-specific dominance, our work demonstrates the viability of training a single, versatile vision backbone that generalizes across semantic, spatial, and temporal reasoning, i.e., a more meaningful step toward general-purpose visual understanding for interactive and embodied agents.

  </details>



- **GRADE: Benchmarking Discipline-Informed Reasoning in Image Editing**  
  Mingxin Liu, Ziqian Fan, Zhaokai Wang, Leyao Gu, Zirun Zhu, Yiguo He, Yuchen Yang, Changyao Tian, Xiangyu Zhao, Ning Liao, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12264v1  
  <details><summary>Abstract</summary>

  Unified multimodal models target joint understanding, reasoning, and generation, but current image editing benchmarks are largely confined to natural images and shallow commonsense reasoning, offering limited assessment of this capability under structured, domain-specific constraints. In this work, we introduce GRADE, the first benchmark to assess discipline-informed knowledge and reasoning in image editing. GRADE comprises 520 carefully curated samples across 10 academic domains, spanning from natural science to social science. To support rigorous evaluation, we propose a multi-dimensional evaluation protocol that jointly assesses Discipline Reasoning, Visual Consistency, and Logical Readability. Extensive experiments on 20 state-of-the-art open-source and closed-source models reveal substantial limitations in current models under implicit, knowledge-intensive editing settings, leading to large performance gaps. Beyond quantitative scores, we conduct rigorous analyses and ablations to expose model shortcomings and identify the constraints within disciplinary editing. Together, GRADE pinpoints key directions for the future development of unified multimodal models, advancing the research on discipline-informed image editing and reasoning. Our benchmark and evaluation code are publicly released.

  </details>



- **DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning**  
  Yujie Wei, Xinyu Liu, Shiwei Zhang, Hangjie Yuan, Jinbo Xing, Zhekai Chen, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12257v1  
  <details><summary>Abstract</summary>

  While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control. In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm. In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements. To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances. In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone. This provides motion-aware identity rewards in the latent space, prioritizing identity preservation aligned with human preferences. Supported by our curated large-scale dataset and the comprehensive DreamOmni Bench for multi-subject and omni-motion control evaluation, DreamVideo-Omni demonstrates superior performance in generating high-quality videos with precise controllability.

  </details>



- **Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training**  
  Fangfu Liu, Diankun Wu, Jiawei Chi, Yimo Cai, Yi-Hsin Hung, Xumin Yu, Hao Li, Han Hu, Yongming Rao, Yueqi Duan  
  _2026-03-12_ · https://arxiv.org/abs/2603.12255v1  
  <details><summary>Abstract</summary>

  Humans perceive and understand real-world spaces through a stream of visual observations. Therefore, the ability to streamingly maintain and update spatial evidence from potentially unbounded video streams is essential for spatial intelligence. The core challenge is not simply longer context windows but how spatial information is selected, organized, and retained over time. In this paper, we propose Spatial-TTT towards streaming visual-based spatial intelligence with test-time training (TTT), which adapts a subset of parameters (fast weights) to capture and organize spatial evidence over long-horizon scene videos. Specifically, we design a hybrid architecture and adopt large-chunk updates parallel with sliding-window attention for efficient spatial video processing. To further promote spatial awareness, we introduce a spatial-predictive mechanism applied to TTT layers with 3D spatiotemporal convolution, which encourages the model to capture geometric correspondence and temporal continuity across frames. Beyond architecture design, we construct a dataset with dense 3D spatial descriptions, which guides the model to update its fast weights to memorize and organize global 3D spatial signals in a structured manner. Extensive experiments demonstrate that Spatial-TTT improves long-horizon spatial understanding and achieves state-of-the-art performance on video spatial benchmarks. Project page: https://liuff19.github.io/Spatial-TTT.

  </details>



- **Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing**  
  Baifeng Shi, Stephanie Fu, Long Lian, Hanrong Ye, David Eigen, Aaron Reite, Boyi Li, Jan Kautz, Song Han, David M. Chan, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12254v1  
  <details><summary>Abstract</summary>

  Multi-modal large language models (MLLMs) have advanced general-purpose video understanding but struggle with long, high-resolution videos -- they process every pixel equally in their vision transformers (ViTs) or LLMs despite significant spatiotemporal redundancy. We introduce AutoGaze, a lightweight module that removes redundant patches before processed by a ViT or an MLLM. Trained with next-token prediction and reinforcement learning, AutoGaze autoregressively selects a minimal set of multi-scale patches that can reconstruct the video within a user-specified error threshold, eliminating redundancy while preserving information. Empirically, AutoGaze reduces visual tokens by 4x-100x and accelerates ViTs and MLLMs by up to 19x, enabling scaling MLLMs to 1K-frame 4K-resolution videos and achieving superior results on video benchmarks (e.g., 67.0% on VideoMME). Furthermore, we introduce HLVid: the first high-resolution, long-form video QA benchmark with 5-minute 4K-resolution videos, where an MLLM scaled with AutoGaze improves over the baseline by 10.1% and outperforms the previous best MLLM by 4.5%. Project page: https://autogaze.github.io/.

  </details>



- **SciMDR: Benchmarking and Advancing Scientific Multimodal Document Reasoning**  
  Ziyu Chen, Yilun Zhao, Chengye Wang, Rilyn Han, Manasi Patwardhan, Arman Cohan  
  _2026-03-12_ · https://arxiv.org/abs/2603.12249v1  
  <details><summary>Abstract</summary>

  Constructing scientific multimodal document reasoning datasets for foundation model training involves an inherent trade-off among scale, faithfulness, and realism. To address this challenge, we introduce the synthesize-and-reground framework, a two-stage pipeline comprising: (1) Claim-Centric QA Synthesis, which generates faithful, isolated QA pairs and reasoning on focused segments, and (2) Document-Scale Regrounding, which programmatically re-embeds these pairs into full-document tasks to ensure realistic complexity. Using this framework, we construct SciMDR, a large-scale training dataset for cross-modal comprehension, comprising 300K QA pairs with explicit reasoning chains across 20K scientific papers. We further construct SciMDR-Eval, an expert-annotated benchmark to evaluate multimodal comprehension within full-length scientific workflows. Experiments demonstrate that models fine-tuned on SciMDR achieve significant improvements across multiple scientific QA benchmarks, particularly in those tasks requiring complex document-level reasoning.

  </details>



- **Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation**  
  Xiangyu Zhao, Peiyuan Zhang, Junming Lin, Tianhao Liang, Yuchen Duan, Shengyuan Ding, Changyao Tian, Yuhang Zang, Junchi Yan, Xue Yang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12247v1  
  <details><summary>Abstract</summary>

  Reinforcement learning (RL) has emerged as a promising paradigm for enhancing image editing and text-to-image (T2I) generation. However, current reward models, which act as critics during RL, often suffer from hallucinations and assign noisy scores, inherently misguiding the optimization process. In this paper, we present FIRM (Faithful Image Reward Modeling), a comprehensive framework that develops robust reward models to provide accurate and reliable guidance for faithful image generation and editing. First, we design tailored data curation pipelines to construct high-quality scoring datasets. Specifically, we evaluate editing using both execution and consistency, while generation is primarily assessed via instruction following. Using these pipelines, we collect the FIRM-Edit-370K and FIRM-Gen-293K datasets, and train specialized reward models (FIRM-Edit-8B and FIRM-Gen-8B) that accurately reflect these criteria. Second, we introduce FIRM-Bench, a comprehensive benchmark specifically designed for editing and generation critics. Evaluations demonstrate that our models achieve superior alignment with human judgment compared to existing metrics. Furthermore, to seamlessly integrate these critics into the RL pipeline, we formulate a novel "Base-and-Bonus" reward strategy that balances competing objectives: Consistency-Modulated Execution (CME) for editing and Quality-Modulated Alignment (QMA) for generation. Empowered by this framework, our resulting models FIRM-Qwen-Edit and FIRM-SD3.5 achieve substantial performance breakthroughs. Comprehensive experiments demonstrate that FIRM mitigates hallucinations, establishing a new standard for fidelity and instruction adherence over existing general models. All of our datasets, models, and code have been publicly available at https://firm-reward.github.io.

  </details>



- **A Two-Stage Dual-Modality Model for Facial Emotional Expression Recognition**  
  Jiajun Sun, Zhe Gao  
  _2026-03-12_ · https://arxiv.org/abs/2603.12221v1  
  <details><summary>Abstract</summary>

  This paper addresses the expression (EXPR) recognition challenge in the 10th Affective Behavior Analysis in-the-Wild (ABAW) workshop and competition, which requires frame-level classification of eight facial emotional expressions from unconstrained videos. This task is challenging due to inaccurate face localization, large pose and scale variations, motion blur, temporal instability, and other confounding factors across adjacent frames. We propose a two-stage dual-modal (audio-visual) model to address these difficulties. Stage I focuses on robust visual feature extraction with a pretrained DINOv2-based encoder. Specifically, DINOv2 ViT-L/14 is used as the backbone, a padding-aware augmentation (PadAug) strategy is employed for image padding and data preprocessing from raw videos, and a mixture-of-experts (MoE) training head is introduced to enhance classifier diversity. Stage II addresses modality fusion and temporal consistency. For the visual modality, faces are re-cropped from raw videos at multiple scales, and the extracted visual features are averaged to form a robust frame-level representation. Concurrently, frame-aligned Wav2Vec 2.0 audio features are derived from short audio windows to provide complementary acoustic cues. These dual-modal features are integrated via a lightweight gated fusion module, followed by inference-time temporal smoothing. Experiments on the ABAW dataset demonstrate the effectiveness of the proposed method. The two-stage model achieves a Macro-F1 score of 0.5368 on the official validation set and 0.5122 +/- 0.0277 under 5-fold cross-validation, outperforming the official baselines.

  </details>



- **Real-World Point Tracking with Verifier-Guided Pseudo-Labeling**  
  Görkay Aydemir, Fatma Güney, Weidi Xie  
  _2026-03-12_ · https://arxiv.org/abs/2603.12217v1  
  <details><summary>Abstract</summary>

  Models for long-term point tracking are typically trained on large synthetic datasets. The performance of these models degrades in real-world videos due to different characteristics and the absence of dense ground-truth annotations. Self-training on unlabeled videos has been explored as a practical solution, but the quality of pseudo-labels strongly depends on the reliability of teacher models, which vary across frames and scenes. In this paper, we address the problem of real-world fine-tuning and introduce verifier, a meta-model that learns to assess the reliability of tracker predictions and guide pseudo-label generation. Given candidate trajectories from multiple pretrained trackers, the verifier evaluates them per frame and selects the most trustworthy predictions, resulting in high-quality pseudo-label trajectories. When applied for fine-tuning, verifier-guided pseudo-labeling substantially improves the quality of supervision and enables data-efficient adaptation to unlabeled videos. Extensive experiments on four real-world benchmarks demonstrate that our approach achieves state-of-the-art results while requiring less data than prior self-training methods. Project page: https://kuis-ai.github.io/track_on_r

  </details>



- **SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics**  
  Mengzhen Liu, Enshen Zhou, Cheng Chi, Yi Han, Shanyu Rong, Liming Chen, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12193v1  
  <details><summary>Abstract</summary>

  Active perception and manipulation are crucial for robots to interact with complex scenes. Existing methods struggle to unify semantic-driven active perception with robust, viewpoint-invariant execution. We propose SaPaVe, an end-to-end framework that jointly learns these capabilities in a data-efficient manner. Our approach decouples camera and manipulation actions rather than placing them in a shared action space, and follows a bottom-up training strategy: we first train semantic camera control on a large-scale dataset, then jointly optimize both action types using hybrid data. To support this framework, we introduce ActiveViewPose-200K, a dataset of 200k image-language-camera movement pairs for semantic camera movement learning, and a 3D geometry-aware module that improves execution robustness under dynamic viewpoints. We also present ActiveManip-Bench, the first benchmark for evaluating active manipulation beyond fixed-view settings. Extensive experiments in both simulation and real-world environments show that SaPaVe outperforms recent vision-language-action models such as GR00T N1 and \(π_0\), achieving up to 31.25\% higher success rates in real-world tasks. These results show that tightly coupled perception and execution, when trained with decoupled yet coordinated strategies, enable efficient and generalizable active manipulation. Project page: https://lmzpai.github.io/SaPaVe

  </details>



- **ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine for Scalable Contact-Rich Robotics Simulation and Control**  
  Chetan Borse, Zhixian Xie, Wei-Cheng Huang, Wanxin Jin  
  _2026-03-12_ · https://arxiv.org/abs/2603.12185v1  
  <details><summary>Abstract</summary>

  Physics simulation for contact-rich robotics is often bottlenecked by contact resolution: mainstream engines enforce non-penetration and Coulomb friction via complementarity constraints or constrained optimization, requiring per-step iterative solves whose cost grows superlinearly with contact density. We present ComFree-Sim, a GPU-parallelized analytical contact physics engine built on complementarity-free contact modeling. ComFree-Sim computes contact impulses in closed form via an impedance-style prediction--correction update in the dual cone of Coulomb friction. Contact computation decouples across contact pairs and becomes separable across cone facets, mapping naturally to GPU kernels and yielding near-linear runtime scaling with the number of contacts. We further extend the formulation to a unified 6D contact model capturing tangential, torsional, and rolling friction, and introduce a practical dual-cone impedance heuristic. ComFree-Sim is implemented in Warp and exposed through a MuJoCo-compatible interface as a drop-in backend alternative to MuJoCo Warp (MJWarp). Experiments benchmark penetration, friction behaviors, stability, and simulation runtime scaling against MJWarp, demonstrating near-linear scaling and 2--3 times higher throughput in dense contact scenes with comparable physical fidelity. We deploy ComFree-Sim in real-time MPC for in-hand dexterous manipulation on a real-world multi-fingered LEAP hand and in dynamics-aware motion retargeting, demonstrating that low-latency simulation yields higher closed-loop success rates and enables practical high-frequency control in contact-rich tasks.

  </details>



- **BehaviorVLM: Unified Finetuning-Free Behavioral Understanding with Vision-Language Reasoning**  
  Jingyang Ke, Weihan Li, Amartya Pradhan, Jeffrey Markowitz, Anqi Wu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12176v1  
  <details><summary>Abstract</summary>

  Understanding freely moving animal behavior is central to neuroscience, where pose estimation and behavioral understanding form the foundation for linking neural activity to natural actions. Yet both tasks still depend heavily on human annotation or unstable unsupervised pipelines, limiting scalability and reproducibility. We present BehaviorVLM, a unified vision-language framework for pose estimation and behavioral understanding that requires no task-specific finetuning and minimal human labeling by guiding pretrained Vision-Language Models (VLMs) through detailed, explicit, and verifiable reasoning steps. For pose estimation, we leverage quantum-dot-grounded behavioral data and propose a multi-stage pipeline that integrates temporal, spatial, and cross-view reasoning. This design greatly reduces human annotation effort, exposes low-confidence labels through geometric checks such as reprojection error, and produces labels that can later be filtered, corrected, or used to fine-tune downstream pose models. For behavioral understanding, we propose a pipeline that integrates deep embedded clustering for over-segmented behavior discovery, VLM-based per-clip video captioning, and LLM-based reasoning to merge and semantically label behavioral segments. The behavioral pipeline can operate directly from visual information and does not require keypoints to segment behavior. Together, these components enable scalable, interpretable, and label-light analysis of multi-animal behavior.

  </details>



- **LatentGeo: Learnable Auxiliary Constructions in Latent Space for Multimodal Geometric Reasoning**  
  Haiying Xu, Zihan Wang, Song Dai, Zhengxuan Zhang, Kairan Dou, Xuming Hu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12166v1  
  <details><summary>Abstract</summary>

  Despite recent advances in multimodal reasoning, representing auxiliary geometric constructions remains a fundamental challenge for multimodal large language models (MLLMs). Such constructions are absent from the original diagram and must be introduced before theorems apply. Existing approaches predominantly rely on explicit construction paradigms, including text-based geometric specification, visual-token interleaving during reasoning, and tool-augmented geometric execution. However, these methods either fail to faithfully represent complex spatial relationships, incur representation mismatch between discrete symbols and continuous geometric structures, or rely on external capabilities that hinder end-to-end optimization. To address these limitations, we propose LatentGeo, a framework that learns continuous latent visual representations to internalize auxiliary geometric constructions without pixel-level rendering or external executors. We design a three-stage curriculum that progressively aligns and internalizes these latent representations through auxiliary visual supervision, followed by LaGDPO, a latent-aware reinforcement learning procedure that stabilizes latent representations during policy optimization while improving end-task correctness. To systematically evaluate construction-centric representation quality, we introduce GeoAux, a new benchmark targeting visually dependent geometry problems, and conduct experiments on GeoAux and MathVerse. Results show that LatentGeo achieves substantial gains on geometric reasoning tasks, particularly those requiring auxiliary constructions. Extensive analyses and ablation studies further validate the effectiveness of each component in our framework.

  </details>



- **GlyphBanana: Advancing Precise Text Rendering Through Agentic Workflows**  
  Zexuan Yan, Jiarui Jin, Yue Ma, Shijian Wang, Jiahui Hu, Wenxiang Jiao, Yuan Lu, Linfeng Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12155v1  
  <details><summary>Abstract</summary>

  Despite recent advances in generative models driving significant progress in text rendering, accurately generating complex text and mathematical formulas remains a formidable challenge. This difficulty primarily stems from the limited instruction-following capabilities of current models when encountering out-of-distribution prompts. To address this, we introduce GlyphBanana, alongside a corresponding benchmark specifically designed for rendering complex characters and formulas. GlyphBanana employs an agentic workflow that integrates auxiliary tools to inject glyph templates into both the latent space and attention maps, facilitating the iterative refinement of generated images. Notably, our training-free approach can be seamlessly applied to various Text-to-Image (T2I) models, achieving superior precision compared to existing baselines. Extensive experiments demonstrate the effectiveness of our proposed workflow. Associated code is publicly available at https://github.com/yuriYanZeXuan/GlyphBanana.

  </details>



- **EgoIntent: An Egocentric Step-level Benchmark for Understanding What, Why, and Next**  
  Ye Pan, Chi Kit Wong, Yuanhuiyi Lyu, Hanqian Li, Jiahao Huo, Jiacheng Chen, Lutao Jiang, Xu Zheng, Xuming Hu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12147v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have demonstrated remarkable video reasoning capabilities across diverse tasks. However, their ability to understand human intent at a fine-grained level in egocentric videos remains largely unexplored. Existing benchmarks focus primarily on episode-level intent reasoning, overlooking the finer granularity of step-level intent understanding. Yet applications such as intelligent assistants, robotic imitation learning, and augmented reality guidance require understanding not only what a person is doing at each step, but also why and what comes next, in order to provide timely and context-aware support. To this end, we introduce EgoIntent, a step-level intent understanding benchmark for egocentric videos. It comprises 3,014 steps spanning 15 diverse indoor and outdoor daily-life scenarios, and evaluates models on three complementary dimensions: local intent (What), global intent (Why), and next-step plan (Next). Crucially, each clip is truncated immediately before the key outcome of the queried step (e.g., contact or grasp) occurs and contains no frames from subsequent steps, preventing future-frame leakage and enabling a clean evaluation of anticipatory step understanding and next-step planning. We evaluate 15 MLLMs, including both state-of-the-art closed-source and open-source models. Even the best-performing model achieves an average score of only 33.31 across the three intent dimensions, underscoring that step-level intent understanding in egocentric videos remains a highly challenging problem that calls for further investigation.

  </details>



- **FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance**  
  Quanhao Li, Zhen Xing, Rui Wang, Haidong Cao, Qi Dai, Daoguo Dong, Zuxuan Wu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12146v1  
  <details><summary>Abstract</summary>

  Recent advances in trajectory-controllable video generation have achieved remarkable progress. Previous methods mainly use adapter-based architectures for precise motion control along predefined trajectories. However, all these methods rely on a multi-step denoising process, leading to substantial time redundancy and computational overhead. While existing video distillation methods successfully distill multi-step generators into few-step, directly applying these approaches to trajectory-controllable video generation results in noticeable degradation in both video quality and trajectory accuracy. To bridge this gap, we introduce FlashMotion, a novel training framework designed for few-step trajectory-controllable video generation. We first train a trajectory adapter on a multi-step video generator for precise trajectory control. Then, we distill the generator into a few-step version to accelerate video generation. Finally, we finetune the adapter using a hybrid strategy that combines diffusion and adversarial objectives, aligning it with the few-step generator to produce high-quality, trajectory-accurate videos. For evaluation, we introduce FlashBench, a benchmark for long-sequence trajectory-controllable video generation that measures both video quality and trajectory accuracy across varying numbers of foreground objects. Experiments on two adapter architectures show that FlashMotion surpasses existing video distillation methods and previous multi-step models in both visual quality and trajectory consistency.

  </details>



- **HATS: Hardness-Aware Trajectory Synthesis for GUI Agents**  
  Rui Shao, Ruize Gao, Bin Xie, Yixing Li, Kaiwen Zhou, Shuai Wang, Weili Guan, Gongwei Chen  
  _2026-03-12_ · https://arxiv.org/abs/2603.12138v1  
  <details><summary>Abstract</summary>

  Graphical user interface (GUI) agents powered by large vision-language models (VLMs) have shown remarkable potential in automating digital tasks, highlighting the need for high-quality trajectory data to support effective agent training. Yet existing trajectory synthesis pipelines often yield agents that fail to generalize beyond simple interactions. We identify this limitation as stemming from the neglect of semantically ambiguous actions, whose meanings are context-dependent, sequentially dependent, or visually ambiguous. Such actions are crucial for real-world robustness but are under-represented and poorly processed in current datasets, leading to semantic misalignment between task instructions and execution. To address these issues, we propose HATS, a Hardness-Aware Trajectory Synthesis framework designed to mitigate the impact of semantic ambiguity. We define hardness as the degree of semantic ambiguity associated with an action and develop two complementary modules: (1) hardness-driven exploration, which guides data collection toward ambiguous yet informative interactions, and (2) alignment-guided refinement, which iteratively validates and repairs instruction-execution alignment. The two modules operate in a closed loop: exploration supplies refinement with challenging trajectories, while refinement feedback updates the hardness signal to guide future exploration. Extensive experiments show that agents trained with HATS consistently outperform state-of-the-art baselines across benchmark GUI environments.

  </details>



- **EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation**  
  Yan Li, Ning Liao, Xiangyu Zhao, Shaofeng Zhang, Xiaoxing Wang, Yifan Yang, Junchi Yan, Xue Yang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12108v1  
  <details><summary>Abstract</summary>

  The development of unified multimodal large language models (MLLMs) is fundamentally challenged by the granularity gap between visual understanding and generation: understanding requires high-level semantic abstractions, while image generation demands fine-grained pixel-level representations. Existing approaches usually enforce the two supervision on the same set of representation or decouple these two supervision on separate feature spaces, leading to interference and inconsistency, respectively. In this work, we propose EvoTok, a unified image tokenizer that reconciles these requirements through a residual evolution process within a shared latent space. Instead of maintaining separate token spaces for pixels and semantics, EvoTok encodes an image into a cascaded sequence of residual tokens via residual vector quantization. This residual sequence forms an evolution trajectory where earlier stages capture low-level details and deeper stages progressively transition toward high-level semantic representations. Despite being trained on a relatively modest dataset of 13M images, far smaller than the billion-scale datasets used by many previous unified tokenizers, EvoTok achieves a strong reconstruction quality of 0.43 rFID on ImageNet-1K at 256x256 resolution. When integrated with a large language model, EvoTok shows promising performance across 7 out of 9 visual understanding benchmarks, and remarkable results on image generation benchmarks such as GenEval and GenAI-Bench. These results demonstrate that modeling visual representations as an evolving trajectory provides an effective and principled solution for unifying visual understanding and generation.

  </details>



- **Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis**  
  Xiaolong Qian, Qi Jiang, Yao Gao, Lei Sun, Zhonghua Yi, Kailun Yang, Luc Van Gool, Kaiwei Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.12083v1  
  <details><summary>Abstract</summary>

  Prevalent Computational Aberration Correction (CAC) methods are typically tailored to specific optical systems, leading to poor generalization and labor-intensive re-training for new lenses. Developing CAC paradigms capable of generalizing across diverse photographic lenses offers a promising solution to these challenges. However, efforts to achieve such cross-lens universality within consumer photography are still in their early stages due to the lack of a comprehensive benchmark that encompasses a sufficiently wide range of optical aberrations. Furthermore, it remains unclear which specific factors influence existing CAC methods and how these factors affect their performance. In this paper, we present comprehensive experiments and evaluations involving 24 image restoration and CAC algorithms, utilizing our newly proposed UniCAC, a large-scale benchmark for photographic cameras constructed via automatic optical design. The Optical Degradation Evaluator (ODE) is introduced as a novel framework to objectively assess the difficulty of CAC tasks, offering credible quantification of optical aberrations and enabling reliable evaluation. Drawing on our comparative analysis, we identify three key factors -- prior utilization, network architecture, and training strategy -- that most significantly influence CAC performance, and further investigate their respective effects. We believe that our benchmark, dataset, and observations contribute foundational insights to related areas and lay the groundwork for future investigations. Benchmarks, codes, and Zemax files will be available at https://github.com/XiaolongQian/UniCAC.

  </details>



- **Paper Title: LoV3D: Grounding Cognitive Prognosis Reasoning in Longitudinal 3D Brain MRI via Regional Volume Assessments**  
  Zhaoyang Jiang, Zhizhong Fu, David McAllister, Yunsoo Kim, Honghan Wu  
  _2026-03-12_ · https://arxiv.org/abs/2603.12071v1  
  <details><summary>Abstract</summary>

  Longitudinal brain MRI is essential for characterizing the progression of neurological diseases such as Alzheimer's disease assessment. However, current deep-learning tools fragment this process: classifiers reduce a scan to a label, volumetric pipelines produce uninterpreted measurements, and vision-language models (VLMs) may generate fluent but potentially hallucinated conclusions. We present LoV3D, a pipeline for training 3D vision-language models, which reads longitudinal T1-weighted brain MRI, produces a region-level anatomical assessment, conducts longitudinal comparison with the prior scan, and finally outputs a three-class diagnosis (Cognitively Normal, Mild Cognitive Impairment, or Dementia) along with a synthesized diagnostic summary. The stepped pipeline grounds the final diagnosis by enforcing label consistency, longitudinal coherence, and biological plausibility, thereby reducing the risks of hallucinations. The training process introduces a clinically-weighted Verifier that scores candidate outputs automatically against normative references derived from standardized volume metrics, driving Direct Preference Optimization without a single human annotation. On a subject-level held-out ADNI test set (479 scans, 258 subjects), LoV3D achieves 93.7% three-class diagnostic accuracy (+34.8% over the no-grounding baseline), 97.2% on two-class diagnosis accuracy (+4% over the SOTA) and 82.6% region-level anatomical classification accuracy (+33.1% over VLM baselines). Zero-shot transfer yields 95.4% on MIRIAD (100% Dementia recall) and 82.9% three-class accuracy on AIBL, confirming high generalizability across sites, scanners, and populations. Code is available at https://github.com/Anonymous-TEVC/LoV-3D.

  </details>



- **Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos**  
  Shuo Sun, Unal Artan, Malcolm Mielle, Achim J. Lilienthaland, Martin Magnusson  
  _2026-03-12_ · https://arxiv.org/abs/2603.12064v1  
  <details><summary>Abstract</summary>

  We address the challenging problem of dense dynamic scene reconstruction and camera pose estimation from multiple freely moving cameras -- a setting that arises naturally when multiple observers capture a shared event. Prior approaches either handle only single-camera input or require rigidly mounted, pre-calibrated camera rigs, limiting their practical applicability. We propose a two-stage optimization framework that decouples the task into robust camera tracking and dense depth refinement. In the first stage, we extend single-camera visual SLAM to the multi-camera setting by constructing a spatiotemporal connection graph that exploits both intra-camera temporal continuity and inter-camera spatial overlap, enabling consistent scale and robust tracking. To ensure robustness under limited overlap, we introduce a wide-baseline initialization strategy using feed-forward reconstruction models. In the second stage, we refine depth and camera poses by optimizing dense inter- and intra-camera consistency using wide-baseline optical flow. Additionally, we introduce MultiCamRobolab, a new real-world dataset with ground-truth poses from a motion capture system. Finally, we demonstrate that our method significantly outperforms state-of-the-art feed-forward models on both synthetic and real-world benchmarks, while requiring less memory.

  </details>



- **Pano360: Perspective to Panoramic Vision with Geometric Consistency**  
  Zhengdong Zhu, Weiyi Xue, Zuyuan Yang, Wenlve Zhou, Zhiheng Zhou  
  _2026-03-12_ · https://arxiv.org/abs/2603.12013v1  
  <details><summary>Abstract</summary>

  Prior panorama stitching approaches heavily rely on pairwise feature correspondences and are unable to leverage geometric consistency across multiple views. This leads to severe distortion and misalignment, especially in challenging scenes with weak textures, large parallax, and repetitive patterns. Given that multi-view geometric correspondences can be directly constructed in 3D space, making them more accurate and globally consistent, we extend the 2D alignment task to the 3D photogrammetric space. We adopt a novel transformer-based architecture to achieve 3D awareness and aggregate global information across all views. It directly utilizes camera poses to guide image warping for global alignment in 3D space and employs a multi-feature joint optimization strategy to compute the seams. Additionally, to establish an evaluation benchmark and train our network, we constructed a large-scale dataset of real-world scenes. Extensive experiments show that our method significantly outperforms existing alternatives in alignment accuracy and perceptual quality.

  </details>



- **CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation**  
  Ziqi Ye, Ziyang Gong, Ning Liao, Xiaoxing Hu, Di Wang, Hongruixuan Chen, Chen Huang, Yiguo He, Yuru Jia, Xiaoxing Wang, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12008v1  
  <details><summary>Abstract</summary>

  Synthetic Aperture Radar (SAR) enables global, all-weather earth observation. However, owing to diverse imaging mechanisms, domain shifts across sensors and regions severely hinder its semantic generalization. To address this, we present CrossEarth-SAR, the first billion-scale SAR vision foundation model built upon a novel physics-guided sparse mixture-of-experts (MoE) architecture incorporating physical descriptors, explicitly designed for cross-domain semantic segmentation. To facilitate large-scale pre-training, we develop CrossEarth-SAR-200K, a weakly and fully supervised dataset that unifies public and private SAR imagery. We also introduce a benchmark suite comprising 22 sub-benchmarks across 8 distinct domain gaps, establishing the first unified standard for domain generalization semantic segmentation on SAR imagery. Extensive experiments demonstrate that CrossEarth-SAR achieves state-of-the-art results on 20 benchmarks, surpassing previous methods by over 10\% mIoU on some benchmarks under multi-gap transfer. All code, benchmark and datasets will be publicly available.

  </details>



- **HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios**  
  Jiayue Pu, Zhongxiang Sun, Zilu Zhang, Xiao Zhang, Jun Xu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11975v1  
  <details><summary>Abstract</summary>

  The rapid evolution of embodied agents has accelerated the deployment of household robots in real-world environments. However, unlike structured industrial settings, household spaces introduce unpredictable safety risks, where system limitations such as perception latency and lack of common sense knowledge can lead to dangerous errors. Current safety evaluations, often restricted to static images, text, or general hazards, fail to adequately benchmark dynamic unsafe action detection in these specific contexts. To bridge this gap, we introduce \textbf{HomeSafe-Bench}, a challenging benchmark designed to evaluate Vision-Language Models (VLMs) on unsafe action detection in household scenarios. HomeSafe-Bench is contrusted via a hybrid pipeline combining physical simulation with advanced video generation and features 438 diverse cases across six functional areas with fine-grained multidimensional annotations. Beyond benchmarking, we propose \textbf{Hierarchical Dual-Brain Guard for Household Safety (HD-Guard)}, a hierarchical streaming architecture for real-time safety monitoring. HD-Guard coordinates a lightweight FastBrain for continuous high-frequency screening with an asynchronous large-scale SlowBrain for deep multimodal reasoning, effectively balancing inference efficiency with detection accuracy. Evaluations demonstrate that HD-Guard achieves a superior trade-off between latency and performance, while our analysis identifies critical bottlenecks in current VLM-based safety detection.

  </details>



- **Multimodal Emotion Recognition via Bi-directional Cross-Attention and Temporal Modeling**  
  Junhyeong Byeon, Jeongyeol Kim, Sejoon Lim  
  _2026-03-12_ · https://arxiv.org/abs/2603.11971v1  
  <details><summary>Abstract</summary>

  Emotion recognition in in-the-wild video data remains a challenging problem due to large variations in facial appearance, head pose, illumination, background noise, and the inherently dynamic nature of human affect. Relying on a single modality, such as facial expressions or speech, is often insufficient to capture these complex emotional cues. To address this issue, we propose a multimodal emotion recognition framework for the Expression (EXPR) Recognition task in the 10th Affective Behavior Analysis in-the-wild (ABAW) Challenge. Our approach leverages large-scale pre-trained models, namely CLIP for visual encoding and Wav2Vec 2.0 for audio representation learning, as frozen backbone networks. To model temporal dependencies in facial expression sequences, we employ a Temporal Convolutional Network (TCN) over fixed-length video windows. In addition, we introduce a bi-directional cross-attention fusion module, in which visual and audio features interact symmetrically to enhance cross-modal contextualization and capture complementary emotional information. A lightweight classification head is then used for final emotion prediction. We further incorporate a text-guided contrastive objective based on CLIP text features to encourage semantically aligned visual representations. Experimental results on the ABAW 10th EXPR benchmark show that the proposed framework provides a strong multimodal baseline and achieves improved performance over unimodal modeling. These results demonstrate the effectiveness of combining temporal visual modeling, audio representation learning, and cross-modal fusion for robust emotion recognition in unconstrained real-world environments.

  </details>



- **Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting**  
  Chantal Pellegrini, Adrian Delchev, Ege Özsoy, Nassir Navab, Matthias Keicher  
  _2026-03-12_ · https://arxiv.org/abs/2603.11938v1  
  <details><summary>Abstract</summary>

  Structured radiology reporting promises faster, more consistent communication than free text, but automation remains difficult as models must make many fine-grained, discrete decisions about rare findings and attributes from limited structured supervision. In contrast, free-text reports are produced at scale in routine care and implicitly encode fine-grained, image-linked information through detailed descriptions. To leverage this unstructured knowledge, we propose ProtoSR, an approach for injecting free-text information into structured report population. First, we introduce an automatic extraction pipeline that uses an instruction-tuned LLM to mine 80k+ MIMIC-CXR studies and build a multimodal knowledge base aligned with a structured reporting template, representing each answer option with a visual prototype. Using this knowledge base, ProtoSR is trained to retrieve prototypes relevant for the current image-question pair and augment the model predictions through a prototype-conditioned residual, providing a data-driven second opinion that selectively corrects predictions. On the Rad-ReStruct benchmark, ProtoSR achieves state-of-the-art results, with the largest improvements on detailed attribute questions, demonstrating the value of integrating free-text derived signal for fine-grained image understanding.

  </details>



- **Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models**  
  Lu Wang, Zhuoran Jin, Yupu Hao, Yubo Chen, Kang Liu, Yulong Ao, Jun Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11896v1  
  <details><summary>Abstract</summary>

  Multimodal large language models (MLLMs) have shown strong performance on offline video understanding, but most are limited to offline inference or have weak online reasoning, making multi-turn interaction over continuously arriving video streams difficult. Existing streaming methods typically use an interleaved perception-generation paradigm, which prevents concurrent perception and generation and leads to early memory decay as streams grow, hurting long-range dependency modeling. We propose Think While Watching, a memory-anchored streaming video reasoning framework that preserves continuous segment-level memory during multi-turn interaction. We build a three-stage, multi-round chain-of-thought dataset and adopt a stage-matched training strategy, while enforcing strict causality through a segment-level streaming causal mask and streaming positional encoding. During inference, we introduce an efficient pipeline that overlaps watching and thinking and adaptively selects the best attention backend. Under both single-round and multi-round streaming input protocols, our method achieves strong results. Built on Qwen3-VL, it improves single-round accuracy by 2.6% on StreamingBench and by 3.79% on OVO-Bench. In the multi-round setting, it maintains performance while reducing output tokens by 56%. Code is available at: https://github.com/wl666hhh/Think_While_Watching/

  </details>



- **ZeroSense:How Vision matters in Long Context Compression**  
  Yonghan Gao, Zehong Chen, Lijian Xu, Jingzhi Chen, Jingwei Guan, Xingyu Zeng  
  _2026-03-12_ · https://arxiv.org/abs/2603.11846v1  
  <details><summary>Abstract</summary>

  Recent visual-text compression (VTC) methods, typified by DeepSeek-OCR, report impressive high token compression ratios for long-context modeling tasks by leveraging text-to-image rendering. However, existing evaluation protocols heavily rely on downstream task performance. Such evaluation metrics fail to accurately measure text preservation due to the strong inherent linguistic priors of Multimodal Large Language Models (MLLMs). In this work, we introduce a new evaluation framework that decouples MLLMs' capabilities to faithfully assess VTC quality. Within this framework, we further introduce the ZeroSense Benchmark to ensure low semantic correlation of testing samples. By eliminating contextual dependencies, our benchmark guarantees that the evaluation results are purely reflective of VTC quality, unaffected by the semantic inference capabilities of downstream models. Extensive experiments across multiple datasets demonstrate that VTC quality and downstream task accuracy diverge significantly, highlighting the necessity of our decoupled evaluation framework.

  </details>



- **Towards High-Fidelity CAD Generation via LLM-Driven Program Generation and Text-Based B-Rep Primitive Grounding**  
  Jiahao Li, Qingwang Zhang, Qiuyu Chen, Guozhan Qiu, Yunzhong Lou, Xiangdong Zhou  
  _2026-03-12_ · https://arxiv.org/abs/2603.11831v1  
  <details><summary>Abstract</summary>

  The field of Computer-Aided Design (CAD) generation has made significant progress in recent years. Existing methods typically fall into two separate categorie: parametric CAD modeling and direct boundary representation (B-Rep) synthesis. In modern feature-based CAD systems, parametric modeling and B-Rep are inherently intertwined, as advanced parametric operations (e.g., fillet and chamfer) require explicit selection of B-Rep geometric primitives, and the B-Rep itself is derived from parametric operations. Consequently, this paradigm gap remains a critical factor limiting AI-driven CAD modeling for complex industrial product design. This paper present FutureCAD, a novel text-to-CAD framework that leverages large language models (LLMs) and a B-Rep grounding transformer (BRepGround) for high-fidelity CAD generation. Our method generates executable CadQuery scripts, and introduces a text-based query mechanism that enables the LLM to specify geometric selections via natural language, which BRepGround then grounds to the target primitives. To train our framework, we construct a new dataset comprising real-world CAD models. For the LLM, we apply supervised fine-tuning (SFT) to establish fundamental CAD generation capabilities, followed by reinforcement learning (RL) to improve generalization. Experiments show that FutureCAD achieves state-of-the-art CAD generation performance.

  </details>



- **Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI**  
  Md. Hasin Sarwar Ifty, Nisharga Nirjan, Labib Islam, M. A. Diganta, Reeyad Ahmed Ornate, Anika Tasnim, Md. Saiful Islam  
  _2026-03-12_ · https://arxiv.org/abs/2603.11818v1  
  <details><summary>Abstract</summary>

  The unrestrained proliferation of cells that are malignant in nature is cancer. In recent times, medical professionals are constantly acquiring enhanced diagnostic and treatment abilities by implementing deep learning models to analyze medical data for better clinical decision, disease diagnosis and drug discovery. A majority of cancers are studied and treated by incorporating these technologies. However, ovarian cancer remains a dilemma as it has inaccurate non-invasive detection procedures and a time consuming, invasive procedure for accurate detection. Thus, in this research, several Convolutional Neural Networks such as LeNet-5, ResNet, VGGNet and GoogLeNet/Inception have been utilized to develop 15 variants and choose a model that accurately detects and identifies ovarian cancer. For effective model training, the dataset OvarianCancer&SubtypesDatasetHistopathology from Mendeley has been used. After constructing a model, we utilized Explainable Artificial Intelligence (XAI) models such as LIME, Integrated Gradients and SHAP to explain the black box outcome of the selected model. For evaluating the performance of the model, Accuracy, Precision, Recall, F1-Score, ROC Curve and AUC have been used. From the evaluation, it was seen that the slightly compact InceptionV3 model with ReLu had the overall best result achieving an average score of 94% across all the performance metrics in the augmented dataset. Lastly for XAI, the three aforementioned XAI have been used for an overall comparative analysis. It is the aim of this research that the contributions of the study will help in achieving a better detection method for ovarian cancer.

  </details>



- **CEI-3D: Collaborative Explicit-Implicit 3D Reconstruction for Realistic and Fine-Grained Object Editing**  
  Yue Shi, Rui Shi, Yuxuan Xiong, Bingbing Ni, Wenjun Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11810v1  
  <details><summary>Abstract</summary>

  Existing 3D editing methods often produce unrealistic and unrefined results due to the deeply integrated nature of their reconstruction networks. To address the challenge, this paper introduces CEI-3D, an editing-oriented reconstruction pipeline designed to facilitate realistic and fine-grained editing. Specifically, we propose a collaborative explicit-implicit reconstruction approach, which represents the target object using an implicit SDF network and a differentially sampled, locally controllable set of handler points. The implicit network provides a smooth and continuous geometry prior, while the explicit handler points offer localized control, enabling mutual guidance between the global 3D structure and user-specified local editing regions. To independently control each attribute of the handler points, we design a physical properties disentangling module to decouple the color of the handler points into separate physical properties. We also propose a dual-diffuse-albedo network in this module to process the edited and non-edited regions through separate branches, thereby preventing undesired interference from editing operations. Building on the reconstructed collaborative explicit-implicit representation with disentangled properties, we introduce a spatial-aware editing module that enables part-wise adjustment of relevant handler points. This module employs a cross-view propagation-based 3D segmentation strategy, which helps users to edit the specified physical attributes of a target part efficiently. Extensive experiments on both real and synthetic datasets demonstrate that our approach achieves more realistic and fine-grained editing results than the state-of-the-art (SOTA) methods while requiring less editing time. Our code is available on https://github.com/shiyue001/CEI-3D.

  </details>



- **HiSync: Spatio-Temporally Aligning Hand Motion from Wearable IMU and On-Robot Camera for Command Source Identification in Long-Range HRI**  
  Chengwen Zhang, Chun Yu, Borong Zhuang, Haopeng Jin, Qingyang Wan, Zhuojun Li, Zhe He, Zhoutong Ye, Yu Mei, Chang Liu, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.11809v1  
  <details><summary>Abstract</summary>

  Long-range Human-Robot Interaction (HRI) remains underexplored. Within it, Command Source Identification (CSI) - determining who issued a command - is especially challenging due to multi-user and distance-induced sensor ambiguity. We introduce HiSync, an optical-inertial fusion framework that treats hand motion as binding cues by aligning robot-mounted camera optical flow with hand-worn IMU signals. We first elicit a user-defined (N=12) gesture set and collect a multimodal command gesture dataset (N=38) in long-range multi-user HRI scenarios. Next, HiSync extracts frequency-domain hand motion features from both camera and IMU data, and a learned CSINet denoises IMU readings, temporally aligns modalities, and performs distance-aware multi-window fusion to compute cross-modal similarity of subtle, natural gestures, enabling robust CSI. In three-person scenes up to 34m, HiSync achieves 92.32% CSI accuracy, outperforming the prior SOTA by 48.44%. HiSync is also validated on real-robot deployment. By making CSI reliable and natural, HiSync provides a practical primitive and design guidance for public-space HRI.

  </details>



- **OSM-based Domain Adaptation for Remote Sensing VLMs**  
  Stefan Maria Ailuro, Mario Markov, Mohammad Mahdi, Delyan Boychev, Luc Van Gool, Danda Pani Paudel  
  _2026-03-12_ · https://arxiv.org/abs/2603.11804v1  
  <details><summary>Abstract</summary>

  Vision-Language Models (VLMs) adapted to remote sensing rely heavily on domain-specific image-text supervision, yet high-quality annotations for satellite and aerial imagery remain scarce and expensive to produce. Prevailing pseudo-labeling pipelines address this gap by distilling knowledge from large frontier models, but this dependence on large teachers is costly, limits scalability, and caps achievable performance at the ceiling of the teacher. We propose OSMDA: a self-contained domain adaptation framework that eliminates this dependency. Our key insight is that a capable base VLM can serve as its own annotation engine: by pairing aerial images with rendered OpenStreetMap (OSM) tiles, we leverage optical character recognition and chart comprehension capabilities of the model to generate captions enriched by OSM's vast auxiliary metadata. The model is then fine-tuned on the resulting corpus with satellite imagery alone, yielding OSMDA-VLM, a domain-adapted VLM that requires no manual labeling and no stronger external model. We conduct exhaustive evaluations spanning 10 benchmarks across image-text-to-text tasks and comparing against 9 competitive baselines. When equally mixed with real data, our method achieves state-of-the-art results, while being substantially cheaper to train than teacher-dependent alternatives. These results suggest that, given a strong foundation model, alignment with crowd-sourced geographic data is a practical and scalable path towards remote sensing domain adaptation. Dataset and model weights will be made publicly available.

  </details>



- **Locating Demographic Bias at the Attention-Head Level in CLIP's Vision Encoder**  
  Alaa Yasser, Kittipat Phunjanna, Marcos Escudero Viñolo, Catarina Barata, Jenny Benois-Pineau  
  _2026-03-12_ · https://arxiv.org/abs/2603.11793v1  
  <details><summary>Abstract</summary>

  Standard fairness audits of foundation models quantify that a model is biased, but not where inside the network the bias resides. We propose a mechanistic fairness audit that combines projected residual-stream decomposition, zero-shot Concept Activation Vectors, and bias-augmented TextSpan analysis to locate demographic bias at the level of individual attention heads in vision transformers. As a feasibility case study, we apply this pipeline to the CLIP ViT-L-14 encoder on 42 profession classes of the FACET benchmark, auditing both gender and age bias. For gender, the pipeline identifies four terminal-layer heads whose ablation reduces global bias (Cramer's V: 0.381 -> 0.362) while marginally improving accuracy (+0.42%); a layer-matched random control confirms that this effect is specific to the identified heads. A single head in the final layer contributes to the majority of the reduction in the most stereotyped classes, and class-level analysis shows that corrected predictions shift toward the correct occupation. For age, the same pipeline identifies candidate heads, but ablation produces weaker and less consistent effects, suggesting that age bias is encoded more diffusely than gender bias in this model. These results provide preliminary evidence that head-level bias localisation is feasible for discriminative vision encoders and that the degree of localisability may vary across protected attributes. keywords: Bias . CLIP . Mechanistic Interpretability . Vision Transformer . Fairness

  </details>



- **Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints**  
  Chenyangguang Zhang, Botao Ye, Boqi Chen, Alexandros Delitzas, Fangjinhua Wang, Marc Pollefeys, Xi Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11755v1  
  <details><summary>Abstract</summary>

  Motion-controllable video generation is crucial for egocentric applications in virtual reality and embodied AI. However, existing methods often struggle to achieve 3D-consistent fine-grained hand articulation. By adopting on 2D trajectories or implicit poses, they collapse 3D geometry into spatially ambiguous signals or over rely on human-centric priors. Under severe egocentric occlusions, this causes motion inconsistencies and hallucinated artifacts, as well as preventing cross-embodiment generalization to robotic hands. To address these limitations, we propose a novel framework that generates egocentric videos from a single reference frame, leveraging sparse 3D hand joints as embodiment-agnostic control signals with clear semantic and geometric structures. We introduce an efficient control module that resolves occlusion ambiguities while fully preserving 3D information. Specifically, it extracts occlusion-aware features from the source reference frame by penalizing unreliable visual signals from hidden joints, and employs a 3D-based weighting mechanism to robustly handle dynamically occluded target joints during motion propagation. Concurrently, the module directly injects 3D geometric embeddings into the latent space to strictly enforce structural consistency. To facilitate robust training and evaluation, we develop an automated annotation pipeline that yields over one million high-quality egocentric video clips paired with precise hand trajectories. Additionally, we register humanoid kinematic and camera data to construct a cross-embodiment benchmark. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art baselines, generating high-fidelity egocentric videos with realistic interactions and exhibiting exceptional cross-embodiment generalization to robotic hands.

  </details>



- **VTEdit-Bench: A Comprehensive Benchmark for Multi-Reference Image Editing Models in Virtual Try-On**  
  Xiaoye Liang, Zhiyuan Qu, Mingye Zou, Jiaxin Liu, Lai Jiang, Mai Xu, Yiheng Zhu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11734v1  
  <details><summary>Abstract</summary>

  As virtual try-on (VTON) continues to advance, a growing number of real-world scenarios have emerged, pushing beyond the ability of the existing specialized VTON models. Meanwhile, universal multi-reference image editing models have progressed rapidly and exhibit strong generalization in visual editing, suggesting a promising route toward more flexible VTON systems. However, despite their strong capabilities, the strengths and limitations of universal editors for VTON remain insufficiently explored due to the lack of systematic evaluation benchmarks. To address this gap, we introduce VTEdit-Bench, a comprehensive benchmark designed to evaluate universal multi-reference image editing models across various realistic VTON scenarios. VTEdit-Bench contains 24,220 test image pairs spanning five representative VTON tasks with progressively increasing complexity, enabling systematic analysis of robustness and generalization. We further propose VTEdit-QA, a reference-aware VLM-based evaluator that assesses VTON performance from three key aspects: model consistency, cloth consistency, and overall image quality. Through this framework, we systematically evaluate eight universal editing models and compare them with seven specialized VTON models. Results show that top universal editors are competitive on conventional tasks and generalize more stably to harder scenarios, but remain challenged by complex reference configurations, particularly multi-cloth conditioning.

  </details>



- **OSCBench: Benchmarking Object State Change in Text-to-Video Generation**  
  Xianjing Han, Bin Zhu, Shiqi Hu, Franklin Mingzhe Li, Patrick Carrington, Roger Zimmermann, Jingjing Chen  
  _2026-03-12_ · https://arxiv.org/abs/2603.11698v1  
  <details><summary>Abstract</summary>

  Text-to-video (T2V) generation models have made rapid progress in producing visually high-quality and temporally coherent videos. However, existing benchmarks primarily focus on perceptual quality, text-video alignment, or physical plausibility, leaving a critical aspect of action understanding largely unexplored: object state change (OSC) explicitly specified in the text prompt. OSC refers to the transformation of an object's state induced by an action, such as peeling a potato or slicing a lemon. In this paper, we introduce OSCBench, a benchmark specifically designed to assess OSC performance in T2V models. OSCBench is constructed from instructional cooking data and systematically organizes action-object interactions into regular, novel, and compositional scenarios to probe both in-distribution performance and generalization. We evaluate six representative open-source and proprietary T2V models using both human user study and multimodal large language model (MLLM)-based automatic evaluation. Our results show that, despite strong performance on semantic and scene alignment, current T2V models consistently struggle with accurate and temporally consistent object state changes, especially in novel and compositional settings. These findings position OSC as a key bottleneck in text-to-video generation and establish OSCBench as a diagnostic benchmark for advancing state-aware video generation models.

  </details>



- **FL-MedSegBench: A Comprehensive Benchmark for Federated Learning on Medical Image Segmentation**  
  Meilu Zhu, Zhiwei Wang, Axiu Mao, Yuxing Li, Xiaohan Xing, Yixuan Yuan, Edmund Y. Lam  
  _2026-03-12_ · https://arxiv.org/abs/2603.11659v1  
  <details><summary>Abstract</summary>

  Federated learning (FL) offers a privacy-preserving paradigm for collaborative medical image analysis without sharing raw data. However, the absence of standardized benchmarks for medical image segmentation hinders fair and comprehensive evaluation of FL methods. To address this gap, we introduce FL-MedSegBench, the first comprehensive benchmark for federated learning on medical image segmentation. Our benchmark encompasses nine segmentation tasks across ten imaging modalities, covering both 2D and 3D formats with realistic clinical heterogeneity. We systematically evaluate eight generic FL (gFL) and five personalized FL (pFL) methods across multiple dimensions: segmentation accuracy, fairness, communication efficiency, convergence behavior, and generalization to unseen domains. Extensive experiments reveal several key insights: (i) pFL methods, particularly those with client-specific batch normalization (\textit{e.g.}, FedBN), consistently outperform generic approaches; (ii) No single method universally dominates, with performance being dataset-dependent; (iii) Communication frequency analysis shows normalization-based personalization methods exhibit remarkable robustness to reduced communication frequency; (iv) Fairness evaluation identifies methods like Ditto and FedRDN that protect underperforming clients; (v) A method's generalization to unseen domains is strongly tied to its ability to perform well across participating clients. We will release an open-source toolkit to foster reproducible research and accelerate clinically applicable FL solutions, providing empirically grounded guidelines for real-world clinical deployment. The source code is available at https://github.com/meiluzhu/FL-MedSegBench.

  </details>



- **Diversity You Can Actually Measure: A Fast, Model-Free Diversity Metric for Robotics Datasets**  
  Sreevardhan Sirigiri, Nathan Samuel de Lara, Christopher Agia, Florian Shkurti, Fabio Ramos  
  _2026-03-12_ · https://arxiv.org/abs/2603.11634v1  
  <details><summary>Abstract</summary>

  Robotics datasets for imitation learning typically consist of long-horizon trajectories of different lengths over states, actions, and high-dimensional observations (e.g., RGB video), making it non-trivial to quantify diversity in a way that respects the underlying trajectory structure and geometry. We extend Shannon and von Neumann entropy to this setting by defining signature transform-based entropy on the Gram matrix of a signature kernel over demonstrations, yielding entropy and diversity metrics that operate directly on the demonstration dataset. Building on these metrics, we study how dataset diversity affects generalization performance in robot imitation learning and propose a simple, model-free way to curate diverse demonstrations. We introduce FAKTUAL (FAst trajectory Kernel enTropy cUration for imitation Learning), a data curation algorithm that selects a subset of demonstrations maximizing entropy given a subset-size budget. FAKTUAL is fully model-free, requires no access to the imitation policy or rollouts, and adds negligible overhead relative to policy training. We evaluate our approach on image and state-based RoboMimic and MetaWorld benchmarks, as well as four real-world manipulation tasks. Across tasks and architectures, diversity-aware curation with FAKTUAL consistently improves downstream success rates over random selection, while being substantially more computationally efficient compared to recent robot data curation methods. Our results suggest that the entropy of demonstration datasets is a practical tool for understanding and improving dataset diversity in robot imitation learning.

  </details>



- **VisDoT : Enhancing Visual Reasoning through Human-Like Interpretation Grounding and Decomposition of Thought**  
  Eunsoo Lee, Jeongwoo Lee, Minki Hong, Jangho Choi, Jihie Kim  
  _2026-03-12_ · https://arxiv.org/abs/2603.11631v1  
  <details><summary>Abstract</summary>

  Large vision-language models (LVLMs) struggle to reliably detect visual primitives in charts and align them with semantic representations, which severely limits their performance on complex visual reasoning. This lack of perceptual grounding constitutes a major bottleneck for chart-based reasoning. We propose VisDoT, a framework that enhances visual reasoning through human-like interpretation grounding. We formalize four perceptual tasks based on the theory of graphical perception, including position and length. Building on this foundation, we introduce Decomposition-of-Thought (DoT) prompting, which sequentially separates questions into visual perception sub-questions and logic sub-questions. Fine-tuning InternVL with VisDoT achieves a +11.2% improvement on ChartQA and surpasses GPT-4o on the more challenging ChartQAPro benchmark. On the newly introduced VisDoTQA benchmark, the model improves by +33.2%. Furthermore, consistent zero-shot gains on diverse open-domain VQA benchmarks confirm the generalizability of the perception-logic separation strategy for visual question answering. VisDoT leverages human-like perception to enhance visual grounding, achieving state-of-the-art chart understanding and interpretable visual reasoning.

  </details>



- **Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography**  
  Yichi Zhang, Le Xue, Wenbo Zhang, Lanlan Li, Feiyang Xiao, Yuchen Liu, Xiaohui Zhang, Hongwei Zhang, Shuqi Wang, Gang Feng, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.11627v1  
  <details><summary>Abstract</summary>

  Positron emission tomography (PET) is a key nuclear medicine imaging modality that visualizes radiotracer distributions to quantify in vivo physiological and metabolic processes, playing an irreplaceable role in disease management. Despite its clinical importance, the development of deep learning models for quantitative PET image analysis remains severely limited, driven by both the inherent segmentation challenge from PET's paucity of anatomical contrast and the high costs of data acquisition and annotation. To bridge this gap, we develop generalist foundational models for universal segmentation from 3D whole-body PET imaging. We first build the largest and most comprehensive PET dataset to date, comprising 11041 3D whole-body PET scans with 59831 segmentation masks for model development. Based on this dataset, we present SegAnyPET, an innovative foundational model with general-purpose applicability to diverse segmentation tasks. Built on a 3D architecture with a prompt engineering strategy for mask generation, SegAnyPET enables universal and scalable organ and lesion segmentation, supports efficient human correction with minimal effort, and enables a clinical human-in-the-loop workflow. Extensive evaluations on multi-center, multi-tracer, multi-disease datasets demonstrate that SegAnyPET achieves strong zero-shot performance across a wide range of segmentation tasks, highlighting its potential to advance the clinical applications of molecular imaging.

  </details>



- **Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild**  
  Jiin Im, Sisung Liu, Je Hyeong Hong  
  _2026-03-12_ · https://arxiv.org/abs/2603.11618v1  
  <details><summary>Abstract</summary>

  Semantic correspondence is essential for handling diverse in-the-wild images lacking explicit correspondence annotations. While recent 2D foundation models offer powerful features, adapting them for unsupervised learning via nearest-neighbor pseudo-labels has key limitations: it operates locally, ignoring structural relationships, and consequently its reliance on 2D appearance fails to resolve geometric ambiguities arising from symmetries or repetitive features. In this work, we address this by reformulating pseudo-label generation as a Fused Gromov-Wasserstein (FGW) problem, which jointly optimizes inter-feature similarity and intra-structural consistency. Our framework, Shape-of-You (SoY), leverages a 3D foundation model to define this intra-structure in the geometric space, resolving abovementioned ambiguity. However, since FGW is a computationally prohibitive quadratic problem, we approximate it through anchor-based linearization. The resulting probabilistic transport plan provides a structurally consistent but noisy supervisory signal. Thus, we introduce a soft-target loss dynamically blending guidance from this plan with network predictions to build a learning framework robust to this noise. SoY achieves state-of-the-art performance on SPair-71k and AP-10k datasets, establishing a new benchmark in semantic correspondence without explicit geometric annotations. Code is available at Shape-of-You.

  </details>



- **SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation**  
  Muyi Sun, Yifan Gao, Ziang Jia, Xingqun Qi, Qianli Zhang, Qian Liu, Tianzheng Deng  
  _2026-03-12_ · https://arxiv.org/abs/2603.11616v1  
  <details><summary>Abstract</summary>

  With the rapid advancement of artificial intelligence, intelligent dentistry for clinical diagnosis and treatment has become increasingly promising. As the primary clinical dentistry task, tooth structure segmentation for Cone-Beam Computed Tomography (CBCT) has made significant progress in recent years. However, challenges arise from the obtainment difficulty of full-annotated data, and the acquisition variability of multi-source data across different institutions, which have caused low-quality utilization, voxel-level inconsistency, and domain-specific disparity in CBCT slices. Thus, the rational and efficient utilization of multi-source and unlabeled data represents a pivotal problem. In this paper, we propose SemiTooth, a generalizable semi-supervised framework for multi-source tooth segmentation. Specifically, we first compile MS3Toothset, Multi-Source Semi-Supervised Tooth DataSet for clinical dental CBCT, which contains data from three sources with different-level annotations. Then, we design a multi-teacher and multi-student framework, i.e., SemiTooth, which promotes semi-supervised learning for multi-source data. SemiTooth employs distinct student networks that learn from unlabeled data with different sources, supervised by its respective teachers. Furthermore, a Stricter Weighted-Confidence Constraint is introduced for multiple teachers to improve the multi-source accuracy.Extensive experiments are conducted on MS3Toothset to verify the feasibility and superiority of the SemiTooth framework, which achieves SOTA performance on the semi-supervised and multi-source tooth segmentation scenario.

  </details>


