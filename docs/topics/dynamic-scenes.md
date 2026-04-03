# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-03 07:35 UTC_

Total papers shown: **14**


---

- **GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding**  
  Rong Fan, Kaiyan Xiao, Minghao Zhu, Liuyi Wang, Kai Dai, Zhao Yang  
  _2026-04-02_ · https://arxiv.org/abs/2604.02093v1  
  <details><summary>Abstract</summary>

  Video temporal grounding (VTG) is a critical task in video understanding and a key capability for extending video large language models (Vid-LLMs) to broader applications. However, existing Vid-LLMs rely on uniform frame sampling to extract video information, resulting in a sparse distribution of key frames and the loss of crucial temporal cues. To address this limitation, we propose Grounded Visual Token Sampling (GroundVTS), a Vid-LLM architecture that focuses on the most informative temporal segments. GroundVTS employs a fine-grained, query-guided mechanism to filter visual tokens before feeding them into the LLM, thereby preserving essential spatio-temporal information and maintaining temporal coherence. Futhermore, we introduce a progressive optimization strategy that enables the LLM to effectively adapt to the non-uniform distribution of visual features, enhancing its ability to model temporal dependencies and achieve precise video localization. We comprehensively evaluate GroundVTS on three standard VTG benchmarks, where it outperforms existing methods, achieving a 7.7-point improvement in mIoU for moment retrieval and 12.0-point improvement in mAP for highlight detection. Code is available at https://github.com/Florence365/GroundVTS.

  </details>



- **MAVFusion: Efficient Infrared and Visible Video Fusion via Motion-Aware Sparse Interaction**  
  Xilai Li, Weijun Jiang, Xiaosong Li, Yang Liu, Hongbin Wang, Tao Ye, Huafeng Li, Haishu Tan  
  _2026-04-02_ · https://arxiv.org/abs/2604.01958v1  
  <details><summary>Abstract</summary>

  Infrared and visible video fusion combines the object saliency from infrared images with the texture details from visible images to produce semantically rich fusion results. However, most existing methods are designed for static image fusion and cannot effectively handle frame-to-frame motion in videos. Current video fusion methods improve temporal consistency by introducing interactions across frames, but they often require high computational cost. To mitigate these challenges, we propose MAVFusion, an end-to-end video fusion framework featuring a motion-aware sparse interaction mechanism that enhances efficiency while maintaining superior fusion quality. Specifically, we leverage optical flow to identify dynamic regions in multi-modal sequences, adaptively allocating computationally intensive cross-modal attention to these sparse areas to capture salient transitions and facilitate inter-modal information exchange. For static background regions, a lightweight weak interaction module is employed to maintain structural and appearance integrity. By decoupling the processing of dynamic and static regions, MAVFusion simultaneously preserves temporal consistency and fine-grained details while significantly accelerating inference. Extensive experiments demonstrate that MAVFusion achieves state-of-the-art performance on multiple infrared and visible video benchmarks, achieving a speed of 14.16\,FPS at $640 \times 480$ resolution. The source code will be available at https://github.com/ixilai/MAVFusion.

  </details>



- **FTPFusion: Frequency-Aware Infrared and Visible Video Fusion with Temporal Perturbation**  
  Xilai Li, Chusheng Fang, Xiaosong Li  
  _2026-04-02_ · https://arxiv.org/abs/2604.01900v1  
  <details><summary>Abstract</summary>

  Infrared and visible video fusion plays a critical role in intelligent surveillance and low-light monitoring. However, maintaining temporal stability while preserving spatial detail remains a fundamental challenge. Existing methods either focus on frame-wise enhancement with limited temporal modeling or rely on heavy spatio-temporal aggregation that often sacrifices high-frequency details. In this paper, we propose FTPFusion, a frequency-aware infrared and visible video fusion method based on temporal perturbation and sparse cross-modal interaction. Specifically, FTPFusion decomposes the feature representations into high-frequency and low-frequency components for collaborative modeling. The high-frequency branch performs sparse cross-modal spatio-temporal interaction to capture motion-related context and complementary details. The low-frequency branch introduces a temporal perturbation strategy to enhance robustness against complex video variations, such as flickering, jitter, and local misalignment. Furthermore, we design an offset-aware temporal consistency constraint to explicitly stabilize cross-frame representations under temporal disturbances. Extensive experiments on multiple public benchmarks demonstrate that FTPFusion consistently outperforms state-of-the-art methods across multiple metrics in both spatial fidelity and temporal consistency. The source code will be available at https://github.com/ixilai/FTPFusion.

  </details>



- **DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning**  
  Yang Zhou, Xiaofeng Wang, Hao Shao, Letian Wang, Guosheng Zhao, Jiangnan Shao, Jiagang Zhu, Tingdong Yu, Zheng Zhu, Guan Huang, et al.  
  _2026-04-02_ · https://arxiv.org/abs/2604.01765v1  
  <details><summary>Abstract</summary>

  Recently, world-action models (WAM) have emerged to bridge vision-language-action (VLA) models and world models, unifying their reasoning and instruction-following capabilities and spatio-temporal world modeling. However, existing WAM approaches often focus on modeling 2D appearance or latent representations, with limited geometric grounding-an essential element for embodied systems operating in the physical world. We present DriveDreamer-Policy, a unified driving world-action model that integrates depth generation, future video generation, and motion planning within a single modular architecture. The model employs a large language model to process language instructions, multi-view images, and actions, followed by three lightweight generators that produce depth, future video, and actions. By learning a geometry-aware world representation and using it to guide both future prediction and planning within a unified framework, the proposed model produces more coherent imagined futures and more informed driving actions, while maintaining modularity and controllable latency. Experiments on the Navsim v1 and v2 benchmarks demonstrate that DriveDreamer-Policy achieves strong performance on both closed-loop planning and world generation tasks. In particular, our model reaches 89.2 PDMS on Navsim v1 and 88.7 EPDMS on Navsim v2, outperforming existing world-model-based approaches while producing higher-quality future video and depth predictions. Ablation studies further show that explicit depth learning provides complementary benefits to video imagination and improves planning robustness.

  </details>



- **From Understanding to Erasing: Towards Complete and Stable Video Object Removal**  
  Dingming Liu, Wenjing Wang, Chen Li, Jing Lyu  
  _2026-04-02_ · https://arxiv.org/abs/2604.01693v1  
  <details><summary>Abstract</summary>

  Video object removal aims to eliminate target objects from videos while plausibly completing missing regions and preserving spatio-temporal consistency. Although diffusion models have recently advanced this task, it remains challenging to remove object-induced side effects (e.g., shadows, reflections, and illumination changes) without compromising overall coherence. This limitation stems from the insufficient physical and semantic understanding of the target object and its interactions with the scene. In this paper, we propose to introduce understanding into erasing from two complementary perspectives. Externally, we introduce a distillation scheme that transfers the relationships between objects and their induced effects from vision foundation models to video diffusion models. Internally, we propose a framewise context cross-attention mechanism that grounds each denoising block in informative, unmasked context surrounding the target region. External and internal guidance jointly enable our model to understand the target object, its induced effects, and the global background context, resulting in clear and coherent object removal. Extensive experiments demonstrate our state-of-the-art performance, and we establish the first real-world benchmark for video object removal to facilitate future research and community progress. Our code, data, and models are available at: https://github.com/WeChatCV/UnderEraser.

  </details>



- **Director: Instance-aware Gaussian Splatting for Dynamic Scene Modeling and Understanding**  
  Yuheng Jiang, Yiwen Cai, Zihao Wang, Yize Wu, Sicheng Li, Zhuo Su, Shaohui Jiao, Lan Xu  
  _2026-04-02_ · https://arxiv.org/abs/2604.01678v1  
  <details><summary>Abstract</summary>

  Volumetric video seeks to model dynamic scenes as temporally coherent 4D representations. While recent Gaussian-based approaches achieve impressive rendering fidelity, they primarily emphasize appearance but are largely agnostic to instance-level structure, limiting stable tracking and semantic reasoning in highly dynamic scenarios. In this paper, we present Director, a unified spatio-temporal Gaussian representation that jointly models human performance, high-fidelity rendering, and instance-level semantics. Our key insight is that embedding instance-consistent semantics naturally complements 4D modeling, enabling more accurate scene decomposition while supporting robust dynamic scene understanding. To this end, we leverage temporally aligned instance masks and sentence embeddings derived from Multimodal Large Language Models to supervise the learnable semantic features of each Gaussian via two MLP decoders, enabling language-aligned 4D representations and enforcing identity consistency over time. To enhance temporal stability, we bridge 2D optical flow with 4D Gaussians and finetune their motions, yielding reliable initialization and reducing drift. For the training, we further introduce a geometry-aware SDF constraints, along with regularization terms that enforces surface continuity, enhancing temporal coherence in dynamic foreground modeling. Experiments demonstrate that Director achieves temporally coherent 4D reconstructions while simultaneously enabling instance segmentation and open-vocabulary querying.

  </details>



- **VideoZeroBench: Probing the Limits of Video MLLMs with Spatio-Temporal Evidence Verification**  
  Jiahao Meng, Tan Yue, Qi Xu, Haochen Wang, Zhongwei Ren, Weisong Liu, Yuhao Wang, Renrui Zhang, Yunhai Tong, Haodong Duan  
  _2026-04-02_ · https://arxiv.org/abs/2604.01569v1  
  <details><summary>Abstract</summary>

  Recent video multimodal large language models achieve impressive results across various benchmarks. However, current evaluations suffer from two critical limitations: (1) inflated scores can mask deficiencies in fine-grained visual understanding and reasoning, and (2) answer correctness is often measured without verifying whether models identify the precise spatio-temporal evidence supporting their predictions. To address this, we present VideoZeroBench, a hierarchical benchmark designed for challenging long-video question answering that rigorously verifies spatio-temporal evidence. It comprises 500 manually annotated questions across 13 domains, paired with temporal intervals and spatial bounding boxes as evidence. To disentangle answering generation, temporal grounding, and spatial grounding, we introduce a five-level evaluation protocol that progressively tightens evidence requirements. Experiments show that even Gemini-3-Pro correctly answers fewer than 17% of questions under the standard end-to-end QA setting (Level-3). When grounding constraints are imposed, performance drops sharply: No model exceeds 1% accuracy when both correct answering and accurate spatio-temporal localization are required (Level-5), with most failing to achieve any correct grounded predictions. These results expose a significant gap between surface-level answer correctness and genuine evidence-based reasoning, revealing that grounded video understanding remains a bottleneck for long-video QA. We further analyze performance across minimal evidence spans, atomic abilities, and inference paradigms, providing insights for future research in grounded video reasoning. The benchmark and code will be made publicly available.

  </details>



- **ReFlow: Self-correction Motion Learning for Dynamic Scene Reconstruction**  
  Yanzhe Liang, Ruijie Zhu, Hanzhi Chang, Zhuoyuan Li, Jiahao Lu, Tianzhu Zhang  
  _2026-04-02_ · https://arxiv.org/abs/2604.01561v1  
  <details><summary>Abstract</summary>

  We present ReFlow, a unified framework for monocular dynamic scene reconstruction that learns 3D motion in a novel self-correction manner from raw video. Existing methods often suffer from incomplete scene initialization for dynamic regions, leading to unstable reconstruction and motion estimation, which often resorts to external dense motion guidance such as pre-computed optical flow to further stabilize and constrain the reconstruction of dynamic components. However, this introduces additional complexity and potential error propagation. To address these issues, ReFlow integrates a Complete Canonical Space Construction module for enhanced initialization of both static and dynamic regions, and a Separation-Based Dynamic Scene Modeling module that decouples static and dynamic components for targeted motion supervision. The core of ReFlow is a novel self-correction flow matching mechanism, consisting of Full Flow Matching to align 3D scene flow with time-varying 2D observations, and Camera Flow Matching to enforce multi-view consistency for static objects. Together, these modules enable robust and accurate dynamic scene reconstruction. Extensive experiments across diverse scenarios demonstrate that ReFlow achieves superior reconstruction quality and robustness, establishing a novel self-correction paradigm for monocular 4D reconstruction.

  </details>



- **UniRecGen: Unifying Multi-View 3D Reconstruction and Generation**  
  Zhisheng Huang, Jiahao Chen, Cheng Lin, Chenyu Hu, Hanzhuo Huang, Zhengming Yu, Mengfei Li, Yuheng Liu, Zekai Gu, Zibo Zhao, et al.  
  _2026-04-01_ · https://arxiv.org/abs/2604.01479v1  
  <details><summary>Abstract</summary>

  Sparse-view 3D modeling represents a fundamental tension between reconstruction fidelity and generative plausibility. While feed-forward reconstruction excels in efficiency and input alignment, it often lacks the global priors needed for structural completeness. Conversely, diffusion-based generation provides rich geometric details but struggles with multi-view consistency. We present UniRecGen, a unified framework that integrates these two paradigms into a single cooperative system. To overcome inherent conflicts in coordinate spaces, 3D representations, and training objectives, we align both models within a shared canonical space. We employ disentangled cooperative learning, which maintains stable training while enabling seamless collaboration during inference. Specifically, the reconstruction module is adapted to provide canonical geometric anchors, while the diffusion generator leverages latent-augmented conditioning to refine and complete the geometric structure. Experimental results demonstrate that UniRecGen achieves superior fidelity and robustness, outperforming existing methods in creating complete and consistent 3D models from sparse observations.

  </details>



- **GRAZE: Grounded Refinement and Motion-Aware Zero-Shot Event Localization**  
  Syed Ahsan Masud Zaidi, Lior Shamir, William Hsu, Scott Dietrich, Talha Zaidi  
  _2026-04-01_ · https://arxiv.org/abs/2604.01383v1  
  <details><summary>Abstract</summary>

  American football practice generates video at scale, yet the interaction of interest occupies only a brief window of each long, untrimmed clip. Reliable biomechanical analysis, therefore, depends on spatiotemporal localization that identifies both the interacting entities and the onset of contact. We study First Point of Contact (FPOC), defined as the first frame in which a player physically touches a tackle dummy, in unconstrained practice footage with camera motion, clutter, multiple similarly equipped athletes, and rapid pose changes around impact. We present GRAZE, a training-free pipeline for FPOC localization that requires no labeled tackle-contact examples. GRAZE uses Grounding DINO to discover candidate player-dummy interactions, refines them with motion-aware temporal reasoning, and uses SAM2 as an explicit pixel-level verifier of contact rather than relying on detection confidence alone. This separation between candidate discovery and contact confirmation makes the approach robust to cluttered scenes and unstable grounding near impact. On 738 tackle-practice videos, GRAZE produces valid outputs for 97.4% of clips and localizes FPOC within $\pm$ 10 frames on 77.5% of all clips and within $\pm$ 20 frames on 82.7% of all clips. These results show that frame-accurate contact onset localization in real-world practice footage is feasible without task-specific training.

  </details>



- **LAtent Phase Inference from Short time sequences using SHallow REcurrent Decoders (LAPIS-SHRED)**  
  Yuxuan Bao, Xingyue Zhang, J. Nathan Kutz  
  _2026-04-01_ · https://arxiv.org/abs/2604.01216v1  
  <details><summary>Abstract</summary>

  Reconstructing full spatio-temporal dynamics from sparse observations in both space and time remains a central challenge in complex systems, as measurements can be spatially incomplete and can be also limited to narrow temporal windows. Yet approximating the complete spatio-temporal trajectory is essential for mechanistic insight and understanding, model calibration, and operational decision-making. We introduce LAPIS-SHRED (LAtent Phase Inference from Short time sequence using SHallow REcurrent Decoders), a modular architecture that reconstructs and/or forecasts complete spatiotemporal dynamics from sparse sensor observations confined to short temporal windows. LAPIS-SHRED operates through a three-stage pipeline: (i) a SHRED model is pre-trained entirely on simulation data to map sensor time-histories into a structured latent space, (ii) a temporal sequence model, trained on simulation-derived latent trajectories, learns to propagate latent states forward or backward in time to span unobserved temporal regions from short observational time windows, and (iii) at deployment, only a short observation window of hyper-sparse sensor measurements from the true system is provided, from which the frozen SHRED model and the temporal model jointly reconstruct or forecast the complete spatiotemporal trajectory. The framework supports bidirectional inference, inherits data assimilation and multiscale reconstruction capabilities from its modular structure, and accommodates extreme observational constraints including single-frame terminal inputs. We evaluate LAPIS-SHRED on six experiments spanning complex spatio-temporal physics: turbulent flows, multiscale propulsion physics, volatile combustion transients, and satellite-derived environmental fields, highlighting a lightweight, modular architecture suited for operational settings where observation is constrained by physical or logistical limitations.

  </details>



- **ACT Now: Preempting LVLM Hallucinations via Adaptive Context Integration**  
  Bei Yan, Yuecong Min, Jie Zhang, Shiguang Shan, Xilin Chen  
  _2026-04-01_ · https://arxiv.org/abs/2604.00983v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (LVLMs) frequently suffer from severe hallucination issues. Existing mitigation strategies predominantly rely on isolated, single-step states to enhance visual focus or suppress strong linguistic priors. However, these static approaches neglect dynamic context changes across the generation process and struggles to correct inherited information loss. To address this limitation, we propose Adaptive Context inTegration (ACT), a training-free inference intervention method that mitigates hallucination through the adaptive integration of contextual information. Specifically, we first propose visual context exploration, which leverages spatio-temporal profiling to adaptively amplify attention heads responsible for visual exploration. To further facilitate vision-language alignment, we propose semantic context aggregation that marginalizes potential semantic queries to effectively aggregate visual evidence, thereby resolving the information loss caused by the discrete nature of token prediction. Extensive experiments across diverse LVLMs demonstrate that ACT significantly reduces hallucinations and achieves competitive results on both discriminative and generative benchmarks, acting as a robust and highly adaptable solution without compromising fundamental generation capabilities.

  </details>



- **Learning Quantised Structure-Preserving Motion Representations for Dance Fingerprinting**  
  Arina Kharlamova, Bowei He, Chen Ma, Xue Liu  
  _2026-04-01_ · https://arxiv.org/abs/2604.00927v1  
  <details><summary>Abstract</summary>

  We present DANCEMATCH, an end-to-end framework for motion-based dance retrieval, the task of identifying semantically similar choreographies directly from raw video, defined as DANCE FINGERPRINTING. While existing motion analysis and retrieval methods can compare pose sequences, they rely on continuous embeddings that are difficult to index, interpret, or scale. In contrast, DANCEMATCH constructs compact, discrete motion signatures that capture the spatio-temporal structure of dance while enabling efficient large-scale retrieval. Our system integrates Skeleton Motion Quantisation (SMQ) with Spatio-Temporal Transformers (STT) to encode human poses, extracted via Apple CoMotion, into a structured motion vocabulary. We further design DANCE RETRIEVAL ENGINE (DRE), which performs sub-linear retrieval using a histogram-based index followed by re-ranking for refined matching. To facilitate reproducible research, we release DANCETYPESBENCHMARK, a pose-aligned dataset annotated with quantised motion tokens. Experiments demonstrate robust retrieval across diverse dance styles and strong generalisation to unseen choreographies, establishing a foundation for scalable motion fingerprinting and quantitative choreographic analysis.

  </details>



- **TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting**  
  Suwoong Yeom, Joonsik Nam, Seunggyu Choi, Lucas Yunkyu Lee, Sangmin Kim, Jaesik Park, Joonsoo Kim, Kugjin Yun, Kyeongbo Kong, Sukju Kang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00538v1  
  <details><summary>Abstract</summary>

  Recent 4D Gaussian Splatting (4DGS) methods achieve impressive dynamic scene reconstruction but often rely on piecewise linear velocity approximations and short temporal windows. This disjointed modeling leads to severe temporal fragmentation, forcing primitives to be repeatedly eliminated and regenerated to track complex nonlinear dynamics. This makeshift approximation eliminates the long-term temporal identity of objects and causes an inevitable proliferation of Gaussians, hindering scalability to extended video sequences. To address this, we propose TRiGS, a novel 4D representation that utilizes unified, continuous geometric transformations. By integrating $SE(3)$ transformations, hierarchical Bezier residuals, and learnable local anchors, TRiGS models geometrically consistent rigid motions for individual primitives. This continuous formulation preserves temporal identity and effectively mitigates unbounded memory growth. Extensive experiments demonstrate that TRiGS achieves high fidelity rendering on standard benchmarks while uniquely scaling to extended video sequences (e.g., 600 to 1200 frames) without severe memory bottlenecks, significantly outperforming prior works in temporal stability.

  </details>


