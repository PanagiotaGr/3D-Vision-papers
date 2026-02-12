# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-12 07:16 UTC_

Total papers shown: **12**


---

- **From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?**  
  Krishna Kanth Nakka, Vedasri Nakka  
  _2026-02-11_ · https://arxiv.org/abs/2602.10771v1  
  <details><summary>Abstract</summary>

  Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.

  </details>



- **C^2ROPE: Causal Continuous Rotary Positional Encoding for 3D Large Multimodal-Models Reasoning**  
  Guanting Ye, Qiyan Zhao, Wenhao Yu, Xiaofeng Zhang, Jianmin Ji, Yanyong Zhang, Ka-Veng Yuen  
  _2026-02-11_ · https://arxiv.org/abs/2602.10551v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Large Multimodal Models (LMMs) built on Large Language Models (LLMs) have established the alignment of 3D visual features with LLM representations as the dominant paradigm. However, the inherited Rotary Position Embedding (RoPE) introduces limitations for multimodal processing. Specifically, applying 1D temporal positional indices disrupts the continuity of visual features along the column dimension, resulting in spatial locality loss. Moreover, RoPE follows the prior that temporally closer image tokens are more causally related, leading to long-term decay in attention allocation and causing the model to progressively neglect earlier visual tokens as the sequence length increases. To address these issues, we propose C^2RoPE, an improved RoPE that explicitly models local spatial Continuity and spatial Causal relationships for visual processing. C^2RoPE introduces a spatio-temporal continuous positional embedding mechanism for visual tokens. It first integrates 1D temporal positions with Cartesian-based spatial coordinates to construct a triplet hybrid positional index, and then employs a frequency allocation strategy to encode spatio-temporal positional information across the three index components. Additionally, we introduce Chebyshev Causal Masking, which determines causal dependencies by computing the Chebyshev distance of image tokens in 2D space. Evaluation results across various benchmarks, including 3D scene reasoning and 3D visual question answering, demonstrate C^2RoPE's effectiveness. The code is be available at https://github.com/ErikZ719/C2RoPE.

  </details>



- **ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters**  
  Reese Kneeland, Wangshu Jiang, Ugo Bruzadin Nunes, Paul Steven Scotti, Arnaud Delorme, Jonathan Xu  
  _2026-02-10_ · https://arxiv.org/abs/2602.10361v1  
  <details><summary>Abstract</summary>

  To be practical for real-life applications, models for brain-computer interfaces must be easily and quickly deployable on new subjects, effective on affordable scanning hardware, and small enough to run locally on accessible computing resources. To directly address these current limitations, we introduce ENIGMA, a multi-subject electroencephalography (EEG)-to-Image decoding model that reconstructs seen images from EEG recordings and achieves state-of-the-art (SOTA) performance on the research-grade THINGS-EEG2 and consumer-grade AllJoined-1.6M benchmarks, while fine-tuning effectively on new subjects with as little as 15 minutes of data. ENIGMA boasts a simpler architecture and requires less than 1% of the trainable parameters necessary for previous approaches. Our approach integrates a subject-unified spatio-temporal backbone along with a set of multi-subject latent alignment layers and an MLP projector to map raw EEG signals to a rich visual latent space. We evaluate our approach using a broad suite of image reconstruction metrics that have been standardized in the adjacent field of fMRI-to-Image research, and we describe the first EEG-to-Image study to conduct extensive behavioral evaluations of our reconstructions using human raters. Our simple and robust architecture provides a significant performance boost across both research-grade and consumer-grade EEG hardware, and a substantial improvement in fine-tuning efficiency and inference cost. Finally, we provide extensive ablations to determine the architectural choices most responsible for our performance gains in both single and multi-subject cases across multiple benchmark datasets. Collectively, our work provides a substantial step towards the development of practical brain-computer interface applications.

  </details>



- **4RC: 4D Reconstruction via Conditional Querying Anytime and Anywhere**  
  Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, Chen Change Loy  
  _2026-02-10_ · https://arxiv.org/abs/2602.10094v1  
  <details><summary>Abstract</summary>

  We present 4RC, a unified feed-forward framework for 4D reconstruction from monocular videos. Unlike existing approaches that typically decouple motion from geometry or produce limited 4D attributes such as sparse trajectories or two-view scene flow, 4RC learns a holistic 4D representation that jointly captures dense scene geometry and motion dynamics. At its core, 4RC introduces a novel encode-once, query-anywhere and anytime paradigm: a transformer backbone encodes the entire video into a compact spatio-temporal latent space, from which a conditional decoder can efficiently query 3D geometry and motion for any query frame at any target timestamp. To facilitate learning, we represent per-view 4D attributes in a minimally factorized form by decomposing them into base geometry and time-dependent relative motion. Extensive experiments demonstrate that 4RC outperforms prior and concurrent methods across a wide range of 4D reconstruction tasks.

  </details>



- **Spatio-Temporal Attention for Consistent Video Semantic Segmentation in Automated Driving**  
  Serin Varghese, Kevin Ross, Fabian Hueger, Kira Maag  
  _2026-02-10_ · https://arxiv.org/abs/2602.10052v1  
  <details><summary>Abstract</summary>

  Deep neural networks, especially transformer-based architectures, have achieved remarkable success in semantic segmentation for environmental perception. However, existing models process video frames independently, thus failing to leverage temporal consistency, which could significantly improve both accuracy and stability in dynamic scenes. In this work, we propose a Spatio-Temporal Attention (STA) mechanism that extends transformer attention blocks to incorporate multi-frame context, enabling robust temporal feature representations for video semantic segmentation. Our approach modifies standard self-attention to process spatio-temporal feature sequences while maintaining computational efficiency and requiring minimal changes to existing architectures. STA demonstrates broad applicability across diverse transformer architectures and remains effective across both lightweight and larger-scale models. A comprehensive evaluation on the Cityscapes and BDD100k datasets shows substantial improvements of 9.20 percentage points in temporal consistency metrics and up to 1.76 percentage points in mean intersection over union compared to single-frame baselines. These results demonstrate STA as an effective architectural enhancement for video-based semantic segmentation applications.

  </details>



- **Time2General: Learning Spatiotemporal Invariant Representations for Domain-Generalization Video Semantic Segmentation**  
  Siyu Chen, Ting Han, Haoling Huang, Chaolei Wang, Chengzheng Fu, Duxin Zhu, Guorong Cai, Jinhe Su  
  _2026-02-10_ · https://arxiv.org/abs/2602.09648v1  
  <details><summary>Abstract</summary>

  Domain Generalized Video Semantic Segmentation (DGVSS) is trained on a single labeled driving domain and is directly deployed on unseen domains without target labels and test-time adaptation while maintaining temporally consistent predictions over video streams. In practice, both domain shift and temporal-sampling shift break correspondence-based propagation and fixed-stride temporal aggregation, causing severe frame-to-frame flicker even in label-stable regions. We propose Time2General, a DGVSS framework built on Stability Queries. Time2General introduces a Spatio-Temporal Memory Decoder that aggregates multi-frame context into a clip-level spatio-temporal memory and decodes temporally consistent per-frame masks without explicit correspondence propagation. To further suppress flicker and improve robustness to varying sampling rates, the Masked Temporal Consistency Loss is proposed to regularize temporal prediction discrepancies across different strides, and randomize training strides to expose the model to diverse temporal gaps. Extensive experiments on multiple driving benchmarks show that Time2General achieves a substantial improvement in cross-domain accuracy and temporal stability over prior DGSS and VSS baselines while running at up to 18 FPS. Code will be released after the review process.

  </details>



- **$χ_{0}$: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies**  
  Checheng Yu, Chonghao Sima, Gangcheng Jiang, Hai Zhang, Haoguang Mai, Hongyang Li, Huijie Wang, Jin Chen, Kaiyang Wu, Li Chen, et al.  
  _2026-02-09_ · https://arxiv.org/abs/2602.09021v1  
  <details><summary>Abstract</summary>

  High-reliability long-horizon robotic manipulation has traditionally relied on large-scale data and compute to understand complex real-world dynamics. However, we identify that the primary bottleneck to real-world robustness is not resource scale alone, but the distributional shift among the human demonstration distribution, the inductive bias learned by the policy, and the test-time execution distribution -- a systematic inconsistency that causes compounding errors in multi-stage tasks. To mitigate these inconsistencies, we propose $χ_{0}$, a resource-efficient framework with effective modules designated to achieve production-level robustness in robotic manipulation. Our approach builds off three technical pillars: (i) Model Arithmetic, a weight-space merging strategy that efficiently soaks up diverse distributions of different demonstrations, varying from object appearance to state variations; (ii) Stage Advantage, a stage-aware advantage estimator that provides stable, dense progress signals, overcoming the numerical instability of prior non-stage approaches; and (iii) Train-Deploy Alignment, which bridges the distribution gap via spatio-temporal augmentation, heuristic DAgger corrections, and temporal chunk-wise smoothing. $χ_{0}$ enables two sets of dual-arm robots to collaboratively orchestrate long-horizon garment manipulation, spanning tasks from flattening, folding, to hanging different clothes. Our method exhibits high-reliability autonomy; we are able to run the system from arbitrary initial state for consecutive 24 hours non-stop. Experiments validate that $χ_{0}$ surpasses the state-of-the-art $π_{0.5}$ in success rate by nearly 250%, with only 20-hour data and 8 A100 GPUs. Code, data and models will be released to facilitate the community.

  </details>



- **MotionCrafter: Dense Geometry and Motion Reconstruction with a 4D VAE**  
  Ruijie Zhu, Jiahao Lu, Wenbo Hu, Xiaoguang Han, Jianfei Cai, Ying Shan, Chuanxia Zheng  
  _2026-02-09_ · https://arxiv.org/abs/2602.08961v1  
  <details><summary>Abstract</summary>

  We introduce MotionCrafter, a video diffusion-based framework that jointly reconstructs 4D geometry and estimates dense motion from a monocular video. The core of our method is a novel joint representation of dense 3D point maps and 3D scene flows in a shared coordinate system, and a novel 4D VAE to effectively learn this representation. Unlike prior work that forces the 3D value and latents to align strictly with RGB VAE latents-despite their fundamentally different distributions-we show that such alignment is unnecessary and leads to suboptimal performance. Instead, we introduce a new data normalization and VAE training strategy that better transfers diffusion priors and greatly improves reconstruction quality. Extensive experiments across multiple datasets demonstrate that MotionCrafter achieves state-of-the-art performance in both geometry reconstruction and dense scene flow estimation, delivering 38.64% and 25.0% improvements in geometry and motion reconstruction, respectively, all without any post-optimization. Project page: https://ruijiezhu94.github.io/MotionCrafter_Page

  </details>



- **Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields**  
  Weihan Luo, Lily Goli, Sherwin Bahmani, Felix Taubner, Andrea Tagliasacchi, David B. Lindell  
  _2026-02-09_ · https://arxiv.org/abs/2602.08958v2  
  <details><summary>Abstract</summary>

  Modeling the time-varying 3D appearance of plants during their growth poses unique challenges: unlike many dynamic scenes, plants generate new geometry over time as they expand, branch, and differentiate. Recent motion modeling techniques are ill-suited to this problem setting. For example, deformation fields cannot introduce new geometry, and 4D Gaussian splatting constrains motion to a linear trajectory in space and time and cannot track the same set of Gaussians over time. Here, we introduce a 3D Gaussian flow field representation that models plant growth as a time-varying derivative over Gaussian parameters -- position, scale, orientation, color, and opacity -- enabling nonlinear and continuous-time growth dynamics. To initialize a sufficient set of Gaussian primitives, we reconstruct the mature plant and learn a process of reverse growth, effectively simulating the plant's developmental history in reverse. Our approach achieves superior image quality and geometric accuracy compared to prior methods on multi-view timelapse datasets of plant growth, providing a new approach for appearance modeling of growing 3D structures.

  </details>



- **WiFlow: A Lightweight WiFi-based Continuous Human Pose Estimation Network with Spatio-Temporal Feature Decoupling**  
  Yi Dao, Lankai Zhang, Hao Liu, Haiwei Zhang, Wenbo Wang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08661v1  
  <details><summary>Abstract</summary>

  Human pose estimation is fundamental to intelligent perception in the Internet of Things (IoT), enabling applications ranging from smart healthcare to human-computer interaction. While WiFi-based methods have gained traction, they often struggle with continuous motion and high computational overhead. This work presents WiFlow, a novel framework for continuous human pose estimation using WiFi signals. Unlike vision-based approaches such as two-dimensional deep residual networks that treat Channel State Information (CSI) as images, WiFlow employs an encoder-decoder architecture. The encoder captures spatio-temporal features of CSI using temporal and asymmetric convolutions, preserving the original sequential structure of signals. It then refines keypoint features of human bodies to be tracked and capture their structural dependencies via axial attention. The decoder subsequently maps the encoded high-dimensional features into keypoint coordinates. Trained on a self-collected dataset of 360,000 synchronized CSI-pose samples from 5 subjects performing continuous sequences of 8 daily activities, WiFlow achieves a Percentage of Correct Keypoints (PCK) of 97.00% at a threshold of 20% (PCK@20) and 99.48% at PCK@50, with a mean per-joint position error of 0.008m. With only 4.82M parameters, WiFlow significantly reduces model complexity and computational cost, establishing a new performance baseline for practical WiFi-based human pose estimation. Our code and datasets are available at https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling.git.

  </details>



- **FLAG-4D: Flow-Guided Local-Global Dual-Deformation Model for 4D Reconstruction**  
  Guan Yuan Tan, Ngoc Tuan Vu, Arghya Pal, Sailaja Rajanala, Raphael Phan C. -W., Mettu Srinivas, Chee-Ming Ting  
  _2026-02-09_ · https://arxiv.org/abs/2602.08558v1  
  <details><summary>Abstract</summary>

  We introduce FLAG-4D, a novel framework for generating novel views of dynamic scenes by reconstructing how 3D Gaussian primitives evolve through space and time. Existing methods typically rely on a single Multilayer Perceptron (MLP) to model temporal deformations, and they often struggle to capture complex point motions and fine-grained dynamic details consistently over time, especially from sparse input views. Our approach, FLAG-4D, overcomes this by employing a dual-deformation network that dynamically warps a canonical set of 3D Gaussians over time into new positions and anisotropic shapes. This dual-deformation network consists of an Instantaneous Deformation Network (IDN) for modeling fine-grained, local deformations and a Global Motion Network (GMN) for capturing long-range dynamics, refined through mutual learning. To ensure these deformations are both accurate and temporally smooth, FLAG-4D incorporates dense motion features from a pretrained optical flow backbone. We fuse these motion cues from adjacent timeframes and use a deformation-guided attention mechanism to align this flow information with the current state of each evolving 3D Gaussian. Extensive experiments demonstrate that FLAG-4D achieves higher-fidelity and more temporally coherent reconstructions with finer detail preservation than state-of-the-art methods.

  </details>



- **BiManiBench: A Hierarchical Benchmark for Evaluating Bimanual Coordination of Multimodal Large Language Models**  
  Xin Wu, Zhixuan Liang, Yue Ma, Mengkang Hu, Zhiyuan Qin, Xiu Li  
  _2026-02-09_ · https://arxiv.org/abs/2602.08392v1  
  <details><summary>Abstract</summary>

  Multimodal Large Language Models (MLLMs) have significantly advanced embodied AI, and using them to benchmark robotic intelligence has become a pivotal trend. However, existing frameworks remain predominantly confined to single-arm manipulation, failing to capture the spatio-temporal coordination required for bimanual tasks like lifting a heavy pot. To address this, we introduce BiManiBench, a hierarchical benchmark evaluating MLLMs across three tiers: fundamental spatial reasoning, high-level action planning, and low-level end-effector control. Our framework isolates unique bimanual challenges, such as arm reachability and kinematic constraints, thereby distinguishing perceptual hallucinations from planning failures. Analysis of over 30 state-of-the-art models reveals that despite high-level reasoning proficiency, MLLMs struggle with dual-arm spatial grounding and control, frequently resulting in mutual interference and sequencing errors. These findings suggest the current paradigm lacks a deep understanding of mutual kinematic constraints, highlighting the need for future research to focus on inter-arm collision-avoidance and fine-grained temporal sequencing.

  </details>


