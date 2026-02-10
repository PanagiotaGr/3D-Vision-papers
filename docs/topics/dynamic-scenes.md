# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-10 07:20 UTC_

Total papers shown: **10**


---

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
  _2026-02-09_ · https://arxiv.org/abs/2602.08958v1  
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



- **CAE-AV: Improving Audio-Visual Learning via Cross-modal Interactive Enrichment**  
  Yunzuo Hu, Wen Li, Jing Zhang  
  _2026-02-09_ · https://arxiv.org/abs/2602.08309v1  
  <details><summary>Abstract</summary>

  Audio-visual learning suffers from modality misalignment caused by off-screen sources and background clutter, and current methods usually amplify irrelevant regions or moments, leading to unstable training and degraded representation quality. To address this challenge, we proposed a novel Caption-aligned and Agreement-guided Enhancement framework (CAE-AV) for audio-visual learning, which used two complementary modules: Cross-modal Agreement-guided Spatio-Temporal Enrichment (CASTE) and Caption-Aligned Saliency-guided Enrichment (CASE) to relieve audio-visual misalignment. CASTE dynamically balances spatial and temporal relations by evaluating frame-level audio-visual agreement, ensuring that key information is captured from both preceding and subsequent frames under misalignment. CASE injects cross-modal semantic guidance into selected spatio-temporal positions, leveraging high-level semantic cues to further alleviate misalignment. In addition, we design lightweight objectives, caption-to-modality InfoNCE, visual-audio consistency, and entropy regularization to guide token selection and strengthen cross-modal semantic alignment. With frozen backbones, CAE-AV achieves state-of-the-art performance on AVE, AVVP, AVS, and AVQA benchmarks, and qualitative analyses further validate its robustness against audio-visual misalignment.

  </details>



- **Generating Adversarial Events: A Motion-Aware Point Cloud Framework**  
  Hongwei Ren, Youxin Jiang, Qifei Gu, Xiangqian Wu  
  _2026-02-09_ · https://arxiv.org/abs/2602.08230v1  
  <details><summary>Abstract</summary>

  Event cameras have been widely adopted in safety-critical domains such as autonomous driving, robotics, and human-computer interaction. A pressing challenge arises from the vulnerability of deep neural networks to adversarial examples, which poses a significant threat to the reliability of event-based systems. Nevertheless, research into adversarial attacks on events is scarce. This is primarily due to the non-differentiable nature of mainstream event representations, which hinders the extension of gradient-based attack methods. In this paper, we propose MA-ADV, a novel \textbf{M}otion-\textbf{A}ware \textbf{Adv}ersarial framework. To the best of our knowledge, this is the first work to generate adversarial events by leveraging point cloud representations. MA-ADV accounts for high-frequency noise in events and employs a diffusion-based approach to smooth perturbations, while fully leveraging the spatial and temporal relationships among events. Finally, MA-ADV identifies the minimal-cost perturbation through a combination of sample-wise Adam optimization, iterative refinement, and binary search. Extensive experimental results validate that MA-ADV ensures a 100\% attack success rate with minimal perturbation cost, and also demonstrate enhanced robustness against defenses, underscoring the critical security challenges facing future event-based perception systems.

  </details>



- **ForecastOcc: Vision-based Semantic Occupancy Forecasting**  
  Riya Mohan, Juana Valeria Hurtado, Rohit Mohan, Abhinav Valada  
  _2026-02-08_ · https://arxiv.org/abs/2602.08006v1  
  <details><summary>Abstract</summary>

  Autonomous driving requires forecasting both geometry and semantics over time to effectively reason about future environment states. Existing vision-based occupancy forecasting methods focus on motion-related categories such as static and dynamic objects, while semantic information remains largely absent. Recent semantic occupancy forecasting approaches address this gap but rely on past occupancy predictions obtained from separate networks. This makes current methods sensitive to error accumulation and prevents learning spatio-temporal features directly from images. In this work, we present ForecastOcc, the first framework for vision-based semantic occupancy forecasting that jointly predicts future occupancy states and semantic categories. Our framework yields semantic occupancy forecasts for multiple horizons directly from past camera images, without relying on externally estimated maps. We evaluate ForecastOcc in two complementary settings: multi-view forecasting on the Occ3D-nuScenes dataset and monocular forecasting on SemanticKITTI, where we establish the first benchmark for this task. We introduce the first baselines by adapting two 2D forecasting modules within our framework. Importantly, we propose a novel architecture that incorporates a temporal cross-attention forecasting module, a 2D-to-3D view transformer, a 3D encoder for occupancy prediction, and a semantic occupancy head for voxel-level forecasts across multiple horizons. Extensive experiments on both datasets show that ForecastOcc consistently outperforms baselines, yielding semantically rich, future-aware predictions that capture scene dynamics and semantics critical for autonomous driving.

  </details>



- **Integrating Specialized and Generic Agent Motion Prediction with Dynamic Occupancy Grid Maps**  
  Rabbia Asghar, Lukas Rummelhard, Wenqian Liu, Anne Spalanzani, Christian Laugier  
  _2026-02-08_ · https://arxiv.org/abs/2602.07938v1  
  <details><summary>Abstract</summary>

  Accurate prediction of driving scene is a challenging task due to uncertainty in sensor data, the complex behaviors of agents, and the possibility of multiple feasible futures. Existing prediction methods using occupancy grid maps primarily focus on agent-agnostic scene predictions, while agent-specific predictions provide specialized behavior insights with the help of semantic information. However, both paradigms face distinct limitations: agent-agnostic models struggle to capture the behavioral complexities of dynamic actors, whereas agent-specific approaches fail to generalize to poorly perceived or unrecognized agents; combining both enables robust and safer motion forecasting. To address this, we propose a unified framework by leveraging Dynamic Occupancy Grid Maps within a streamlined temporal decoding pipeline to simultaneously predict future occupancy state grids, vehicle grids, and scene flow grids. Relying on a lightweight spatiotemporal backbone, our approach is centered on a tailored, interdependent loss function that captures inter-grid dependencies and enables diverse future predictions. By using occupancy state information to enforce flow-guided transitions, the loss function acts as a regularizer that directs occupancy evolution while accounting for obstacles and occlusions. Consequently, the model not only predicts the specific behaviors of vehicle agents, but also identifies other dynamic entities and anticipates their evolution within the complex scene. Evaluations on real-world nuScenes and Woven Planet datasets demonstrate superior prediction performances for dynamic vehicles and generic dynamic scene elements compared to baseline methods.

  </details>


