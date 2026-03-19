# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-19 07:12 UTC_

Total papers shown: **13**


---

- **Unified Spatio-Temporal Token Scoring for Efficient Video VLMs**  
  Jianrui Zhang, Yue Yang, Rohun Tripathi, Winson Han, Ranjay Krishna, Christopher Clark, Yong Jae Lee, Sangho Lee  
  _2026-03-18_ · https://arxiv.org/abs/2603.18004v1  
  <details><summary>Abstract</summary>

  Token pruning is essential for enhancing the computational efficiency of vision-language models (VLMs), particularly for video-based tasks where temporal redundancy is prevalent. Prior approaches typically prune tokens either (1) within the vision transformer (ViT) exclusively for unimodal perception tasks such as action recognition and object segmentation, without adapting to downstream vision-language tasks; or (2) only within the LLM while leaving the ViT output intact, often requiring complex text-conditioned token selection mechanisms. In this paper, we introduce Spatio-Temporal Token Scoring (STTS), a simple and lightweight module that prunes vision tokens across both the ViT and the LLM without text conditioning or token merging, and is fully compatible with end-to-end training. By learning how to score temporally via an auxiliary loss and spatially via LLM downstream gradients, aided by our efficient packing algorithm, STTS prunes 50% of vision tokens throughout the entire architecture, resulting in a 62% improvement in efficiency during both training and inference with only a 0.7% drop in average performance across 13 short and long video QA tasks. Efficiency gains increase with more sampled frames per video. Applying test-time scaling for long-video QA further yields performance gains of 0.5-1% compared to the baseline. Overall, STTS represents a novel, simple yet effective technique for unified, architecture-wide vision token pruning.

  </details>



- **Feeling the Space: Egomotion-Aware Video Representation for Efficient and Accurate 3D Scene Understanding**  
  Shuyao Shi, Kang G. Shin  
  _2026-03-18_ · https://arxiv.org/abs/2603.17980v1  
  <details><summary>Abstract</summary>

  Recent Multimodal Large Language Models (MLLMs) have shown high potential for spatial reasoning within 3D scenes. However, they typically rely on computationally expensive 3D representations like point clouds or reconstructed Bird's-Eye View (BEV) maps, or lack physical grounding to resolve ambiguities in scale and size. This paper significantly enhances MLLMs with egomotion modality data, captured by Inertial Measurement Units (IMUs) concurrently with the video. In particular, we propose a novel framework, called Motion-MLLM, introducing two key components: (1) a cascaded motion-visual keyframe filtering module that leverages both IMU data and visual features to efficiently select a sparse yet representative set of keyframes, and (2) an asymmetric cross-modal fusion module where motion tokens serve as intermediaries that channel egomotion cues and cross-frame visual context into the visual representation. By grounding visual content in physical egomotion trajectories, Motion-MLLM can reason about absolute scale and spatial relationships across the scene. Our extensive evaluation shows that Motion-MLLM makes significant improvements in various tasks related to 3D scene understanding and spatial reasoning. Compared to state-of-the-art (SOTA) methods based on video frames and explicit 3D data, Motion-MLLM exhibits similar or even higher accuracy with significantly less overhead (i.e., 1.40$\times$ and 1.63$\times$ higher cost-effectiveness, respectively).

  </details>



- **VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs**  
  Chaokang Jiang, Desen Zhou, Jiuming Liu, Kevin Li Sun  
  _2026-03-18_ · https://arxiv.org/abs/2603.17652v1  
  <details><summary>Abstract</summary>

  Closed-loop evaluation of autonomous-driving policies requires interactive simulation beyond log replay. However, existing generative world models often degrade in closed loop due to (i) history-free initialization that mismatches policy inputs, (ii) multi-step sampling latency that violates real-time budgets, and (iii) compounding kinematic infeasibility over long horizons. We propose VectorWorld, a streaming world model that incrementally generates ego-centric $64 \mathrm{m}\times 64\mathrm{m}$ lane--agent vector-graph tiles during rollout. VectorWorld aligns initialization with history-conditioned policies by producing a policy-compatible interaction state via a motion-aware gated VAE. It enables real-time outpainting via solver-free one-step masked completion with an edge-gated relational DiT trained with interval-conditioned MeanFlow and JVP-based large-step supervision. To stabilize long-horizon rollouts, we introduce $Δ$Sim, a physics-aligned non-ego (NPC) policy with hybrid discrete--continuous actions and differentiable kinematic logit shaping. On Waymo open motion and nuPlan, VectorWorld improves map-structure fidelity and initialization validity, and supports stable, real-time $1\mathrm{km}+$ closed-loop rollouts (\href{https://github.com/jiangchaokang/VectorWorld}{code}).

  </details>



- **Towards Motion-aware Referring Image Segmentation**  
  Chaeyun Kim, Seunghoon Yi, Yejin Kim, Yohan Jo, Joonseok Lee  
  _2026-03-18_ · https://arxiv.org/abs/2603.17413v1  
  <details><summary>Abstract</summary>

  Referring Image Segmentation (RIS) requires identifying objects from images based on textual descriptions. We observe that existing methods significantly underperform on motion-related queries compared to appearance-based ones. To address this, we first introduce an efficient data augmentation scheme that extracts motion-centric phrases from original captions, exposing models to more motion expressions without additional annotations. Second, since the same object can be described differently depending on the context, we propose Multimodal Radial Contrastive Learning (MRaCL), performed on fused image-text embeddings rather than unimodal representations. For comprehensive evaluation, we introduce a new test split focusing on motion-centric queries, and introduce a new benchmark called M-Bench, where objects are distinguished primarily by actions. Extensive experiments show our method substantially improves performance on motion-centric queries across multiple RIS models, maintaining competitive results on appearance-based descriptions. Codes are available at https://github.com/snuviplab/MRaCL

  </details>



- **Motion-Adaptive Temporal Attention for Lightweight Video Generation with Stable Diffusion**  
  Rui Hong, Shuxue Quan  
  _2026-03-18_ · https://arxiv.org/abs/2603.17398v1  
  <details><summary>Abstract</summary>

  We present a motion-adaptive temporal attention mechanism for parameter-efficient video generation built upon frozen Stable Diffusion models. Rather than treating all video content uniformly, our method dynamically adjusts temporal attention receptive fields based on estimated motion content: high-motion sequences attend locally across frames to preserve rapidly changing details, while low-motion sequences attend globally to enforce scene consistency. We inject lightweight temporal attention modules into all UNet transformer blocks via a cascaded strategy -- global attention in down-sampling and middle blocks for semantic stabilization, motion-adaptive attention in up-sampling blocks for fine-grained refinement. Combined with temporally correlated noise initialization and motion-aware gating, the system adds only 25.8M trainable parameters (2.9\% of the base UNet) while achieving competitive results on WebVid validation when trained on 100K videos. We demonstrate that the standard denoising objective alone provides sufficient implicit temporal regularization, outperforming approaches that add explicit temporal consistency losses. Our ablation studies reveal a clear trade-off between noise correlation and motion amplitude, providing a practical inference-time control for diverse generation behaviors.

  </details>



- **Adaptive Anchor Policies for Efficient 4D Gaussian Streaming**  
  Ashim Dahal, Rabab Abdelfattah, Nick Rahimi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17227v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}

  </details>



- **Emotion-Aware Classroom Quality Assessment Leveraging IoT-Based Real-Time Student Monitoring**  
  Hai Nguyen, Hieu Dao, Hung Nguyen, Nam Vu, Cong Tran  
  _2026-03-17_ · https://arxiv.org/abs/2603.16719v1  
  <details><summary>Abstract</summary>

  This study presents high-throughput, real-time multi-agent affective computing framework designed to enhance classroom learning through emotional state monitoring. As large classroom sizes and limited teacher student interaction increasingly challenge educators, there is a growing need for scalable, data-driven tools capable of capturing students' emotional and engagement patterns in real time. The system was evaluated using the Classroom Emotion Dataset, consisting of 1,500 labeled images and 300 classroom detection videos. Tailored for IoT devices, the system addresses load balancing and latency challenges through efficient real-time processing. Field testing was conducted across three educational institutions in a large metropolitan area: a primary school (hereafter school A), a secondary school (school B), and a high school (school C). The system demonstrated robust performance, detecting up to 50 faces at 25 FPS and achieving 88% overall accuracy in classifying classroom engagement states. Implementation results showed positive outcomes, with favorable feedback from students, teachers, and parents regarding improved classroom interaction and teaching adaptation. Key contributions of this research include establishing a practical, IoT-based framework for emotion-aware learning environments and introducing the 'Classroom Emotion Dataset' to facilitate further validation and research.

  </details>



- **$x^2$-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space**  
  Ruishan Guo, Ciyu Ruan, Haoyang Wang, Zihang Gong, Jingao Xu, Xinlei Chen  
  _2026-03-17_ · https://arxiv.org/abs/2603.16671v1  
  <details><summary>Abstract</summary>

  Estimating dense 2D optical flow and 3D scene flow is essential for dynamic scene understanding. Recent work combines images, LiDAR, and event data to jointly predict 2D and 3D motion, yet most approaches operate in separate heterogeneous feature spaces. Without a shared latent space that all modalities can align to, these systems rely on multiple modality-specific blocks, leaving cross-sensor mismatches unresolved and making fusion unnecessarily complex.Event cameras naturally provide a spatiotemporal edge signal, which we can treat as an intrinsic edge field to anchor a unified latent representation, termed the Event Edge Space. Building on this idea, we introduce $x^2$-Fusion, which reframes multimodal fusion as representation unification: event-derived spatiotemporal edges define an edge-centric homogeneous space, and image and LiDAR features are explicitly aligned in this shared representation.Within this space, we perform reliability-aware adaptive fusion to estimate modality reliability and emphasize stable cues under degradation. We further employ cross-dimension contrast learning to tightly couple 2D optical flow with 3D scene flow. Extensive experiments on both synthetic and real benchmarks show that $x^2$-Fusion achieves state-of-the-art accuracy under standard conditions and delivers substantial improvements in challenging scenarios.

  </details>



- **SpikeCLR: Contrastive Self-Supervised Learning for Few-Shot Event-Based Vision using Spiking Neural Networks**  
  Maxime Vaillant, Axel Carlier, Lai Xing Ng, Christophe Hurter, Benoit R. Cottereau  
  _2026-03-17_ · https://arxiv.org/abs/2603.16338v1  
  <details><summary>Abstract</summary>

  Event-based vision sensors provide significant advantages for high-speed perception, including microsecond temporal resolution, high dynamic range, and low power consumption. When combined with Spiking Neural Networks (SNNs), they can be deployed on neuromorphic hardware, enabling energy-efficient applications on embedded systems. However, this potential is severely limited by the scarcity of large-scale labeled datasets required to effectively train such models. In this work, we introduce SpikeCLR, a contrastive self-supervised learning framework that enables SNNs to learn robust visual representations from unlabeled event data. We adapt prior frame-based methods to the spiking domain using surrogate gradient training and introduce a suite of event-specific augmentations that leverage spatial, temporal, and polarity transformations. Through extensive experiments on CIFAR10-DVS, N-Caltech101, N-MNIST, and DVS-Gesture benchmarks, we demonstrate that self-supervised pretraining with subsequent fine-tuning outperforms supervised learning in low-data regimes, achieving consistent gains in few-shot and semi-supervised settings. Our ablation studies reveal that combining spatial and temporal augmentations is critical for learning effective spatio-temporal invariances in event data. We further show that learned representations transfer across datasets, contributing to efforts for powerful event-based models in label-scarce settings.

  </details>



- **DriveFix: Spatio-Temporally Coherent Driving Scene Restoration**  
  Heyu Si, Brandon James Denis, Muyang Sun, Dragos Datcu, Yaoru Li, Xin Jin, Ruiju Fu, Yuliia Tatarinova, Federico Landi, Jie Song, et al.  
  _2026-03-17_ · https://arxiv.org/abs/2603.16306v1  
  <details><summary>Abstract</summary>

  Recent advancements in 4D scene reconstruction, particularly those leveraging diffusion priors, have shown promise for novel view synthesis in autonomous driving. However, these methods often process frames independently or in a view-by-view manner, leading to a critical lack of spatio-temporal synergy. This results in spatial misalignment across cameras and temporal drift in sequences. We propose DriveFix, a novel multi-view restoration framework that ensures spatio-temporal coherence for driving scenes. Our approach employs an interleaved diffusion transformer architecture with specialized blocks to explicitly model both temporal dependencies and cross-camera spatial consistency. By conditioning the generation on historical context and integrating geometry-aware training losses, DriveFix enforces that the restored views adhere to a unified 3D geometry. This enables the consistent propagation of high-fidelity textures and significantly reduces artifacts. Extensive evaluations on the Waymo, nuScenes, and PandaSet datasets demonstrate that DriveFix achieves state-of-the-art performance in both reconstruction and novel view synthesis, marking a substantial step toward robust 4D world modeling for real-world deployment.

  </details>



- **STARK: Spatio-Temporal Attention for Representation of Keypoints for Continuous Sign Language Recognition**  
  Suvajit Patra, Soumitra Samanta  
  _2026-03-17_ · https://arxiv.org/abs/2603.16163v1  
  <details><summary>Abstract</summary>

  Continuous Sign Language Recognition (CSLR) is a crucial task for understanding the languages of deaf communities. Contemporary keypoint-based approaches typically rely on spatio-temporal encoding, where spatial interactions among keypoints are modeled using Graph Convolutional Networks or attention mechanisms, while temporal dynamics are captured using 1D convolutional networks. However, such designs often introduce a large number of parameters in both the encoder and the decoder. This paper introduces a unified spatio-temporal attention network that computes attention scores both spatially (across keypoints) and temporally (within local windows), and aggregates features to produce a local context-aware spatio-temporal representation. The proposed encoder contains approximately $70-80\%$ fewer parameters than existing state-of-the-art models while achieving comparable performance to keypoint-based methods on the Phoenix-14T dataset.

  </details>



- **GATS: Gaussian Aware Temporal Scaling Transformer for Invariant 4D Spatio-Temporal Point Cloud Representation**  
  Jiayi Tian, Jiaze Wang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16154v1  
  <details><summary>Abstract</summary>

  Understanding 4D point cloud videos is essential for enabling intelligent agents to perceive dynamic environments. However, temporal scale bias across varying frame rates and distributional uncertainty in irregular point clouds make it highly challenging to design a unified and robust 4D backbone. Existing CNN or Transformer based methods are constrained either by limited receptive fields or by quadratic computational complexity, while neglecting these implicit distortions. To address this problem, we propose a novel dual invariant framework, termed \textbf{Gaussian Aware Temporal Scaling (GATS)}, which explicitly resolves both distributional inconsistencies and temporal. The proposed \emph{Uncertainty Guided Gaussian Convolution (UGGC)} incorporates local Gaussian statistics and uncertainty aware gating into point convolution, thereby achieving robust neighborhood aggregation under density variation, noise, and occlusion. In parallel, the \emph{Temporal Scaling Attention (TSA)} introduces a learnable scaling factor to normalize temporal distances, ensuring frame partition invariance and consistent velocity estimation across different frame rates. These two modules are complementary: temporal scaling normalizes time intervals prior to Gaussian estimation, while Gaussian modeling enhances robustness to irregular distributions. Our experiments on mainstream benchmarks MSR-Action3D (\textbf{+6.62\%} accuracy), NTU RGBD (\textbf{+1.4\%} accuracy), and Synthia4D (\textbf{+1.8\%} mIoU) demonstrate significant performance gains, offering a more efficient and principled paradigm for invariant 4D point cloud video understanding with superior accuracy, robustness, and scalability compared to Transformer based counterparts.

  </details>



- **Volumetrically Consistent Implicit Atlas Learning via Neural Diffeomorphic Flow for Placenta MRI**  
  Athena Taymourtash, S. Mazdak Abulnaga, Esra Abaci Turk, P. Ellen Grant, Polina Golland  
  _2026-03-17_ · https://arxiv.org/abs/2603.16078v1  
  <details><summary>Abstract</summary>

  Establishing dense volumetric correspondences across anatomical shapes is essential for group-level analysis but remains challenging for implicit neural representations. Most existing implicit registration methods rely on supervision near the zero-level set and thus capture only surface correspondences, leaving interior deformations under-constrained. We introduce a volumetrically consistent implicit model that couples reconstruction of signed distance functions (SDFs) with neural diffeomorphic flow to learn a shared canonical template of the placenta. Volumetric regularization, including Jacobian-determinant and biharmonic penalties, suppresses local folding and promotes globally coherent deformations. In the motivating application to placenta MRI, our formulation jointly reconstructs individual placentas, aligns them to a population-derived implicit template, and enables voxel-wise intensity mapping in a unified canonical space. Experiments on in-vivo placenta MRI scans demonstrate improved geometric fidelity and volumetric alignment over surface-based implicit baseline methods, yielding anatomically interpretable and topologically consistent flattening suitable for group analysis.

  </details>


