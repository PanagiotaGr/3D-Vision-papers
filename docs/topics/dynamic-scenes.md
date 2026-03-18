# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-18 07:16 UTC_

Total papers shown: **12**


---

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


