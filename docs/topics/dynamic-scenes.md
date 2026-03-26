# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-26 07:34 UTC_

Total papers shown: **13**


---

- **Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timestep**  
  Tianyi Liu, Ye Lu, Linfeng Zhang, Chen Cai, Jianjun Gao, Yi Wang, Kim-Hui Yap, Lap-Pui Chau  
  _2026-03-25_ · https://arxiv.org/abs/2603.24260v1  
  <details><summary>Abstract</summary>

  Diffusion-based video editing has emerged as an important paradigm for high-quality and flexible content generation. However, despite their generality and strong modeling capacity, Diffusion Transformers (DiT) remain computationally expensive due to the iterative denoising process, posing challenges for practical deployment. Existing video diffusion acceleration methods primarily exploit denoising timestep-level feature reuse, which mitigates the redundancy in denoising process, but overlooks the architectural redundancy within the DiT that many attention operations over spatio-temporal tokens are redundantly executed, offering little to no incremental contribution to the model output. This work introduces HetCache, a training-free diffusion acceleration framework designed to exploit the inherent heterogeneity in diffusion-based masked video-to-video (MV2V) generation and editing. Instead of uniformly reuse or randomly sampling tokens, HetCache assesses the contextual relevance and interaction strength among various types of tokens in designated computing steps. Guided by spatial priors, it divides the spatial-temporal tokens in DiT model into context and generative tokens, and selectively caches the context tokens that exhibit the strongest correlation and most representative semantics with generative ones. This strategy reduces redundant attention operations while maintaining editing consistency and fidelity. Experiments show that HetCache achieves a noticeable acceleration, including a 2.67$\times$ latency speedup and FLOPs reduction over commonly used foundation models, with negligible degradation in editing quality.

  </details>



- **Spectral Scalpel: Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering for Skeleton-Based Action Segmentation**  
  Haoyu Ji, Bowen Chen, Zhihao Yang, Wenze Huang, Yu Gao, Xueting Liu, Weihong Ren, Zhiyong Wang, Honghai Liu  
  _2026-03-25_ · https://arxiv.org/abs/2603.24134v1  
  <details><summary>Abstract</summary>

  Skeleton-based Temporal Action Segmentation (STAS) seeks to densely segment and classify diverse actions within long, untrimmed skeletal motion sequences. However, existing STAS methodologies face challenges of limited inter-class discriminability and blurred segmentation boundaries, primarily due to insufficient distinction of spatio-temporal patterns between adjacent actions. To address these limitations, we propose Spectral Scalpel, a frequency-selective filtering framework aimed at suppressing shared frequency components between adjacent distinct actions while amplifying their action-specific frequencies, thereby enhancing inter-action discrepancies and sharpening transition boundaries. Specifically, Spectral Scalpel employs adaptive multi-scale spectral filters as scalpels to edit frequency spectra, coupled with a discrepancy loss between adjacent actions serving as the surgical objective. This design amplifies representational disparities between neighboring actions, effectively mitigating boundary localization ambiguities and inter-class confusion. Furthermore, complementing long-term temporal modeling, we introduce a frequency-aware channel mixer to strengthen channel evolution by aggregating spectra across channels. This work presents a novel paradigm for STAS that extends conventional spatio-temporal modeling by incorporating frequency-domain analysis. Extensive experiments on five public datasets demonstrate that Spectral Scalpel achieves state-of-the-art performance. Code is available at https://github.com/HaoyuJi/SpecScalpel.

  </details>



- **LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation via Spatial-Temporal Modulation**  
  Haoyu Ji, Xueting Liu, Yu Gao, Wenze Huang, Zhihao Yang, Weihong Ren, Zhiyong Wang, Honghai Liu  
  _2026-03-25_ · https://arxiv.org/abs/2603.24097v1  
  <details><summary>Abstract</summary>

  Skeleton-based Temporal Action Segmentation (STAS) aims to densely parse untrimmed skeletal sequences into frame-level action categories. However, existing methods, while proficient at capturing spatio-temporal kinematics, neglect the underlying physical dynamics that govern human motion. This oversight limits inter-class discriminability between actions with similar kinematics but distinct dynamic intents, and hinders precise boundary localization where dynamic force profiles shift. To address these, we propose the Lagrangian-Dynamic Informed Network (LaDy), a framework integrating principles of Lagrangian dynamics into the segmentation process. Specifically, LaDy first computes generalized coordinates from joint positions and then estimates Lagrangian terms under physical constraints to explicitly synthesize the generalized forces. To further ensure physical coherence, our Energy Consistency Loss enforces the work-energy theorem, aligning kinetic energy change with the work done by the net force. The learned dynamics then drive a Spatio-Temporal Modulation module: Spatially, generalized forces are fused with spatial representations to provide more discriminative semantics. Temporally, salient dynamic signals are constructed for temporal gating, thereby significantly enhancing boundary awareness. Experiments on challenging datasets show that LaDy achieves state-of-the-art performance, validating the integration of physical dynamics for action segmentation. Code is available at https://github.com/HaoyuJi/LaDy.

  </details>



- **Mind the Hitch: Dynamic Calibration and Articulated Perception for Autonomous Trucks**  
  Morui Zhu, Yongqi Zhu, Song Fu, Qing Yang  
  _2026-03-24_ · https://arxiv.org/abs/2603.23711v1  
  <details><summary>Abstract</summary>

  Autonomous trucking poses unique challenges due to articulated tractor-trailer geometry, and time-varying sensor poses caused by the fifth-wheel joint and trailer flex. Existing perception and calibration methods assume static baselines or rely on high-parallax and texture-rich scenes, limiting their reliability under real-world settings. We propose dCAP (dynamic Calibration and Articulated Perception), a vision-based framework that continuously estimates the 6-DoF (degree of freedom) relative pose between tractor and trailer cameras. dCAP employs a transformer with cross-view and temporal attention to robustly aggregate spatial cues while maintaining temporal consistency, enabling accurate perception under rapid articulation and occlusion. Integrated with BEVFormer, dCAP improves 3D object detection by replacing static calibration with dynamically predicted extrinsics. To facilitate evaluation, we introduce STT4AT, a CARLA-based benchmark simulating semi-trailer trucks with synchronized multi-sensor suites and time-varying inter-rig geometry across diverse environments. Experiments demonstrate that dCAP achieves stable, accurate perception while addressing the limitations of static calibration in autonomous trucking. The dataset, development kit, and source code will be publicly released.

  </details>



- **DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models**  
  Jaewon Min, Jaeeun Lee, Yeji Choi, Paul Hyunbin Cho, Jin Hyeon Kim, Tae-Young Lee, Jongsik Ahn, Hwayeong Lee, Seonghyun Park, Seungryong Kim  
  _2026-03-24_ · https://arxiv.org/abs/2603.23499v1  
  <details><summary>Abstract</summary>

  Optical flow models trained on high-quality data often degrade severely when confronted with real-world corruptions such as blur, noise, and compression artifacts. To overcome this limitation, we formulate Degradation-Aware Optical Flow, a new task targeting accurate dense correspondence estimation from real-world corrupted videos. Our key insight is that the intermediate representations of image restoration diffusion models are inherently corruption-aware but lack temporal awareness. To address this limitation, we lift the model to attend across adjacent frames via full spatio-temporal attention, and empirically demonstrate that the resulting features exhibit zero-shot correspondence capabilities. Based on this finding, we present DA-Flow, a hybrid architecture that fuses these diffusion features with convolutional features within an iterative refinement framework. DA-Flow substantially outperforms existing optical flow methods under severe degradation across multiple benchmarks.

  </details>



- **AgentRVOS: Reasoning over Object Tracks for Zero-Shot Referring Video Object Segmentation**  
  Woojeong Jin, Jaeho Lee, Heeseong Shin, Seungho Jang, Junhwan Heo, Seungryong Kim  
  _2026-03-24_ · https://arxiv.org/abs/2603.23489v1  
  <details><summary>Abstract</summary>

  Referring Video Object Segmentation (RVOS) aims to segment a target object throughout a video given a natural language query. Training-free methods for this task follow a common pipeline: a MLLM selects keyframes, grounds the referred object within those frames, and a video segmentation model propagates the results. While intuitive, this design asks the MLLM to make temporal decisions before any object-level evidence is available, limiting both reasoning quality and spatio-temporal coverage. To overcome this, we propose AgentRVOS, a training-free agentic pipeline built on the complementary strengths of SAM3 and a MLLM. Given a concept derived from the query, SAM3 provides reliable perception over the full spatio-temporal extent through generated mask tracks. The MLLM then identifies the target through query-grounded reasoning over this object-level evidence, iteratively pruning guided by SAM3's temporal existence information. Extensive experiments show that AgentRVOS achieves state-of-the-art performance among training-free methods across multiple benchmarks, with consistent results across diverse MLLM backbones. Our project page is available at: https://cvlab-kaist.github.io/AgentRVOS/.

  </details>



- **TETO: Tracking Events with Teacher Observation for Motion Estimation and Frame Interpolation**  
  Jini Yang, Eunbeen Hong, Soowon Son, Hyunkoo Lee, Sunghwan Hong, Sunok Kim, Seungryong Kim  
  _2026-03-24_ · https://arxiv.org/abs/2603.23487v1  
  <details><summary>Abstract</summary>

  Event cameras capture per-pixel brightness changes with microsecond resolution, offering continuous motion information lost between RGB frames. However, existing event-based motion estimators depend on large-scale synthetic data that often suffers from a significant sim-to-real gap. We propose TETO (Tracking Events with Teacher Observation), a teacher-student framework that learns event motion estimation from only $\sim$25 minutes of unannotated real-world recordings through knowledge distillation from a pretrained RGB tracker. Our motion-aware data curation and query sampling strategy maximizes learning from limited data by disentangling object motion from dominant ego-motion. The resulting estimator jointly predicts point trajectories and dense optical flow, which we leverage as explicit motion priors to condition a pretrained video diffusion transformer for frame interpolation. We achieve state-of-the-art point tracking on EVIMO2 and optical flow on DSEC using orders of magnitude less training data, and demonstrate that accurate motion estimation translates directly to superior frame interpolation quality on BS-ERGB and HQ-EVFI.

  </details>



- **Object Pose Transformer: Unifying Unseen Object Pose Estimation**  
  Weihang Li, Lorenzo Garattoni, Fabien Despinoy, Nassir Navab, Benjamin Busam  
  _2026-03-24_ · https://arxiv.org/abs/2603.23370v1  
  <details><summary>Abstract</summary>

  Learning model-free object pose estimation for unseen instances remains a fundamental challenge in 3D vision. Existing methods typically fall into two disjoint paradigms: category-level approaches predict absolute poses in a canonical space but rely on predefined taxonomies, while relative pose methods estimate cross-view transformations but cannot recover single-view absolute pose. In this work, we propose Object Pose Transformer (\ours{}), a unified feed-forward framework that bridges these paradigms through task factorization within a single model. \ours{} jointly predicts depth, point maps, camera parameters, and normalized object coordinates (NOCS) from RGB inputs, enabling both category-level absolute SA(3) pose and unseen-object relative SE(3) pose. Our approach leverages contrastive object-centric latent embeddings for canonicalization without requiring semantic labels at inference time, and uses point maps as a camera-space representation to enable multi-view relative geometric reasoning. Through cross-frame feature interaction and shared object embeddings, our model leverages relative geometric consistency across views to improve absolute pose estimation, reducing ambiguity in single-view predictions. Furthermore, \ours{} is camera-agnostic, learning camera intrinsics on-the-fly and supporting optional depth input for metric-scale recovery, while remaining fully functional in RGB-only settings. Extensive experiments on diverse benchmarks (NOCS, HouseCat6D, Omni6DPose, Toyota-Light) demonstrate state-of-the-art performance in both absolute and relative pose estimation tasks within a single unified architecture.

  </details>



- **3rd Place of MeViS-Audio Track of the 5th PVUW: VIRST-Audio**  
  Jihwan Hong, Jaeyoung Do  
  _2026-03-24_ · https://arxiv.org/abs/2603.23126v1  
  <details><summary>Abstract</summary>

  Audio-based Referring Video Object Segmentation (ARVOS) requires grounding audio queries into pixel-level object masks over time, posing challenges in bridging acoustic signals with spatio-temporal visual representations. In this report, we present VIRST-Audio, a practical framework built upon a pretrained RVOS model integrated with a vision-language architecture. Instead of relying on audio-specific training, we convert input audio into text using an ASR module and perform segmentation using text-based supervision, enabling effective transfer from text-based reasoning to audio-driven scenarios. To improve robustness, we further incorporate an existence-aware gating mechanism that estimates whether the referred target object is present in the video and suppresses predictions when it is absent, reducing hallucinated masks and stabilizing segmentation behavior. We evaluate our approach on the MeViS-Audio track of the 5th PVUW Challenge, where VIRST-Audio achieves 3rd place, demonstrating strong generalization and reliable performance in audio-based referring video segmentation.

  </details>



- **Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining**  
  Weijun Zhuang, Yuqing Huang, Weikang Meng, Xin Li, Ming Liu, Xiaopeng Hong, Yaowei Wang, Wangmeng Zuo  
  _2026-03-24_ · https://arxiv.org/abs/2603.22953v1  
  <details><summary>Abstract</summary>

  Large-scale video-language pretraining enables strong generalization across multimodal tasks but often incurs prohibitive computational costs. Although recent advances in masked visual modeling help mitigate this issue, they still suffer from two fundamental limitations: severe visual information loss under high masking ratios and temporal information leakage caused by inter-frame correlations. To address these challenges, we propose ClusterSTM, a Cluster-Wise Spatio-Temporal Masking strategy for efficient video-language pretraining. ClusterSTM first performs intra-frame clustering to partition visual tokens into multiple semantically independent clusters, then conducts cluster-wise masking by retaining the token with the highest temporal density within each cluster. Our masking strategy ensure that the retained tokens capture holistic video content while exhibit strong temporal correlation. Additionally, we introduce a video-text relevance reconstruction objective that aligns high-level multimodal semantics beyond conventional visual reconstruction. Extensive experiments across multiple benchmarks demonstrate that ClusterSTM achieves superior performance on video-text retrieval, video question answering, and video captioning tasks, establishing a new state-of-the-art among efficient video-language models.

  </details>



- **SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes**  
  Zhicheng Qiu, Jiarui Meng, Tong-an Luo, Yican Huang, Xuan Feng, Xuanfu Li, ZHan Xu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22893v2  
  <details><summary>Abstract</summary>

  We propose SLARM, a feed-forward model that unifies dynamic scene reconstruction, semantic understanding, and real-time streaming inference. SLARM captures complex, non-uniform motion through higher-order motion modeling, trained solely on differentiable renderings without any flow supervision. Besides, SLARM distills semantic features from LSeg to obtain language-aligned representations. This design enables semantic querying via natural language, and the tight coupling between semantics and geometry further enhances the accuracy and robustness of dynamic reconstruction. Moreover, SLARM processes image sequences using window-based causal attention, achieving stable, low-latency streaming inference without accumulating memory cost. Within this unified framework, SLARM achieves state-of-the-art results in dynamic estimation, rendering quality, and scene parsing, improving motion accuracy by 21%, reconstruction PSNR by 1.6 dB, and segmentation mIoU by 20% over existing methods.

  </details>



- **WorldCache: Content-Aware Caching for Accelerated Video World Models**  
  Umair Nawaz, Ahmed Heakl, Ufaq Khan, Abdelrahman Shaker, Salman Khan, Fahad Shahbaz Khan  
  _2026-03-23_ · https://arxiv.org/abs/2603.22286v1  
  <details><summary>Abstract</summary>

  Diffusion Transformers (DiTs) power high-fidelity video world models but remain computationally expensive due to sequential denoising and costly spatio-temporal attention. Training-free feature caching accelerates inference by reusing intermediate activations across denoising steps; however, existing methods largely rely on a Zero-Order Hold assumption i.e., reusing cached features as static snapshots when global drift is small. This often leads to ghosting artifacts, blur, and motion inconsistencies in dynamic scenes. We propose \textbf{WorldCache}, a Perception-Constrained Dynamical Caching framework that improves both when and how to reuse features. WorldCache introduces motion-adaptive thresholds, saliency-weighted drift estimation, optimal approximation via blending and warping, and phase-aware threshold scheduling across diffusion steps. Our cohesive approach enables adaptive, motion-consistent feature reuse without retraining. On Cosmos-Predict2.5-2B evaluated on PAI-Bench, WorldCache achieves \textbf{2.3$\times$} inference speedup while preserving \textbf{99.4\%} of baseline quality, substantially outperforming prior training-free caching approaches. Our code can be accessed on \href{https://umair1221.github.io/World-Cache/}{World-Cache}.

  </details>



- **UniMotion: A Unified Framework for Motion-Text-Vision Understanding and Generation**  
  Ziyi Wang, Xinshun Wang, Shuang Chen, Yang Cong, Mengyuan Liu  
  _2026-03-23_ · https://arxiv.org/abs/2603.22282v1  
  <details><summary>Abstract</summary>

  We present UniMotion, to our knowledge the first unified framework for simultaneous understanding and generation of human motion, natural language, and RGB images within a single architecture. Existing unified models handle only restricted modality subsets (e.g., Motion-Text or static Pose-Image) and predominantly rely on discrete tokenization, which introduces quantization errors and disrupts temporal continuity. UniMotion overcomes both limitations through a core principle: treating motion as a first-class continuous modality on equal footing with RGB. A novel Cross-Modal Aligned Motion VAE (CMA-VAE) and symmetric dual-path embedders construct parallel continuous pathways for Motion and RGB within a shared LLM backbone. To inject visual-semantic priors into motion representations without requiring images at inference, we propose Dual-Posterior KL Alignment (DPA), which distills a vision-fused encoder's richer posterior into the motion-only encoder. To address the cold-start problem -- where text supervision alone is too sparse to calibrate the newly introduced motion pathway -- we further propose Latent Reconstruction Alignment (LRA), a self-supervised pre-training strategy that uses dense motion latents as unambiguous conditions to co-calibrate the embedder, backbone, and flow head, establishing a stable motion-aware foundation for all downstream tasks. UniMotion achieves state-of-the-art performance across seven tasks spanning any-to-any understanding, generation, and editing among the three modalities, with especially strong advantages on cross-modal compositional tasks.

  </details>


