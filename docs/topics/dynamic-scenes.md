# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-25 07:16 UTC_

Total papers shown: **15**


---

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
  _2026-03-24_ · https://arxiv.org/abs/2603.22893v1  
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



- **Biophysics-Enhanced Neural Representations for Patient-Specific Respiratory Motion Modeling**  
  Jan Boysen, Hristina Uzunova, Heinz Handels, Jan Ehrhardt  
  _2026-03-23_ · https://arxiv.org/abs/2603.22123v1  
  <details><summary>Abstract</summary>

  A precise spatial delivery of the radiation dose is crucial for the treatment success in radiotherapy. In the lung and upper abdominal region, respiratory motion introduces significant treatment uncertainties, requiring special motion management techniques. To address this, respiratory motion models are commonly used to infer the patient-specific respiratory motion and target the dose more efficiently. In this work, we investigate the possibility of using implicit neural representations (INR) for surrogate-based motion modeling. Therefore, we propose physics-regularized implicit surrogate-based modeling for respiratory motion (PRISM-RM). Our new integrated respiratory motion model is free of a fixed reference breathing state. Unlike conventional pairwise registration techniques, our approach provides a trajectory-aware spatio-temporally continuous and diffeomorphic motion representation, improving generalization to extrapolation scenarios. We introduce biophysical constraints, ensuring physiologically plausible motion estimation across time beyond the training data. Our results show that our trajectory-aware approach performs on par in interpolation and improves the extrapolation ability compared to our initially proposed INR-based approach. Compared to sequential registration-based approaches both our approaches perform equally well in interpolation, but underperform in extrapolation scenarios. However, the methodical features of INRs make them particularly effective for respiratory motion modeling, and with their performance steadily improving, they demonstrate strong potential for advancing this field.

  </details>



- **Cycle Inverse-Consistent TransMorph: A Balanced Deep Learning Framework for Brain MRI Registration**  
  Jiaqi Shang, Haojin Wu, Yinyi Lai, Zongyu Li, Chenghao Zhang, Jia Guo  
  _2026-03-23_ · https://arxiv.org/abs/2603.21760v1  
  <details><summary>Abstract</summary>

  Deformable image registration plays a fundamental role in medical image analysis by enabling spatial alignment of anatomical structures across subjects. While recent deep learning-based approaches have significantly improved computational efficiency, many existing methods remain limited in capturing long-range anatomical correspondence and maintaining deformation consistency. In this work, we present a cycle inverse-consistent transformer-based framework for deformable brain MRI registration. The model integrates a Swin-UNet architecture with bidirectional consistency constraints, enabling the joint estimation of forward and backward deformation fields. This design allows the framework to capture both local anatomical details and global spatial relationships while improving deformation stability. We conduct a comprehensive evaluation of the proposed framework on a large multi-center dataset consisting of 2851 T1-weighted brain MRI scans aggregated from 13 public datasets. Experimental results demonstrate that the proposed framework achieves strong and balanced performance across multiple quantitative evaluation metrics while maintaining stable and physically plausible deformation fields. Detailed quantitative comparisons with baseline methods, including ANTs, ICNet, and VoxelMorph, are provided in the appendix. Experimental results demonstrate that CICTM achieves consistently strong performance across multiple evaluation criteria while maintaining stable and physically plausible deformation fields. These properties make the proposed framework suitable for large-scale neuroimaging datasets where both accuracy and deformation stability are critical.

  </details>



- **HumanOmni-Speaker: Identifying Who said What and When**  
  Detao Bai, Shimin Yao, Weixuan Chen, Xihan Wei, Zhiheng Ma  
  _2026-03-23_ · https://arxiv.org/abs/2603.21664v1  
  <details><summary>Abstract</summary>

  While Omni-modal Large Language Models have made strides in joint sensory processing, they fundamentally struggle with a cornerstone of human interaction: deciphering complex, multi-person conversational dynamics to accurately answer ``Who said what and when.'' Current models suffer from an ``illusion of competence'' -- they exploit visual biases in conventional benchmarks to bypass genuine cross-modal alignment, while relying on sparse, low-frame-rate visual sampling that destroys crucial high-frequency dynamics like lip movements. To shatter this illusion, we introduce Visual-Registered Speaker Diarization and Recognition (VR-SDR) and the HumanOmni-Speaker Benchmark. By strictly eliminating visual shortcuts, this rigorous paradigm demands true end-to-end spatio-temporal identity binding using only natural language queries. To overcome the underlying architectural perception gap, we propose HumanOmni-Speaker, powered by a Visual Delta Encoder. By sampling raw video at 25 fps and explicitly compressing inter-frame motion residuals into just 6 tokens per frame, it captures fine-grained visemes and speaker trajectories without triggering a catastrophic token explosion. Ultimately, HumanOmni-Speaker demonstrates strong multimodal synergy, natively enabling end-to-end lip-reading and high-precision spatial localization without intrusive cropping, and achieving superior performance across a wide spectrum of speaker-centric tasks.

  </details>



- **4DGS360: 360° Gaussian Reconstruction of Dynamic Objects from a Single Video**  
  Jae Won Jang, Yeonjin Chang, Wonsik Shin, Juhwan Cho, Nojun Kwak  
  _2026-03-23_ · https://arxiv.org/abs/2603.21618v1  
  <details><summary>Abstract</summary>

  We introduce 4DGS360, a diffusion-free framework for 360$^{\circ}$ dynamic object reconstruction from casual monocular video. Existing methods often fail to reconstruct consistent 360$^{\circ}$ geometry, as their heavy reliance on 2D-native priors causes initial points to overfit to visible surface in each training view. 4DGS360 addresses this challenge through a advanced 3D-native initialization that mitigates the geometric ambiguity of occluded regions. Our proposed 3D tracker, AnchorTAP3D, produces reinforced 3D point trajectories by leveraging confident 2D track points as anchors, suppressing drift and providing reliable initialization that preserves geometry in occluded regions. This initialization, combined with optimization, yields coherent 360$^{\circ}$ 4D reconstructions. We further present iPhone360, a new benchmark where test cameras are placed up to 135$^{\circ}$ apart from training views, enabling 360$^{\circ}$ evaluation that existing datasets cannot provide. Experiments show that 4DGS360 achieves state-of-the-art performance on the iPhone360, iPhone, and DAVIS datasets, both qualitatively and quantitatively.

  </details>



- **PAS3R: Pose-Adaptive Streaming 3D Reconstruction for Long Video Sequences**  
  Lanbo Xu, Liang Guo, Caigui Jiang, Cheng Wang  
  _2026-03-22_ · https://arxiv.org/abs/2603.21436v1  
  <details><summary>Abstract</summary>

  Online monocular 3D reconstruction enables dense scene recovery from streaming video but remains fundamentally limited by the stability-adaptation dilemma: the reconstruction model must rapidly incorporate novel viewpoints while preserving previously accumulated scene structure. Existing streaming approaches rely on uniform or attention-based update mechanisms that often fail to account for abrupt viewpoint transitions, leading to trajectory drift and geometric inconsistencies over long sequences. We introduce PAS3R, a pose-adaptive streaming reconstruction framework that dynamically modulates state updates according to camera motion and scene structure. Our key insight is that frames contributing significant geometric novelty should exert stronger influence on the reconstruction state, while frames with minor viewpoint variation should prioritize preserving historical context. PAS3R operationalizes this principle through a motion-aware update mechanism that jointly leverages inter-frame pose variation and image frequency cues to estimate frame importance. To further stabilize long-horizon reconstruction, we introduce trajectory-consistent training objectives that incorporate relative pose constraints and acceleration regularization. A lightweight online stabilization module further suppresses high-frequency trajectory jitter and geometric artifacts without increasing memory consumption. Extensive experiments across multiple benchmarks demonstrate that PAS3R significantly improves trajectory accuracy, depth estimation, and point cloud reconstruction quality in long video sequences while maintaining competitive performance on shorter sequences.

  </details>



- **EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization**  
  Haolan Xu, Keli Cheng, Lei Wang, Ning Bi, Xiaoming Liu  
  _2026-03-22_ · https://arxiv.org/abs/2603.21332v1  
  <details><summary>Abstract</summary>

  Audio-driven 3D talking head synthesis has advanced rapidly with Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS). By leveraging rich pre-trained priors, few-shot methods enable instant personalization from just a few seconds of video. However, under expressive facial motion, existing few-shot approaches often suffer from geometric instability and audio-emotion mismatch, highlighting the need for more effective emotion-aware motion modeling. In this work, we present EmoTaG, a few-shot emotion-aware 3D talking head synthesis framework built on the Pretrain-and-Adapt paradigm. Our key insight is to reformulate motion prediction in a structured FLAME parameter space rather than directly deforming 3D Gaussians, thereby introducing explicit geometric priors that improve motion stability. Building upon this, we propose a Gated Residual Motion Network (GRMN), which captures emotional prosody from audio while supplementing head pose and upper-face cues absent from audio, enabling expressive and coherent motion generation. Extensive experiments demonstrate that EmoTaG achieves state-of-the-art performance in emotional expressiveness, lip synchronization, visual realism, and motion stability.

  </details>


