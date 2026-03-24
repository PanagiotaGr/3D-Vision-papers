# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-24 07:17 UTC_

Total papers shown: **11**


---

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



- **DSCSNet: A Dynamic Sparse Compression Sensing Network for Closely-Spaced Infrared Small Target Unmixing**  
  Zhiyang Tang, Yiming Zhu, Ruimin Huang, Meng Yang, Yong Ma, Jun Huang, Fan Fan  
  _2026-03-22_ · https://arxiv.org/abs/2603.21192v1  
  <details><summary>Abstract</summary>

  Due to the limitations of optical lens focal length and detector resolution, distant clustered infrared small targets often appear as mixed spots. The Close Small Object Unmixing (CSOU) task aims to recover the number, sub-pixel positions, and radiant intensities of individual targets from these spots, which is a highly ill-posed inverse problem. Existing methods struggle to balance the rigorous sparsity guarantees of model-driven approaches and the dynamic scene adaptability of data-driven methods. To address this dilemma, this paper proposes a Dynamic Sparse Compressed Sensing Network (DSCSNet), a deep-unfolded network that couples the Alternating Direction Method of Multipliers (ADMM) with learnable parameters. Specifically, we embed a strict $\ell_1$-norm sparsity constraint into the auxiliary variable update step of ADMM to replace the traditional $\ell_2$-norm smoothness-promoting terms, which effectively preserves the discrete energy peaks of small targets. We also integrate a self-attention-based dynamic thresholding mechanism into the reconstruction stage, which adaptively adjusts the sparsification intensity using the sparsity-enhanced information from the iterative process. These modules are jointly optimized end-to-end across the three iterative steps of ADMM. Retaining the physical logic of compressed sensing, DSCSNet achieves robust sparsity induction and scene adaptability, thus enhancing the unmixing accuracy and generalization in complex infrared scenarios. Extensive experiments on the synthetic infrared dataset CSIST-100K demonstrate that DSCSNet outperforms state-of-the-art methods in key metrics such as CSO-mAP and sub-pixel localization error.

  </details>



- **LiFR-Seg: Anytime High-Frame-Rate Segmentation via Event-Guided Propagation**  
  Xiaoshan Wu, Xiaoyang Lyu, Yifei Yu, Bo Wang, Zhongrui Wang, Xiaojuan Qi  
  _2026-03-22_ · https://arxiv.org/abs/2603.21115v1  
  <details><summary>Abstract</summary>

  Dense semantic segmentation in dynamic environments is fundamentally limited by the low-frame-rate (LFR) nature of standard cameras, which creates critical perceptual gaps between frames. To solve this, we introduce Anytime Interframe Semantic Segmentation: a new task for predicting segmentation at any arbitrary time using only a single past RGB frame and a stream of asynchronous event data. This task presents a core challenge: how to robustly propagate dense semantic features using a motion field derived from sparse and often noisy event data, all while mitigating feature degradation in highly dynamic scenes. We propose LiFR-Seg, a novel framework that directly addresses these challenges by propagating deep semantic features through time. The core of our method is an uncertainty-aware warping process, guided by an event-driven motion field and its learned, explicit confidence. A temporal memory attention module further ensures coherence in dynamic scenarios. We validate our method on the DSEC dataset and a new high-frequency synthetic benchmark (SHF-DSEC) we contribute. Remarkably, our LFR system achieves performance (73.82% mIoU on DSEC) that is statistically indistinguishable from an HFR upper-bound (within 0.09%) that has full access to the target frame. This work presents a new, efficient paradigm for achieving robust, high-frame-rate perception with low-frame-rate hardware. Project Page: https://candy-crusher.github.io/LiFR_Seg_Proj/#; Code: https://github.com/Candy-Crusher/LiFR-Seg.git.

  </details>



- **Fast and Robust Deformable 3D Gaussian Splatting**  
  Han Jiao, Jiakai Sun, Lei Zhao, Zhanjie Zhang, Wei Xing, Huaizhong Lin  
  _2026-03-21_ · https://arxiv.org/abs/2603.20857v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has demonstrated remarkable real-time rendering capabilities and superior visual quality in novel view synthesis for static scenes. Building upon these advantages, researchers have progressively extended 3D Gaussians to dynamic scene reconstruction. Deformation field-based methods have emerged as a promising approach among various techniques. These methods maintain 3D Gaussian attributes in a canonical field and employ the deformation field to transform this field across temporal sequences. Nevertheless, these approaches frequently encounter challenges such as suboptimal rendering speeds, significant dependence on initial point clouds, and vulnerability to local optima in dim scenes. To overcome these limitations, we present FRoG, an efficient and robust framework for high-quality dynamic scene reconstruction. FRoG integrates per-Gaussian embedding with a coarse-to-fine temporal embedding strategy, accelerating rendering through the early fusion of temporal embeddings. Moreover, to enhance robustness against sparse initializations, we introduce a novel depth- and error-guided sampling strategy. This strategy populates the canonical field with new 3D Gaussians at low-deviation initial positions, significantly reducing the optimization burden on the deformation field and improving detail reconstruction in both static and dynamic regions. Furthermore, by modulating opacity variations, we mitigate the local optima problem in dim scenes, improving color fidelity. Comprehensive experimental results validate that our method achieves accelerated rendering speeds while maintaining state-of-the-art visual quality.

  </details>


