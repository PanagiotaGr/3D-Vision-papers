# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-11 07:09 UTC_

Total papers shown: **9**


---

- **BrainSTR: Spatio-Temporal Contrastive Learning for Interpretable Dynamic Brain Network Modeling**  
  Guiliang Guo, Guangqi Wen, Lingwen Liu, Ruoxian Song, Peng Cao, Jinzhu Yang, Fei Wang, Xiaoli Liu, Osmar R. Zaiane  
  _2026-03-10_ · https://arxiv.org/abs/2603.09825v1  
  <details><summary>Abstract</summary>

  Dynamic functional connectivity captures time-varying brain states for better neuropsychiatric diagnosis and spatio-temporal interpretability, i.e., identifying when discriminative disease signatures emerge and where they reside in the connectivity topology. Reliable interpretability faces major challenges: diagnostic signals are often subtle and sparsely distributed across both time and topology, while nuisance fluctuations and non-diagnostic connectivities are pervasive. To address these issues, we propose BrainSTR, a spatio-temporal contrastive learning framework for interpretable dynamic brain network modeling. BrainSTR learns state-consistent phase boundaries via a data-driven Adaptive Phase Partition module, identifies diagnostically critical phases with attention, and extracts disease-related connectivity within each phase using an Incremental Graph Structure Generator regularized by binarization, temporal smoothness, and sparsity. Then, we introduce a spatio-temporal supervised contrastive learning approach that leverages diagnosis-relevant spatio-temporal patterns to refine the similarity metric between samples and capture more discriminative spatio-temporal features, thereby constructing a well-structured semantic space for coherent and interpretable representations. Experiments on ASD, BD, and MDD validate the effectiveness of BrainSTR, and the discovered critical phases and subnetworks provide interpretable evidence consistent with prior neuroimaging findings. Our code: https://anonymous.4open.science/r/BrainSTR1.

  </details>



- **FrameDiT: Diffusion Transformer with Frame-Level Matrix Attention for Efficient Video Generation**  
  Minh Khoa Le, Kien Do, Duc Thanh Nguyen, Truyen Tran  
  _2026-03-10_ · https://arxiv.org/abs/2603.09721v1  
  <details><summary>Abstract</summary>

  High-fidelity video generation remains challenging for diffusion models due to the difficulty of modeling complex spatio-temporal dynamics efficiently. Recent video diffusion methods typically represent a video as a sequence of spatio-temporal tokens which can be modeled using Diffusion Transformers (DiTs). However, this approach faces a trade-off between the strong but expensive Full 3D Attention and the efficient but temporally limited Local Factorized Attention. To resolve this trade-off, we propose Matrix Attention, a frame-level temporal attention mechanism that processes an entire frame as a matrix and generates query, key, and value matrices via matrix-native operations. By attending across frames rather than tokens, Matrix Attention effectively preserves global spatio-temporal structure and adapts to significant motion. We build FrameDiT-G, a DiT architecture based on MatrixAttention, and further introduce FrameDiT-H, which integrates Matrix Attention with Local Factorized Attention to capture both large and small motion. Extensive experiments show that FrameDiT-H achieves state-of-the-art results across multiple video generation benchmarks, offering improved temporal coherence and video quality while maintaining efficiency comparable to Local Factorized Attention.

  </details>



- **DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics**  
  Yuanhang Lei, Boming Zhao, Zesong Yang, Xingxuan Li, Tao Cheng, Haocheng Peng, Ru Zhang, Yang Yang, Siyuan Huang, Yujun Shen, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09668v1  
  <details><summary>Abstract</summary>

  Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.

  </details>



- **EventVGGT: Exploring Cross-Modal Distillation for Consistent Event-based Depth Estimation**  
  Yinrui Ren, Jinjing Zhu, Kanghao Chen, Zhuoxiao Li, Jing Ou, Zidong Cao, Tongyan Hua, Peilun Shi, Yingchun Fu, Wufan Zhao, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09385v1  
  <details><summary>Abstract</summary>

  Event cameras offer superior sensitivity to high-speed motion and extreme lighting, making event-based monocular depth estimation a promising approach for robust 3D perception in challenging conditions. However, progress is severely hindered by the scarcity of dense depth annotations. While recent annotation-free approaches mitigate this by distilling knowledge from Vision Foundation Models (VFMs), a critical limitation persists: they process event streams as independent frames. By neglecting the inherent temporal continuity of event data, these methods fail to leverage the rich temporal priors encoded in VFMs, ultimately yielding temporally inconsistent and less accurate depth predictions. To address this, we introduce EventVGGT, a novel framework that explicitly models the event stream as a coherent video sequence. To the best of our knowledge, we are the first to distill spatio-temporal and multi-view geometric priors from the Visual Geometry Grounded Transformer (VGGT) into the event domain. We achieve this via a comprehensive tri-level distillation strategy: (i) Cross-Modal Feature Mixture (CMFM) bridges the modality gap at the output level by fusing RGB and event features to generate auxiliary depth predictions; (ii) Spatio-Temporal Feature Distillation (STFD) distills VGGT's powerful spatio-temporal representations at the feature level; and (iii) Temporal Consistency Distillation (TCD) enforces cross-frame coherence at the temporal level by aligning inter-frame depth changes. Extensive experiments demonstrate that EventVGGT consistently outperforms existing methods -- reducing the absolute mean depth error at 30m by over 53\% on EventScape (from 2.30 to 1.06) -- while exhibiting robust zero-shot generalization on the unseen DENSE and MVSEC datasets.

  </details>



- **Retrieval-Augmented Gaussian Avatars: Improving Expression Generalization**  
  Matan Levy, Gavriel Habib, Issar Tzachor, Dvir Samuel, Rami Ben-Ari, Nir Darshan, Or Litany, Dani Lischinski  
  _2026-03-09_ · https://arxiv.org/abs/2603.08645v1  
  <details><summary>Abstract</summary>

  Template-free animatable head avatars can achieve high visual fidelity by learning expression-dependent facial deformation directly from a subject's capture, avoiding parametric face templates and hand-designed blendshape spaces. However, since learned deformation is supervised only by the expressions observed for a single identity, these models suffer from limited expression coverage and often struggle when driven by motions that deviate from the training distribution. We introduce RAF (Retrieval-Augmented Faces), a simple training-time augmentation designed for template-free head avatars that learn deformation from data. RAF constructs a large unlabeled expression bank and, during training, replaces a subset of the subject's expression features with nearest-neighbor expressions retrieved from this bank while still reconstructing the subject's original frames. This exposes the deformation field to a broader range of expression conditions, encouraging stronger identity-expression decoupling and improving robustness to expression distribution shift without requiring paired cross-identity data, additional annotations, or architectural changes. We further analyze how retrieval augmentation increases expression diversity and validate retrieval quality with a user study showing that retrieved neighbors are perceptually closer in expression and pose. Experiments on the NeRSemble benchmark demonstrate that RAF consistently improves expression fidelity over the baseline, in both self-driving and cross-driving scenarios.

  </details>



- **HDR-NSFF: High Dynamic Range Neural Scene Flow Fields**  
  Shin Dong-Yeon, Kim Jun-Seong, Kwon Byung-Ki, Tae-Hyun Oh  
  _2026-03-09_ · https://arxiv.org/abs/2603.08313v1  
  <details><summary>Abstract</summary>

  Radiance of real-world scenes typically spans a much wider dynamic range than what standard cameras can capture. While conventional HDR methods merge alternating-exposure frames, these approaches are inherently constrained to 2D pixel-level alignment, often leading to ghosting artifacts and temporal inconsistency in dynamic scenes. To address these limitations, we present HDR-NSFF, a paradigm shift from 2D-based merging to 4D spatio-temporal modeling. Our framework reconstructs dynamic HDR radiance fields from alternating-exposure monocular videos by representing the scene as a continuous function of space and time, and is compatible with both neural radiance field and 4D Gaussian Splatting (4DGS) based dynamic representations. This unified end-to-end pipeline explicitly models HDR radiance, 3D scene flow, geometry, and tone-mapping, ensuring physical plausibility and global coherence. We further enhance robustness by (i) extending semantic-based optical flow with DINO features to achieve exposure-invariant motion estimation, and (ii) incorporating a generative prior as a regularizer to compensate for limited observation in monocular captures and saturation-induced information loss. To evaluate HDR space-time view synthesis, we present the first real-world HDR-GoPro dataset specifically designed for dynamic HDR scenes. Experiments demonstrate that HDR-NSFF recovers fine radiance details and coherent dynamics even under challenging exposure variations, thereby achieving state-of-the-art performance in novel space-time view synthesis. Project page: https://shin-dong-yeon.github.io/HDR-NSFF/

  </details>



- **DynamicVGGT: Learning Dynamic Point Maps for 4D Scene Reconstruction in Autonomous Driving**  
  Zhuolin He, Jing Li, Guanghao Li, Xiaolei Chen, Jiacheng Tang, Siyang Zhang, Zhounan Jin, Feipeng Cai, Bin Li, Jian Pu, et al.  
  _2026-03-09_ · https://arxiv.org/abs/2603.08254v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction in autonomous driving remains a fundamental challenge due to significant temporal variations, moving objects, and complex scene dynamics. Existing feed-forward 3D models have demonstrated strong performance in static reconstruction but still struggle to capture dynamic motion. To address these limitations, we propose DynamicVGGT, a unified feed-forward framework that extends VGGT from static 3D perception to dynamic 4D reconstruction. Our goal is to model point motion within feed-forward 3D models in a dynamic and temporally coherent manner. To this end, we jointly predict the current and future point maps within a shared reference coordinate system, allowing the model to implicitly learn dynamic point representations through temporal correspondence. To efficiently capture temporal dependencies, we introduce a Motion-aware Temporal Attention (MTA) module that learns motion continuity. Furthermore, we design a Dynamic 3D Gaussian Splatting Head that explicitly models point motion by predicting Gaussian velocities using learnable motion tokens under scene flow supervision. It refines dynamic geometry through continuous 3D Gaussian optimization. Extensive experiments on autonomous driving datasets demonstrate that DynamicVGGT significantly outperforms existing methods in reconstruction accuracy, achieving robust feed-forward 4D dynamic scene reconstruction under complex driving scenarios.

  </details>



- **AutoTraces: Autoregressive Trajectory Forecasting via Multimodal Large Language Models**  
  Teng Wang, Yanting Lu, Ruize Wang  
  _2026-03-09_ · https://arxiv.org/abs/2603.07989v1  
  <details><summary>Abstract</summary>

  We present AutoTraces, an autoregressive vision-language-trajectory model for robot trajectory forecasting in humam-populated environments, which harnesses the inherent reasoning capabilities of large language models (LLMs) to model complex human behaviors. In contrast to prior works that rely solely on textual representations, our key innovation lies in a novel trajectory tokenization scheme, which represents waypoints with point tokens as categorical and positional markers while encoding waypoint numerical values as corresponding point embeddings, seamlessly integrated into the LLM's space through a lightweight encoder-decoder architecture. This design preserves the LLM's native autoregressive generation mechanism while extending it to physical coordinate spaces, facilitates modeling of long-term interactions in trajectory data. We further introduce an automated chain-of-thought (CoT) generation mechanism that leverages a multimodal LLM to infer spatio-temporal relationships from visual observations and trajectory data, eliminating reliance on manual annotation. Through a two-stage training strategy, our AutoTraces achieves SOTA forecasting accuracy, particularly in long-horizon prediction, while exhibiting strong cross-scene generalization and supporting flexible-length forecasting.

  </details>



- **Extend Your Horizon: A Device-Agnostic Surgical Tool Tracking Framework with Multi-View Optimization for Augmented Reality**  
  Jiaming Zhang, Mingxu Liu, Hongchao Shu, Ruixing Liang, Yihao Liu, Ojas Taskar, Amir Kheradmand, Mehran Armand, Alejandro Martin-Gomez  
  _2026-03-09_ · https://arxiv.org/abs/2603.07981v1  
  <details><summary>Abstract</summary>

  Surgical navigation provides real-time guidance by estimating the pose of patient anatomy and surgical instruments to visualize relevant intraoperative information. In conventional systems, instruments are typically tracked using fiducial markers and stationary optical tracking systems (OTS). Augmented reality (AR) has further enabled intuitive visualization and motivated tracking using sensors embedded in head-mounted displays (HMDs). However, most existing approaches rely on a clear line of sight, which is difficult to maintain in dynamic operating room environments due to frequent occlusions caused by equipment, surgical tools, and personnel. This work introduces a framework for tracking surgical instruments under occlusion by fusing multiple sensing modalities within a dynamic scene graph representation. The proposed approach integrates tracking systems with different accuracy levels and motion characteristics while estimating tracking reliability in real time. Experimental results demonstrate improved robustness and enhanced consistency of AR visualization in the presence of occlusions.

  </details>


