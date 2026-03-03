# Gaussian Splatting & 3DGS

_Updated: 2026-03-03 07:08 UTC_

Total papers shown: **7**


---

- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution**  
  Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02134v1  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

  </details>



- **LiftAvatar: Kinematic-Space Completion for Expression-Controlled 3D Gaussian Avatar Animation**  
  Hualiang Wei, Shunran Jia, Jialun Liu, Wenhui Li  
  _2026-03-02_ · https://arxiv.org/abs/2603.02129v1  
  <details><summary>Abstract</summary>

  We present LiftAvatar, a new paradigm that completes sparse monocular observations in kinematic space (e.g., facial expressions and head pose) and uses the completed signals to drive high-fidelity avatar animation. LiftAvatar is a fine-grained, expression-controllable large-scale video diffusion Transformer that synthesizes high-quality, temporally coherent expression sequences conditioned on single or multiple reference images. The key idea is to lift incomplete input data into a richer kinematic representation, thereby strengthening both reconstruction and animation in downstream 3D avatar pipelines. To this end, we introduce (i) a multi-granularity expression control scheme that combines shading maps with expression coefficients for precise and stable driving, and (ii) a multi-reference conditioning mechanism that aggregates complementary cues from multiple frames, enabling strong 3D consistency and controllability. As a plug-and-play enhancer, LiftAvatar directly addresses the limited expressiveness and reconstruction artifacts of 3D Gaussian Splatting-based avatars caused by sparse kinematic cues in everyday monocular videos. By expanding incomplete observations into diverse pose-expression variations, LiftAvatar also enables effective prior distillation from large-scale video generative models into 3D pipelines, leading to substantial gains. Extensive experiments show that LiftAvatar consistently boosts animation quality and quantitative metrics of state-of-the-art 3D avatar methods, especially under extreme, unseen expressions.

  </details>



- **Sparse View Distractor-Free Gaussian Splatting**  
  Yi Gu, Zhaorui Wang, Jiahang Cao, Jiaxu Wang, Mingle Zhao, Dongjun Ye, Renjing Xu  
  _2026-03-02_ · https://arxiv.org/abs/2603.01603v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enables efficient training and fast novel view synthesis in static environments. To address challenges posed by transient objects, distractor-free 3DGS methods have emerged and shown promising results when dense image captures are available. However, their performance degrades significantly under sparse input conditions. This limitation primarily stems from the reliance on the color residual heuristics to guide the training, which becomes unreliable with limited observations. In this work, we propose a framework to enhance distractor-free 3DGS under sparse-view conditions by incorporating rich prior information. Specifically, we first adopt the geometry foundation model VGGT to estimate camera parameters and generate a dense set of initial 3D points. Then, we harness the attention maps from VGGT for efficient and accurate semantic entity matching. Additionally, we utilize Vision-Language Models (VLMs) to further identify and preserve the large static regions in the scene. We also demonstrate how these priors can be seamlessly integrated into existing distractor-free 3DGS methods. Extensive experiments confirm the effectiveness and robustness of our approach in mitigating transient distractors for sparse-view 3DGS training.

  </details>



- **Radiometrically Consistent Gaussian Surfels for Inverse Rendering**  
  Kyu Beom Han, Jaeyoon Kim, Woo Jae Kim, Jinhwan Seo, Sung-eui Yoon  
  _2026-03-02_ · https://arxiv.org/abs/2603.01491v1  
  <details><summary>Abstract</summary>

  Inverse rendering with Gaussian Splatting has advanced rapidly, but accurately disentangling material properties from complex global illumination effects, particularly indirect illumination, remains a major challenge. Existing methods often query indirect radiance from Gaussian primitives pre-trained for novel-view synthesis. However, these pre-trained Gaussian primitives are supervised only towards limited training viewpoints, thus lack supervision for modeling indirect radiances from unobserved views. To address this issue, we introduce radiometric consistency, a novel physically-based constraint that provides supervision towards unobserved views by minimizing the residual between each Gaussian primitive's learned radiance and its physically-based rendered counterpart. Minimizing the residual for unobserved views establishes a self-correcting feedback loop that provides supervision from both physically-based rendering and novel-view synthesis, enabling accurate modeling of inter-reflection. We then propose Radiometrically Consistent Gaussian Surfels (RadioGS), an inverse rendering framework built upon our principle by efficiently integrating radiometric consistency by utilizing Gaussian surfels and 2D Gaussian ray tracing. We further propose a finetuning-based relighting strategy that adapts Gaussian surfel radiances to new illuminations within minutes, achieving low rendering cost (<10ms). Extensive experiments on existing inverse rendering benchmarks show that RadioGS outperforms existing Gaussian-based methods in inverse rendering, while retaining the computational efficiency.

  </details>



- **HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views**  
  Jiashu Li, Xumeng Han, Zhaoyang Wei, Zipeng Wang, Kuiran Wang, Guorong Li, Zhenjun Han, Jianbin Jiao  
  _2026-03-01_ · https://arxiv.org/abs/2603.01099v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a promising approach in novel view synthesis, combining photorealistic rendering with real-time efficiency. However, its success heavily relies on dense camera coverage; under sparse-view conditions, insufficient supervision leads to irregular Gaussian distributions, characterized by globally sparse coverage, blurred background, and distorted high-frequency areas. To address this, we propose HeroGS, Hierarchical Guidance for Robust 3D Gaussian Splatting, a unified framework that establishes hierarchical guidance across the image, feature, and parameter levels. At the image level, sparse supervision is converted into pseudo-dense guidance, globally regularizing the Gaussian distributions and forming a consistent foundation for subsequent optimization. Building upon this, Feature-Adaptive Densification and Pruning (FADP) at the feature level leverages low-level features to refine high-frequency details and adaptively densifies Gaussians in background regions. The optimized distributions then support Co-Pruned Geometry Consistency (CPG) at parameter level, which guides geometric consistency through parameter freezing and co-pruning, effectively removing inconsistent splats. The hierarchical guidance strategy effectively constrains and optimizes the overall Gaussian distributions, thereby enhancing both structural fidelity and rendering quality. Extensive experiments demonstrate that HeroGS achieves high-fidelity reconstructions and consistently surpasses state-of-the-art baselines under sparse-view conditions.

  </details>



- **Decoupling Motion and Geometry in 4D Gaussian Splatting**  
  Yi Zhang, Yulei Kang, Jian-Fang Hu  
  _2026-03-01_ · https://arxiv.org/abs/2603.00952v1  
  <details><summary>Abstract</summary>

  High-fidelity reconstruction of dynamic scenes is an important yet challenging problem. While recent 4D Gaussian Splatting (4DGS) has demonstrated the ability to model temporal dynamics, it couples Gaussian motion and geometric attributes within a single covariance formulation, which limits its expressiveness for complex motions and often leads to visual artifacts. To address this, we propose VeGaS, a novel velocity-based 4D Gaussian Splatting framework that decouples Gaussian motion and geometry. Specifically, we introduce a Galilean shearing matrix that explicitly incorporates time-varying velocity to flexibly model complex non-linear motions, while strictly isolating the effects of Gaussian motion from the geometry-related conditional Gaussian covariance. Furthermore, a Geometric Deformation Network is introduced to refine Gaussian shapes and orientations using spatio-temporal context and velocity cues, enhancing temporal geometric modeling. Extensive experiments on public datasets demonstrate that VeGaS achieves state-of-the-art performance.

  </details>



- **TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward Pose-free Reconstruction**  
  Yihui Li, Chengxin Lv, Zichen Tang, Hongyu Yang, Di Huang  
  _2026-02-28_ · https://arxiv.org/abs/2603.00697v1  
  <details><summary>Abstract</summary>

  We present TokenSplat, a feed-forward framework for joint 3D Gaussian reconstruction and camera pose estimation from unposed multi-view images. At its core, TokenSplat introduces a Token-aligned Gaussian Prediction module that aligns semantically corresponding information across views directly in the feature space. Guided by coarse token positions and fusion confidence, it aggregates multi-scale contextual features to enable long-range cross-view reasoning and reduce redundancy from overlapping Gaussians. To further enhance pose robustness and disentangle viewpoint cues from scene semantics, TokenSplat employs learnable camera tokens and an Asymmetric Dual-Flow Decoder (ADF-Decoder) that enforces directionally constrained communication between camera and image tokens. This maintains clean factorization within a feed-forward architecture, enabling coherent reconstruction and stable pose estimation without iterative refinement. Extensive experiments demonstrate that TokenSplat achieves higher reconstruction fidelity and novel-view synthesis quality in pose-free settings, and significantly improves pose estimation accuracy compared to prior pose-free methods. Project page: https://kidleyh.github.io/tokensplat/.

  </details>


