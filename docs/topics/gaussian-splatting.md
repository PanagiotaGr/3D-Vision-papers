# Gaussian Splatting & 3DGS

_Updated: 2026-04-12 07:42 UTC_

Total papers shown: **5**


---

- **Visually-grounded Humanoid Agents**  
  Hang Ye, Xiaoxuan Ma, Fan Lu, Wayne Wu, Kwan-Yee Lin, Yizhou Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08509v1  
  <details><summary>Abstract</summary>

  Digital human generation has been studied for decades and supports a wide range of real-world applications. However, most existing systems are passively animated, relying on privileged state or scripted control, which limits scalability to novel environments. We instead ask: how can digital humans actively behave using only visual observations and specified goals in novel scenes? Achieving this would enable populating any 3D environments with digital humans at scale that exhibit spontaneous, natural, goal-directed behaviors. To this end, we introduce Visually-grounded Humanoid Agents, a coupled two-layer (world-agent) paradigm that replicates humans at multiple levels: they look, perceive, reason, and behave like real people in real-world 3D scenes. The World Layer reconstructs semantically rich 3D Gaussian scenes from real-world videos via an occlusion-aware pipeline and accommodates animatable Gaussian-based human avatars. The Agent Layer transforms these avatars into autonomous humanoid agents, equipping them with first-person RGB-D perception and enabling them to perform accurate, embodied planning with spatial awareness and iterative reasoning, which is then executed at the low level as full-body actions to drive their behaviors in the scene. We further introduce a benchmark to evaluate humanoid-scene interaction in diverse reconstructed environments. Experiments show our agents achieve robust autonomous behavior, yielding higher task success rates and fewer collisions than ablations and state-of-the-art planning methods. This work enables active digital human population and advances human-centric embodied AI. Data, code, and models will be open-sourced.

  </details>



- **BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields**  
  Fan Yang, Wenrui Chen, Guorun Yan, Ruize Liao, Wanjun Jia, Dongsheng Luo, Kailun Yang, Zhiyong Li, Yaonan Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.08410v1  
  <details><summary>Abstract</summary>

  In unstructured environments, functional dexterous grasping calls for the tight integration of semantic understanding, precise 3D functional localization, and physically interpretable execution. Modular hierarchical methods are more controllable and interpretable than end-to-end VLA approaches, but existing ones still rely on predefined affordance labels and lack the tight semantic--pose coupling needed for functional dexterous manipulation. To address this, we propose BLaDA (Bridging Language to Dexterous Actions in 3DGS fields), an interpretable zero-shot framework that grounds open-vocabulary instructions as perceptual and control constraints for functional dexterous manipulation. BLaDA establishes an interpretable reasoning chain by first parsing natural language into a structured sextuple of manipulation constraints via a Knowledge-guided Language Parsing (KLP) module. To achieve pose-consistent spatial reasoning, we introduce the Triangular Functional Point Localization (TriLocation) module, which utilizes 3D Gaussian Splatting as a continuous scene representation and identifies functional regions under triangular geometric constraints. Finally, the 3D Keypoint Grasp Matrix Transformation Execution (KGT3D+) module decodes these semantic-geometric constraints into physically plausible wrist poses and finger-level commands. Extensive experiments on complex benchmarks demonstrate that BLaDA significantly outperforms existing methods in both affordance grounding precision and the success rate of functional manipulation across diverse categories and tasks. Code will be publicly available at https://github.com/PopeyePxx/BLaDA.

  </details>



- **SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction**  
  Chensheng Dai, Shengjun Zhang, Min Chen, Yueqi Duan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08370v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated impressive performance in 3D scene reconstruction. Beyond novel view synthesis, it shows great potential for multi-view surface reconstruction. Existing methods employ optimization-based reconstruction pipelines that achieve precise and complete surface extractions. However, these approaches typically require dense input views and high time consumption for per-scene optimization. To address these limitations, we propose SurfelSplat, a feed-forward framework that generates efficient and generalizable pixel-aligned Gaussian surfel representations from sparse-view images. We observe that conventional feed-forward structures struggle to recover accurate geometric attributes of Gaussian surfels because the spatial frequency of pixel-aligned primitives exceeds Nyquist sampling rates. Therefore, we propose a cross-view feature aggregation module based on the Nyquist sampling theorem. Specifically, we first adapt the geometric forms of Gaussian surfels with spatial sampling rate-guided low-pass filters. We then project the filtered surfels across all input views to obtain cross-view feature correlations. By processing these correlations through a specially designed feature fusion network, we can finally regress Gaussian surfels with precise geometry. Extensive experiments on DTU reconstruction benchmarks demonstrate that our model achieves comparable results with state-of-the-art methods, and predict Gaussian surfels within 1 second, offering a 100x speedup without costly per-scene training.

  </details>



- **DP-DeGauss: Dynamic Probabilistic Gaussian Decomposition for Egocentric 4D Scene Reconstruction**  
  Tingxi Chen, Zhengxue Cheng, Houqiang Zhong, Su Wang, Rong Xie, Li Song  
  _2026-04-09_ · https://arxiv.org/abs/2604.07986v1  
  <details><summary>Abstract</summary>

  Egocentric video is crucial for next-generation 4D scene reconstruction, with applications in AR/VR and embodied AI. However, reconstructing dynamic first-person scenes is challenging due to complex ego-motion, occlusions, and hand-object interactions. Existing decomposition methods are ill-suited, assuming fixed viewpoints or merging dynamics into a single foreground. To address these limitations, we introduce DP-DeGauss, a dynamic probabilistic Gaussian decomposition framework for egocentric 4D reconstruction. Our method initializes a unified 3D Gaussian set from COLMAP priors, augments each with a learnable category probability, and dynamically routes them into specialized deformation branches for background, hands, or object modeling. We employ category-specific masks for better disentanglement and introduce brightness and motion-flow control to improve static rendering and dynamic reconstruction. Extensive experiments show that DP-DeGauss outperforms baselines by +1.70dB in PSNR on average with SSIM and LPIPS gains. More importantly, our framework achieves the first and state-of-the-art disentanglement of background, hand, and object components, enabling explicit, fine-grained separation, paving the way for more intuitive ego scene understanding and editing.

  </details>



- **Generative 3D Gaussian Splatting for Arbitrary-ResolutionAtmospheric Downscaling and Forecasting**  
  Tao Hana, Zhibin Wen, Zhenghao Chen, Fenghua Lin, Junyu Gao, Song Guo, Lei Bai  
  _2026-04-09_ · https://arxiv.org/abs/2604.07928v1  
  <details><summary>Abstract</summary>

  While AI-based numerical weather prediction (NWP) enables rapid forecasting, generating high-resolution outputs remains computationally demanding due to limited multi-scale adaptability and inefficient data representations. We propose the 3D Gaussian splatting-based scale-aware vision transformer (GSSA-ViT), a novel framework for arbitrary-resolution forecasting and flexible downscaling of high-dimensional atmospheric fields. Specifically, latitude-longitude grid points are treated as centers of 3D Gaussians. A generative 3D Gaussian prediction scheme is introduced to estimate key parameters, including covariance, attributes, and opacity, for unseen samples, improving generalization and mitigating overfitting. In addition, a scale-aware attention module is designed to capture cross-scale dependencies, enabling the model to effectively integrate information across varying downscaling ratios and support continuous resolution adaptation. To our knowledge, this is the first NWP approach that combines generative 3D Gaussian modeling with scale-aware attention for unified multi-scale prediction. Experiments on ERA5 show that the proposed method accurately forecasts 87 atmospheric variables at arbitrary resolutions, while evaluations on ERA5 and CMIP6 demonstrate its superior performance in downscaling tasks. The proposed framework provides an efficient and scalable solution for high-resolution, multi-scale atmospheric prediction and downscaling. Code is available at: https://github.com/binbin2xs/weather-GS.

  </details>


