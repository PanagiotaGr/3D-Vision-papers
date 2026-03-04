# Neural Rendering & View Synthesis

_Updated: 2026-03-04 07:04 UTC_

Total papers shown: **10**


---

- **VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats**  
  Alessio Mazzucchelli, Ivan Ojeda-Martin, Fernando Rivas-Manzaneque, Elena Garces, Adrian Penate-Sanchez, Francesc Moreno-Noguer  
  _2026-03-03_ · https://arxiv.org/abs/2603.02986v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.

  </details>



- **Multimodal-Prior-Guided Importance Sampling for Hierarchical Gaussian Splatting in Sparse-View Novel View Synthesis**  
  Kaiqiang Xiong, Zhanke Wang, Ronggang Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02866v1  
  <details><summary>Abstract</summary>

  We present multimodal-prior-guided importance sampling as the central mechanism for hierarchical 3D Gaussian Splatting (3DGS) in sparse-view novel view synthesis. Our sampler fuses complementary cues { -- } photometric rendering residuals, semantic priors, and geometric priors { -- } to produce a robust, local recoverability estimate that directly drives where to inject fine Gaussians. Built around this sampling core, our framework comprises (1) a coarse-to-fine Gaussian representation that encodes global shape with a stable coarse layer and selectively adds fine primitives where the multimodal metric indicates recoverable detail; and (2) a geometric-aware sampling and retention policy that concentrates refinement on geometrically critical and complex regions while protecting newly added primitives in underconstrained areas from premature pruning. By prioritizing regions supported by consistent multimodal evidence rather than raw residuals alone, our method alleviates overfitting texture-induced errors and suppresses noise from pose/appearance inconsistencies. Experiments on diverse sparse-view benchmarks demonstrate state-of-the-art reconstructions, with up to +0.3 dB PSNR on DTU.

  </details>



- **R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild**  
  Margherita Lea Corona, Wieland Morgenstern, Peter Eisert, Anna Hilsmann  
  _2026-03-03_ · https://arxiv.org/abs/2603.02801v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS representation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW models view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction artifacts, improving rendering quality at the sky-foreground boundary

  </details>



- **SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding**  
  Sheng Ye, Zhen-Hui Dong, Ruoyu Fan, Tian Lv, Yong-Jin Liu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02548v1  
  <details><summary>Abstract</summary>

  Semantic understanding of 3D scenes is essential for robots to operate effectively and safely in complex environments. Existing methods for semantic scene reconstruction and semantic-aware novel view synthesis often rely on dense multi-view inputs and require scene-specific optimization, limiting their practicality and scalability in real-world applications. To address these challenges, we propose SemGS, a feed-forward framework for reconstructing generalizable semantic fields from sparse image inputs. SemGS uses a dual-branch architecture to extract color and semantic features, where the two branches share shallow CNN layers, allowing semantic reasoning to leverage textural and structural cues in color appearance. We also incorporate a camera-aware attention mechanism into the feature extractor to explicitly model geometric relationships between camera viewpoints. The extracted features are decoded into dual-Gaussians that share geometric consistency while preserving branch-specific attributes, and further rasterized to synthesize semantic maps under novel viewpoints. Additionally, we introduce a regional smoothness loss to enhance semantic coherence. Experiments show that SemGS achieves state-of-the-art performance on benchmark datasets, while providing rapid inference and strong generalization capabilities across diverse synthetic and real-world scenarios.

  </details>



- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution**  
  Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02134v2  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

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



- **You Only Need One Stage: Novel-View Synthesis From A Single Blind Face Image**  
  Taoyue Wang, Xiang Zhang, Xiaotian Li, Huiyuan Yang, Lijun Yin  
  _2026-03-01_ · https://arxiv.org/abs/2603.01328v1  
  <details><summary>Abstract</summary>

  We propose a novel one-stage method, NVB-Face, for generating consistent Novel-View images directly from a single Blind Face image. Existing approaches to novel-view synthesis for objects or faces typically require a high-resolution RGB image as input. When dealing with degraded images, the conventional pipeline follows a two-stage process: first restoring the image to high resolution, then synthesizing novel views from the restored result. However, this approach is highly dependent on the quality of the restored image, often leading to inaccuracies and inconsistencies in the final output. To address this limitation, we extract single-view features directly from the blind face image and introduce a feature manipulator that transforms these features into 3D-aware, multi-view latent representations. Leveraging the powerful generative capacity of a diffusion model, our framework synthesizes high-quality, consistent novel-view face images. Experimental results show that our method significantly outperforms traditional two-stage approaches in both consistency and fidelity.

  </details>



- **RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations**  
  Mochu Xiang, Zhelun Shen, Xuesong Li, Jiahui Ren, Jing Zhang, Chen Zhao, Shanshan Liu, Haocheng Feng, Jingdong Wang, Yuchao Dai  
  _2026-03-01_ · https://arxiv.org/abs/2603.01194v1  
  <details><summary>Abstract</summary>

  Human perceive the 3D world through 2D observations from limited viewpoints. While recent feed-forward generalizable 3D reconstruction models excel at recovering 3D structures from sparse images, their representations are often confined to observed regions, leaving unseen geometry un-modeled. This raises a key, fundamental challenge: Can we infer a complete 3D structure from partial 2D observations? We present RnG (Reconstruction and Generation), a novel feed-forward Transformer that unifies these two tasks by predicting an implicit, complete 3D representation. At the core of RnG, we propose a reconstruction-guided causal attention mechanism that separates reconstruction and generation at the attention level, and treats the KV-cache as an implicit 3D representation. Then, arbitrary poses can efficiently query this cache to render high-fidelity, novel-view RGBD outputs. As a result, RnG not only accurately reconstructs visible geometry but also generates plausible, coherent unseen geometry and appearance. Our method achieves state-of-the-art performance in both generalizable 3D reconstruction and novel view generation, while operating efficiently enough for real-time interactive applications. Project page: https://npucvr.github.io/RnG

  </details>



- **HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views**  
  Jiashu Li, Xumeng Han, Zhaoyang Wei, Zipeng Wang, Kuiran Wang, Guorong Li, Zhenjun Han, Jianbin Jiao  
  _2026-03-01_ · https://arxiv.org/abs/2603.01099v2  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a promising approach in novel view synthesis, combining photorealistic rendering with real-time efficiency. However, its success heavily relies on dense camera coverage; under sparse-view conditions, insufficient supervision leads to irregular Gaussian distributions, characterized by globally sparse coverage, blurred background, and distorted high-frequency areas. To address this, we propose HeroGS, Hierarchical Guidance for Robust 3D Gaussian Splatting, a unified framework that establishes hierarchical guidance across the image, feature, and parameter levels. At the image level, sparse supervision is converted into pseudo-dense guidance, globally regularizing the Gaussian distributions and forming a consistent foundation for subsequent optimization. Building upon this, Feature-Adaptive Densification and Pruning (FADP) at the feature level leverages low-level features to refine high-frequency details and adaptively densifies Gaussians in background regions. The optimized distributions then support Co-Pruned Geometry Consistency (CPG) at parameter level, which guides geometric consistency through parameter freezing and co-pruning, effectively removing inconsistent splats. The hierarchical guidance strategy effectively constrains and optimizes the overall Gaussian distributions, thereby enhancing both structural fidelity and rendering quality. Extensive experiments demonstrate that HeroGS achieves high-fidelity reconstructions and consistently surpasses state-of-the-art baselines under sparse-view conditions.

  </details>


