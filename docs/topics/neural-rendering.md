# Neural Rendering & View Synthesis

_Updated: 2026-03-26 07:34 UTC_

Total papers shown: **12**


---

- **SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision**  
  Avigail Cohen Rimon, Amir Mann, Mirela Ben Chen, Or Litany  
  _2026-03-25_ · https://arxiv.org/abs/2603.24036v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enables real-time, photorealistic novel view synthesis, making it a highly attractive representation for model-based video tracking. However, leveraging the differentiability of the 3DGS renderer "in the wild" remains notoriously fragile. A fundamental bottleneck lies in the compact, local support of the Gaussian primitives. Standard photometric objectives implicitly rely on spatial overlap; if severe camera misalignment places the rendered object outside the target's local footprint, gradients strictly vanish, leaving the optimizer stranded. We introduce SpectralSplats, a robust tracking framework that resolves this "vanishing gradient" problem by shifting the optimization objective from the spatial to the frequency domain. By supervising the rendered image via a set of global complex sinusoidal features (Spectral Moments), we construct a global basin of attraction, ensuring that a valid, directional gradient toward the target exists across the entire image domain, even when pixel overlap is completely nonexistent. To harness this global basin without introducing periodic local minima associated with high frequencies, we derive a principled Frequency Annealing schedule from first principles, gracefully transitioning the optimizer from global convexity to precise spatial alignment. We demonstrate that SpectralSplats acts as a seamless, drop-in replacement for spatial losses across diverse deformation parameterizations (from MLPs to sparse control points), successfully recovering complex deformations even from severely misaligned initializations where standard appearance-based tracking catastrophically fails.

  </details>



- **FilterGS: Traversal-Free Parallel Filtering and Adaptive Shrinking for Large-Scale LoD 3D Gaussian Splatting**  
  Yixian Wang, Haolin Yu, Jiadong Tang, Yu Gao, Xihan Wang, Yufeng Yue, Yi Yang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23891v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has revolutionized neural rendering with real-time performance. However, scaling this approach to large scenes using Level-of-Detail methods faces critical challenges: inefficient serial traversal consuming over 60\% of rendering time, and redundant Gaussian-tile pairs that incur unnecessary processing overhead. To address these limitations, we introduce FilterGS, featuring a parallel filtering mechanism with two complementary filters that select Gaussian elements efficiently without tree traversal. Additionally, we propose a novel GTC metric that quantifies the redundancy of Gaussian-tile key-value pairs. Based on this metric, we introduce a scene-adaptive Gaussian shrinking strategy that effectively reduces redundant pairs. Extensive experiments demonstrate that FilterGS achieves state-of-the-art rendering speeds while maintaining competitive visual quality across multiple large-scale datasets. Project page: https://github.com/xenon-w/FilterGS

  </details>



- **OccAny: Generalized Unconstrained Urban 3D Occupancy**  
  Anh-Quan Cao, Tuan-Hung Vu  
  _2026-03-24_ · https://arxiv.org/abs/2603.23502v1  
  <details><summary>Abstract</summary>

  Relying on in-domain annotations and precise sensor-rig priors, existing 3D occupancy prediction methods are limited in both scalability and out-of-domain generalization. While recent visual geometry foundation models exhibit strong generalization capabilities, they were mainly designed for general purposes and lack one or more key ingredients required for urban occupancy prediction, namely metric prediction, geometry completion in cluttered scenes and adaptation to urban scenarios. We address this gap and present OccAny, the first unconstrained urban 3D occupancy model capable of operating on out-of-domain uncalibrated scenes to predict and complete metric occupancy coupled with segmentation features. OccAny is versatile and can predict occupancy from sequential, monocular, or surround-view images. Our contributions are three-fold: (i) we propose the first generalized 3D occupancy framework with (ii) Segmentation Forcing that improves occupancy quality while enabling mask-level prediction, and (iii) a Novel View Rendering pipeline that infers novel-view geometry to enable test-time view augmentation for geometry completion. Extensive experiments demonstrate that OccAny outperforms all visual geometry baselines on 3D occupancy prediction task, while remaining competitive with in-domain self-supervised methods across three input settings on two established urban occupancy prediction datasets. Our code is available at https://github.com/valeoai/OccAny .

  </details>



- **One View Is Enough! Monocular Training for In-the-Wild Novel View Generation**  
  Adrien Ramanana Rahary, Nicolas Dufour, Patrick Perez, David Picard  
  _2026-03-24_ · https://arxiv.org/abs/2603.23488v1  
  <details><summary>Abstract</summary>

  Monocular novel-view synthesis has long required multi-view image pairs for supervision, limiting training data scale and diversity. We argue it is not necessary: one view is enough. We present OVIE, trained entirely on unpaired internet images. We leverage a monocular depth estimator as a geometric scaffold at training time: we lift a source image into 3D, apply a sampled camera transformation, and project to obtain a pseudo-target view. To handle disocclusions, we introduce a masked training formulation that restricts geometric, perceptual, and textural losses to valid regions, enabling training on 30 million uncurated images. At inference, OVIE is geometry-free, requiring no depth estimator or 3D representation. Trained exclusively on in-the-wild images, OVIE outperforms prior methods in a zero-shot setting, while being 600x faster than the second-best baseline. Code and models are publicly available at https://github.com/AdrienRR/ovie.

  </details>



- **I3DM: Implicit 3D-aware Memory Retrieval and Injection for Consistent Video Scene Generation**  
  Jia Li, Han Yan, Yihang Chen, Siqi Li, Xibin Song, Yifu Wang, Jianfei Cai, Tien-Tsin Wong, Pan Ji  
  _2026-03-24_ · https://arxiv.org/abs/2603.23413v1  
  <details><summary>Abstract</summary>

  Despite remarkable progress in video generation, maintaining long-term scene consistency upon revisiting previously explored areas remains challenging. Existing solutions rely either on explicitly constructing 3D geometry, which suffers from error accumulation and scale ambiguity, or on naive camera Field-of-View (FoV) retrieval, which typically fails under complex occlusions. To overcome these limitations, we propose I3DM, a novel implicit 3D-aware memory mechanism for consistent video scene generation that bypasses explicit 3D reconstruction. At the core of our approach is a 3D-aware memory retrieval strategy, which leverages the intermediate features of a pre-trained Feed-Forward Novel View Synthesis (FF-NVS) model to score view relevance, enabling robust retrieval even in highly occluded scenarios. Furthermore, to fully utilize the retrieved historical frames, we introduce a 3D-aligned memory injection module. This module implicitly warps historical content to the target view and adaptively conditions the generation on reliable warping regions, leading to improved revisit consistency and accurate camera control. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches, achieving superior revisit consistency, generation fidelity, and camera control precision.

  </details>



- **Pose-Free Omnidirectional Gaussian Splatting for 360-Degree Videos with Consistent Depth Priors**  
  Chuanqing Zhuang, Xin Lu, Zehui Deng, Zhengda Lu, Yiqun Wang, Junqi Diao, Jun Xiao  
  _2026-03-24_ · https://arxiv.org/abs/2603.23324v1  
  <details><summary>Abstract</summary>

  Omnidirectional 3D Gaussian Splatting with panoramas is a key technique for 3D scene representation, and existing methods typically rely on slow SfM to provide camera poses and sparse points priors. In this work, we propose a pose-free omnidirectional 3DGS method, named PFGS360, that reconstructs 3D Gaussians from unposed omnidirectional videos. To achieve accurate camera pose estimation, we first construct a spherical consistency-aware pose estimation module, which recovers poses by establishing consistent 2D-3D correspondences between the reconstructed Gaussians and the unposed images using Gaussians' internal depth priors. Besides, to enhance the fidelity of novel view synthesis, we introduce a depth-inlier-aware densification module to extract depth inliers and Gaussian outliers with consistent monocular depth priors, enabling efficient Gaussian densification and achieving photorealistic novel view synthesis. The experiments show significant outperformance over existing pose-free and pose-aware 3DGS methods on both real-world and synthetic 360-degree videos. Code is available at https://github.com/zcq15/PFGS360.

  </details>



- **SLARM: Streaming and Language-Aligned Reconstruction Model for Dynamic Scenes**  
  Zhicheng Qiu, Jiarui Meng, Tong-an Luo, Yican Huang, Xuan Feng, Xuanfu Li, ZHan Xu  
  _2026-03-24_ · https://arxiv.org/abs/2603.22893v2  
  <details><summary>Abstract</summary>

  We propose SLARM, a feed-forward model that unifies dynamic scene reconstruction, semantic understanding, and real-time streaming inference. SLARM captures complex, non-uniform motion through higher-order motion modeling, trained solely on differentiable renderings without any flow supervision. Besides, SLARM distills semantic features from LSeg to obtain language-aligned representations. This design enables semantic querying via natural language, and the tight coupling between semantics and geometry further enhances the accuracy and robustness of dynamic reconstruction. Moreover, SLARM processes image sequences using window-based causal attention, achieving stable, low-latency streaming inference without accumulating memory cost. Within this unified framework, SLARM achieves state-of-the-art results in dynamic estimation, rendering quality, and scene parsing, improving motion accuracy by 21%, reconstruction PSNR by 1.6 dB, and segmentation mIoU by 20% over existing methods.

  </details>



- **UniQueR: Unified Query-based Feedforward 3D Reconstruction**  
  Chensheng Peng, Quentin Herau, Jiezhi Yang, Yichen Xie, Yihan Hu, Wenzhao Zheng, Matthew Strong, Masayoshi Tomizuka, Wei Zhan  
  _2026-03-24_ · https://arxiv.org/abs/2603.22851v1  
  <details><summary>Abstract</summary>

  We present UniQueR, a unified query-based feedforward framework for efficient and accurate 3D reconstruction from unposed images. Existing feedforward models such as DUSt3R, VGGT, and AnySplat typically predict per-pixel point maps or pixel-aligned Gaussians, which remain fundamentally 2.5D and limited to visible surfaces. In contrast, UniQueR formulates reconstruction as a sparse 3D query inference problem. Our model learns a compact set of 3D anchor points that act as explicit geometric queries, enabling the network to infer scene structure, including geometry in occluded regions--in a single forward pass. Each query encodes spatial and appearance priors directly in global 3D space (instead of per-frame camera space) and spawns a set of 3D Gaussians for differentiable rendering. By leveraging unified query interactions across multi-view features and a decoupled cross-attention design, UniQueR achieves strong geometric expressiveness while substantially reducing memory and computational cost. Experiments on Mip-NeRF 360 and VR-NeRF demonstrate that UniQueR surpasses state-of-the-art feedforward methods in both rendering quality and geometric accuracy, using an order of magnitude fewer primitives than dense alternatives.

  </details>



- **Predictive Photometric Uncertainty in Gaussian Splatting for Novel View Synthesis**  
  Chamuditha Jayanga Galappaththige, Thomas Gottwald, Peter Stehr, Edgar Heinert, Niko Suenderhauf, Dimity Miller, Matthias Rottmann  
  _2026-03-24_ · https://arxiv.org/abs/2603.22786v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D Gaussian Splatting have enabled impressive photorealistic novel view synthesis. However, to transition from a pure rendering engine to a reliable spatial map for autonomous agents and safety-critical applications, knowing where the representation is uncertain is as important as the rendering fidelity itself. We bridge this critical gap by introducing a lightweight, plug-and-play framework for pixel-wise, view-dependent predictive uncertainty estimation. Our post-hoc method formulates uncertainty as a Bayesian-regularized linear least-squares optimization over reconstruction residuals. This architecture-agnostic approach extracts a per-primitive uncertainty channel without modifying the underlying scene representation or degrading baseline visual fidelity. Crucially, we demonstrate that providing this actionable reliability signal successfully translates 3D Gaussian splatting into a trustworthy spatial map, further improving state-of-the-art performance across three critical downstream perception tasks: active view selection, pose-agnostic scene change detection, and pose-agnostic anomaly detection.

  </details>



- **Viewport-based Neural 360° Image Compression**  
  Jingwei Liao, Bo Chen, Klara Nahrstedt, Zhisheng Yan  
  _2026-03-24_ · https://arxiv.org/abs/2603.22776v1  
  <details><summary>Abstract</summary>

  Given the popularity of 360° images on social media platforms, 360° image compression becomes a critical technology for media storage and transmission. Conventional 360° image compression pipeline projects the spherical image into a single 2D plane, leading to issues of oversampling and distortion. In this paper, we propose a novel viewport-based neural compression pipeline for 360° images. By replacing the image projection in conventional 360° image compression pipelines with viewport extraction and efficiently compressing multiple viewports, the proposed pipeline minimizes the inherent oversampling and distortion issues. However, viewport extraction impedes information sharing between multiple viewports during compression, causing the loss of global information about the spherical image. To tackle this global information loss, we design a neural viewport codec to capture global prior information across multiple viewports and maximally compress the viewport data. The viewport codec is empowered by a transformer-based ViewPort ConText (VPCT) module that can be integrated with canonical learning-based 2D image compression structures. We compare the proposed pipeline with existing 360° image compression models and conventional 360° image compression pipelines building on learning-based 2D image codecs and standard hand-crafted codecs. Results show that our pipeline saves an average of $14.01\%$ bit consumption compared to the best-performing 360° image compression methods without compromising quality. The proposed VPCT-based codec also outperforms existing 2D image codecs in the viewport-based neural compression pipeline. Our code can be found at: https://github.com/Jingwei-Liao/VPCT.

  </details>



- **MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping**  
  Shiyao Li, Antoine Guédon, Shizhe Chen, Vincent Lepetit  
  _2026-03-23_ · https://arxiv.org/abs/2603.22650v1  
  <details><summary>Abstract</summary>

  Active mapping aims to determine how an agent should move to efficiently reconstruct an unknown environment. Most existing approaches rely on greedy next-best-view prediction, resulting in inefficient exploration and incomplete scene reconstruction. To address this limitation, we introduce MAGICIAN, a novel long-term planning framework that maximizes accumulated surface coverage gain through Imagined Gaussians, a scene representation derived from a pre-trained occupancy network with strong structural priors. This representation enables efficient computation of coverage gain for any novel viewpoint via fast volumetric rendering, allowing its integration into a tree-search algorithm for long-horizon planning. We update Imagined Gaussians and refine the planned trajectory in a closed-loop manner. Our method achieves state-of-the-art performance across indoor and outdoor benchmarks with varying action spaces, demonstrating the critical advantage of long-term planning in active mapping.

  </details>



- **Repurposing Geometric Foundation Models for Multi-view Diffusion**  
  Wooseok Jang, Seonghu Jeon, Jisang Han, Jinhyeok Choi, Minkyung Kwon, Seungryong Kim, Saining Xie, Sainan Liu  
  _2026-03-23_ · https://arxiv.org/abs/2603.22275v1  
  <details><summary>Abstract</summary>

  While recent advances in generative latent spaces have driven substantial progress in single-image generation, the optimal latent space for novel view synthesis (NVS) remains largely unexplored. In particular, NVS requires geometrically consistent generation across viewpoints, but existing approaches typically operate in a view-independent VAE latent space. In this paper, we propose Geometric Latent Diffusion (GLD), a framework that repurposes the geometrically consistent feature space of geometric foundation models as the latent space for multi-view diffusion. We show that these features not only support high-fidelity RGB reconstruction but also encode strong cross-view geometric correspondences, providing a well-suited latent space for NVS. Our experiments demonstrate that GLD outperforms both VAE and RAE on 2D image quality and 3D consistency metrics, while accelerating training by more than 4.4x compared to the VAE latent space. Notably, GLD remains competitive with state-of-the-art methods that leverage large-scale text-to-image pretraining, despite training its diffusion model from scratch without such generative pretraining.

  </details>


