# NeRF & Neural Radiance Fields

_Updated: 2026-03-03 07:08 UTC_

Total papers shown: **10**


---

- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution**  
  Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02134v1  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

  </details>



- **Tiny-DroNeRF: Tiny Neural Radiance Fields aboard Federated Learning-enabled Nano-drones**  
  Ilenia Carboni, Elia Cereda, Lorenzo Lamberti, Daniele Malpetti, Francesco Conti, Daniele Palossi  
  _2026-03-02_ · https://arxiv.org/abs/2603.01850v1  
  <details><summary>Abstract</summary>

  Sub-30g nano-sized aerial robots can leverage their agility and form factor to autonomously explore cluttered and narrow environments, like in industrial inspection and search and rescue missions. However, the price for their tiny size is a strong limit in their resources, i.e., sub-100 mW microcontroller units (MCUs) delivering $\sim$100 GOps/s at best, and memory budgets well below 100 MB. Despite these strict constraints, we aim to enable complex vision-based tasks aboard nano-drones, such as dense 3D scene reconstruction: a key robotic task underlying fundamental capabilities like spatial awareness and motion planning. Top-performing 3D reconstruction methods leverage neural radiance fields (NeRF) models, which require GBs of memory and massive computation, usually delivered by high-end GPUs consuming 100s of Watts. Our work introduces Tiny-DroNeRF, a lightweight NeRF model, based on Instant-NGP, and optimized for running on a GAP9 ultra-low-power (ULP) MCU aboard our nano-drones. Then, we further empower our Tiny-DroNeRF by leveraging a collaborative federated learning scheme, which distributes the model training among multiple nano-drones. Our experimental results show a 96% reduction in Tiny-DroNeRF's memory footprint compared to Instant-NGP, with only a 5.7 dB drop in reconstruction accuracy. Finally, our federated learning scheme allows Tiny-DroNeRF to train with an amount of data otherwise impossible to keep in a single drone's memory, increasing the overall reconstruction accuracy. Ultimately, our work combines, for the first time, NeRF training on an ULP MCU with federated learning on nano-drones.

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



- **HeroGS: Hierarchical Guidance for Robust 3D Gaussian Splatting under Sparse Views**  
  Jiashu Li, Xumeng Han, Zhaoyang Wei, Zipeng Wang, Kuiran Wang, Guorong Li, Zhenjun Han, Jianbin Jiao  
  _2026-03-01_ · https://arxiv.org/abs/2603.01099v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a promising approach in novel view synthesis, combining photorealistic rendering with real-time efficiency. However, its success heavily relies on dense camera coverage; under sparse-view conditions, insufficient supervision leads to irregular Gaussian distributions, characterized by globally sparse coverage, blurred background, and distorted high-frequency areas. To address this, we propose HeroGS, Hierarchical Guidance for Robust 3D Gaussian Splatting, a unified framework that establishes hierarchical guidance across the image, feature, and parameter levels. At the image level, sparse supervision is converted into pseudo-dense guidance, globally regularizing the Gaussian distributions and forming a consistent foundation for subsequent optimization. Building upon this, Feature-Adaptive Densification and Pruning (FADP) at the feature level leverages low-level features to refine high-frequency details and adaptively densifies Gaussians in background regions. The optimized distributions then support Co-Pruned Geometry Consistency (CPG) at parameter level, which guides geometric consistency through parameter freezing and co-pruning, effectively removing inconsistent splats. The hierarchical guidance strategy effectively constrains and optimizes the overall Gaussian distributions, thereby enhancing both structural fidelity and rendering quality. Extensive experiments demonstrate that HeroGS achieves high-fidelity reconstructions and consistently surpasses state-of-the-art baselines under sparse-view conditions.

  </details>



- **GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis**  
  Xuqin Wang, Tao Wu, Yanfeng Zhang, Lu Liu, Mingwei Sun, Yongliang Wang, Niclas Zeller, Daniel Cremers  
  _2026-03-01_ · https://arxiv.org/abs/2603.01010v1  
  <details><summary>Abstract</summary>

  Recent advances in generative modeling have substantially enhanced novel view synthesis, yet maintaining consistency across viewpoints remains challenging. Diffusion-based models rely on stochastic noise-to-data transitions, which obscure deterministic structures and yield inconsistent view predictions. We propose a Data-to-Data Flow Matching framework that learns deterministic transformations directly between paired views, enhancing view-consistent synthesis through explicit data coupling. To further enhance geometric coherence, we introduce Probability Density Geodesic Flow Matching (PDG-FM), which constrains flow trajectories using geodesic interpolants derived from probability density metrics of pretrained diffusion models. Such alignment with high-density regions of the data manifold promotes more realistic interpolants between samples. Empirically, our method surpasses diffusion-based NVS baselines, demonstrating improved structural coherence and smoother transitions across views. These results highlight the advantages of incorporating data-dependent geometric regularization into deterministic flow matching for consistent novel view generation.

  </details>



- **StegoNGP: 3D Cryptographic Steganography using Instant-NGP**  
  Wenxiang Jiang, Yujun Lan, Shuo Zhao, Yuanshan Liu, Mingzhu Zhou, Jinxin Wang  
  _2026-03-01_ · https://arxiv.org/abs/2603.00949v1  
  <details><summary>Abstract</summary>

  Recently, Instant Neural Graphics Primitives (Instant-NGP) has achieved significant success in rapid 3D scene reconstruction, but securely embedding high-capacity hidden data, such as an entire 3D scene, remains a challenge. Existing methods rely on external decoders, require architectural modifications, and suffer from limited capacity, which makes them easily detectable. We propose a novel parameter-free 3D Cryptographic Steganography using Instant-NGP (StegoNGP), which leverages the Instant-NGP hash encoding function as a key-controlled scene switcher. By associating a default key with a cover scene and a secret key with a hidden scene, our method trains a single model to interweave both representations within the same network weights. The resulting model is indistinguishable from a standard Instant-NGP in architecture and parameter count. We also introduce an enhanced Multi-Key scheme, which assigns multiple independent keys across hash levels, dramatically expanding the key space and providing high robustness against partial key disclosure attacks. Experimental results demonstrated that StegoNGP can hide a complete high-quality 3D scene with strong imperceptibility and security, providing a new paradigm for high-capacity, undetectable information hiding in neural fields. The code can be found at https://github.com/jiang-wenxiang/StegoNGP.

  </details>



- **NERFIFY: A Multi-Agent Framework for Turning NeRF Papers into Code**  
  Seemandhar Jain, Keshav Gupta, Kunal Gupta, Manmohan Chandraker  
  _2026-02-28_ · https://arxiv.org/abs/2603.00805v1  
  <details><summary>Abstract</summary>

  The proliferation of neural radiance field (NeRF) research requires significant efforts to reimplement papers before building upon them. We introduce NERFIFY, a multi-agent framework that reliably converts NeRF research papers into trainable Nerfstudio plugins, in contrast to generic paper-to-code methods and frontier models like GPT-5 that usually fail to produce runnable code. NERFIFY achieves domain-specific executability through six key innovations: (1) Context-free grammar (CFG): LLM synthesis is constrained by Nerfstudio formalized as a CFG, ensuring generated code satisfies architectural invariants. (2) Graph-of-Thought code synthesis: Specialized multi-file-agents generate repositories in topological dependency order, validating contracts and errors at each node. (3) Compositional citation recovery: Agents automatically retrieve and integrate components (samplers, encoders, proposal networks) from citation graphs of references. (4) Visual feedback: Artifacts are diagnosed through PSNR-minima ROI analysis, cross-view geometric validation, and VLM-guided patching to iteratively improve quality. (5) Knowledge enhancement: Beyond reproduction, methods can be improved with novel optimizations. (6) Benchmarking: An evaluation framework is designed for NeRF paper-to-code synthesis across 30 diverse papers. On papers without public implementations, NERFIFY achieves visual quality matching expert human code (+/-0.5 dB PSNR, +/-0.2 SSIM) while reducing implementation time from weeks to minutes. NERFIFY demonstrates that a domain-aware design enables code translation for complex vision papers, potentiating accelerated and democratized reproducible research. Code, data and implementations will be publicly released.

  </details>



- **TokenSplat: Token-aligned 3D Gaussian Splatting for Feed-forward Pose-free Reconstruction**  
  Yihui Li, Chengxin Lv, Zichen Tang, Hongyu Yang, Di Huang  
  _2026-02-28_ · https://arxiv.org/abs/2603.00697v1  
  <details><summary>Abstract</summary>

  We present TokenSplat, a feed-forward framework for joint 3D Gaussian reconstruction and camera pose estimation from unposed multi-view images. At its core, TokenSplat introduces a Token-aligned Gaussian Prediction module that aligns semantically corresponding information across views directly in the feature space. Guided by coarse token positions and fusion confidence, it aggregates multi-scale contextual features to enable long-range cross-view reasoning and reduce redundancy from overlapping Gaussians. To further enhance pose robustness and disentangle viewpoint cues from scene semantics, TokenSplat employs learnable camera tokens and an Asymmetric Dual-Flow Decoder (ADF-Decoder) that enforces directionally constrained communication between camera and image tokens. This maintains clean factorization within a feed-forward architecture, enabling coherent reconstruction and stable pose estimation without iterative refinement. Extensive experiments demonstrate that TokenSplat achieves higher reconstruction fidelity and novel-view synthesis quality in pose-free settings, and significantly improves pose estimation accuracy compared to prior pose-free methods. Project page: https://kidleyh.github.io/tokensplat/.

  </details>


