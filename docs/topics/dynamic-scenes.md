# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-02 07:39 UTC_

Total papers shown: **9**


---

- **LAtent Phase Inference from Short time sequences using SHallow REcurrent Decoders (LAPIS-SHRED)**  
  Yuxuan Bao, Xingyue Zhang, J. Nathan Kutz  
  _2026-04-01_ · https://arxiv.org/abs/2604.01216v1  
  <details><summary>Abstract</summary>

  Reconstructing full spatio-temporal dynamics from sparse observations in both space and time remains a central challenge in complex systems, as measurements can be spatially incomplete and can be also limited to narrow temporal windows. Yet approximating the complete spatio-temporal trajectory is essential for mechanistic insight and understanding, model calibration, and operational decision-making. We introduce LAPIS-SHRED (LAtent Phase Inference from Short time sequence using SHallow REcurrent Decoders), a modular architecture that reconstructs and/or forecasts complete spatiotemporal dynamics from sparse sensor observations confined to short temporal windows. LAPIS-SHRED operates through a three-stage pipeline: (i) a SHRED model is pre-trained entirely on simulation data to map sensor time-histories into a structured latent space, (ii) a temporal sequence model, trained on simulation-derived latent trajectories, learns to propagate latent states forward or backward in time to span unobserved temporal regions from short observational time windows, and (iii) at deployment, only a short observation window of hyper-sparse sensor measurements from the true system is provided, from which the frozen SHRED model and the temporal model jointly reconstruct or forecast the complete spatiotemporal trajectory. The framework supports bidirectional inference, inherits data assimilation and multiscale reconstruction capabilities from its modular structure, and accommodates extreme observational constraints including single-frame terminal inputs. We evaluate LAPIS-SHRED on six experiments spanning complex spatio-temporal physics: turbulent flows, multiscale propulsion physics, volatile combustion transients, and satellite-derived environmental fields, highlighting a lightweight, modular architecture suited for operational settings where observation is constrained by physical or logistical limitations.

  </details>



- **ACT Now: Preempting LVLM Hallucinations via Adaptive Context Integration**  
  Bei Yan, Yuecong Min, Jie Zhang, Shiguang Shan, Xilin Chen  
  _2026-04-01_ · https://arxiv.org/abs/2604.00983v1  
  <details><summary>Abstract</summary>

  Large Vision-Language Models (LVLMs) frequently suffer from severe hallucination issues. Existing mitigation strategies predominantly rely on isolated, single-step states to enhance visual focus or suppress strong linguistic priors. However, these static approaches neglect dynamic context changes across the generation process and struggles to correct inherited information loss. To address this limitation, we propose Adaptive Context inTegration (ACT), a training-free inference intervention method that mitigates hallucination through the adaptive integration of contextual information. Specifically, we first propose visual context exploration, which leverages spatio-temporal profiling to adaptively amplify attention heads responsible for visual exploration. To further facilitate vision-language alignment, we propose semantic context aggregation that marginalizes potential semantic queries to effectively aggregate visual evidence, thereby resolving the information loss caused by the discrete nature of token prediction. Extensive experiments across diverse LVLMs demonstrate that ACT significantly reduces hallucinations and achieves competitive results on both discriminative and generative benchmarks, acting as a robust and highly adaptable solution without compromising fundamental generation capabilities.

  </details>



- **Learning Quantised Structure-Preserving Motion Representations for Dance Fingerprinting**  
  Arina Kharlamova, Bowei He, Chen Ma, Xue Liu  
  _2026-04-01_ · https://arxiv.org/abs/2604.00927v1  
  <details><summary>Abstract</summary>

  We present DANCEMATCH, an end-to-end framework for motion-based dance retrieval, the task of identifying semantically similar choreographies directly from raw video, defined as DANCE FINGERPRINTING. While existing motion analysis and retrieval methods can compare pose sequences, they rely on continuous embeddings that are difficult to index, interpret, or scale. In contrast, DANCEMATCH constructs compact, discrete motion signatures that capture the spatio-temporal structure of dance while enabling efficient large-scale retrieval. Our system integrates Skeleton Motion Quantisation (SMQ) with Spatio-Temporal Transformers (STT) to encode human poses, extracted via Apple CoMotion, into a structured motion vocabulary. We further design DANCE RETRIEVAL ENGINE (DRE), which performs sub-linear retrieval using a histogram-based index followed by re-ranking for refined matching. To facilitate reproducible research, we release DANCETYPESBENCHMARK, a pose-aligned dataset annotated with quantised motion tokens. Experiments demonstrate robust retrieval across diverse dance styles and strong generalisation to unseen choreographies, establishing a foundation for scalable motion fingerprinting and quantitative choreographic analysis.

  </details>



- **TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting**  
  Suwoong Yeom, Joonsik Nam, Seunggyu Choi, Lucas Yunkyu Lee, Sangmin Kim, Jaesik Park, Joonsoo Kim, Kugjin Yun, Kyeongbo Kong, Sukju Kang  
  _2026-04-01_ · https://arxiv.org/abs/2604.00538v1  
  <details><summary>Abstract</summary>

  Recent 4D Gaussian Splatting (4DGS) methods achieve impressive dynamic scene reconstruction but often rely on piecewise linear velocity approximations and short temporal windows. This disjointed modeling leads to severe temporal fragmentation, forcing primitives to be repeatedly eliminated and regenerated to track complex nonlinear dynamics. This makeshift approximation eliminates the long-term temporal identity of objects and causes an inevitable proliferation of Gaussians, hindering scalability to extended video sequences. To address this, we propose TRiGS, a novel 4D representation that utilizes unified, continuous geometric transformations. By integrating $SE(3)$ transformations, hierarchical Bezier residuals, and learnable local anchors, TRiGS models geometrically consistent rigid motions for individual primitives. This continuous formulation preserves temporal identity and effectively mitigates unbounded memory growth. Extensive experiments demonstrate that TRiGS achieves high fidelity rendering on standard benchmarks while uniquely scaling to extended video sequences (e.g., 600 to 1200 frames) without severe memory bottlenecks, significantly outperforming prior works in temporal stability.

  </details>



- **GRVS: a Generalizable and Recurrent Approach to Monocular Dynamic View Synthesis**  
  Thomas Tanay, Mohammed Brahimi, Michal Nazarczuk, Qingwen Zhang, Sibi Catley-Chandar, Arthur Moreau, Zhensong Zhang, Eduardo Pérez-Pellitero  
  _2026-03-31_ · https://arxiv.org/abs/2603.29734v1  
  <details><summary>Abstract</summary>

  Synthesizing novel views from monocular videos of dynamic scenes remains a challenging problem. Scene-specific methods that optimize 4D representations with explicit motion priors often break down in highly dynamic regions where multi-view information is hard to exploit. Diffusion-based approaches that integrate camera control into large pre-trained models can produce visually plausible videos but frequently suffer from geometric inconsistencies across both static and dynamic areas. Both families of methods also require substantial computational resources. Building on the success of generalizable models for static novel view synthesis, we adapt the framework to dynamic inputs and propose a new model with two key components: (1) a recurrent loop that enables unbounded and asynchronous mapping between input and target videos and (2) an efficient use of plane sweeps over dynamic inputs to disentangle camera and scene motion, and achieve fine-grained, six-degrees-of-freedom camera controls. We train and evaluate our model on the UCSD dataset and on Kubric-4D-dyn, a new monocular dynamic dataset featuring longer, higher resolution sequences with more complex scene dynamics than existing alternatives. Our model outperforms four Gaussian Splatting-based scene-specific approaches, as well as two diffusion-based approaches in reconstructing fine-grained geometric details across both static and dynamic regions.

  </details>



- **Video-Oasis: Rethinking Evaluation of Video Understanding**  
  Geuntaek Lim, Minho Shim, Sungjune Park, Jaeyun Lee, Inwoong Lee, Taeoh Kim, Dongyoon Wee, Yukyung Choi  
  _2026-03-31_ · https://arxiv.org/abs/2603.29616v1  
  <details><summary>Abstract</summary>

  The inherent complexity of video understanding makes it difficult to attribute whether performance gains stem from visual perception, linguistic reasoning, or knowledge priors. While many benchmarks have emerged to assess high-level reasoning, the essential criteria that constitute video understanding remain largely overlooked. Instead of introducing yet another benchmark, we take a step back to re-examine the current landscape of video understanding. In this work, we provide Video-Oasis, a sustainable diagnostic suite designed to systematically evaluate existing evaluations and distill spatio-temporal challenges for video understanding. Our analysis reveals two critical findings: (1) 54% of existing benchmark samples are solvable without visual input or temporal context, and (2) on the remaining samples, state-of-the-art models exhibit performance barely exceeding random guessing. To bridge this gap, we investigate which algorithmic design choices contribute to robust video understanding, providing practical guidelines for future research. We hope our work serves as a standard guideline for benchmark construction and the rigorous evaluation of architecture development. Code is available at https://github.com/sejong-rcv/Video-Oasis.

  </details>



- **MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting**  
  Haoran Zhou, Gim Hee Lee  
  _2026-03-31_ · https://arxiv.org/abs/2603.29296v1  
  <details><summary>Abstract</summary>

  Realistic reconstruction of dynamic 4D scenes from monocular videos is essential for understanding the physical world. Despite recent progress in neural rendering, existing methods often struggle to recover accurate 3D geometry and temporally consistent motion in complex environments. To address these challenges, we propose MotionScale, a 4D Gaussian Splatting framework that scales efficiently to large scenes and extended sequences while maintaining high-fidelity structural and motion coherence. At the core of our approach is a scalable motion field parameterized by cluster-centric basis transformations that adaptively expand to capture diverse and evolving motion patterns. To ensure robust reconstruction over long durations, we introduce a progressive optimization strategy comprising two decoupled propagation stages: 1) A background extension stage that adapts to newly visible regions, refines camera poses, and explicitly models transient shadows; 2) A foreground propagation stage that enforces motion consistency through a specialized three-stage refinement process. Extensive experiments on challenging real-world benchmarks demonstrate that MotionScale significantly outperforms state-of-the-art methods in both reconstruction quality and temporal stability. Project page: https://hrzhou2.github.io/motion-scale-web/.

  </details>



- **GenFusion: Feed-forward Human Performance Capture via Progressive Canonical Space Updates**  
  Youngjoong Kwon, Yao He, Heejung Choi, Chen Geng, Zhengmao Liu, Jiajun Wu, Ehsan Adeli  
  _2026-03-30_ · https://arxiv.org/abs/2603.28997v1  
  <details><summary>Abstract</summary>

  We present a feed-forward human performance capture method that renders novel views of a performer from a monocular RGB stream. A key challenge in this setting is the lack of sufficient observations, especially for unseen regions. Assuming the subject moves continuously over time, we take advantage of the fact that more body parts become observable by maintaining a canonical space that is progressively updated with each incoming frame. This canonical space accumulates appearance information over time and serves as a context bank when direct observations are missing in the current live frame. To effectively utilize this context while respecting the deformation of the live state, we formulate the rendering process as probabilistic regression. This resolves conflicts between past and current observations, producing sharper reconstructions than deterministic regression approaches. Furthermore, it enables plausible synthesis even in regions with no prior observations. Experiments on in-domain (4D-Dress) and out-of-distribution (MVHumanNet) datasets demonstrate the effectiveness of our approach.

  </details>



- **AutoWorld: Scaling Multi-Agent Traffic Simulation with Self-Supervised World Models**  
  Mozhgan Pourkeshavatz, Tianran Liu, Nicholas Rhinehart  
  _2026-03-30_ · https://arxiv.org/abs/2603.28963v1  
  <details><summary>Abstract</summary>

  Multi-agent traffic simulation is central to developing and testing autonomous driving systems. Recent data-driven simulators have achieved promising results, but rely heavily on supervised learning from labeled trajectories or semantic annotations, making it costly to scale their performance. Meanwhile, large amounts of unlabeled sensor data can be collected at scale but remain largely unused by existing traffic simulation frameworks. This raises a key question: How can a method harness unlabeled data to improve traffic simulation performance? In this work, we propose AutoWorld, a traffic simulation framework that employs a world model learned from unlabeled occupancy representations of LiDAR data. Given world model samples, AutoWorld constructs a coarse-to-fine predictive scene context as input to a multi-agent motion generation model. To promote sample diversity, AutoWorld uses a cascaded Determinantal Point Process framework to guide the sampling processes of both the world model and the motion model. Furthermore, we designed a motion-aware latent supervision objective that enhances AutoWorld's representation of scene dynamics. Experiments on the WOSAC benchmark show that AutoWorld ranks first on the leaderboard according to the primary Realism Meta Metric (RMM). We further show that simulation performance consistently improves with the inclusion of unlabeled LiDAR data, and study the efficacy of each component with ablations. Our method paves the way for scaling traffic simulation realism without additional labeling. Our project page contains additional visualizations and released code.

  </details>


