# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-04-01 07:51 UTC_

Total papers shown: **11**


---

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



- **Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes**  
  Luke Palmer, Petar Palasek, Hazem Abdelkawy  
  _2026-03-30_ · https://arxiv.org/abs/2603.28319v1  
  <details><summary>Abstract</summary>

  Accurately modelling human attention is essential for numerous computer vision applications, particularly in the domain of automotive safety. Existing methods typically collapse gaze into saliency maps or scanpaths, treating gaze dynamics only implicitly. We instead formulate gaze modelling as an autoregressive dynamical system and explicitly unroll raw gaze trajectories over time, conditioned on both gaze history and the evolving environment. Driving scenes are represented as gaze-centric graphs processed by the Affinity Relation Transformer (ART), a heterogeneous graph transformer that models interactions between driver gaze, traffic objects, and road structure. We further introduce the Object Density Network (ODN) to predict next-step gaze distributions, capturing the stochastic and object-centric nature of attentional shifts in complex environments. We also release Focus100, a new dataset of raw gaze data from 30 participants viewing egocentric driving footage. Trained directly on raw gaze, without fixation filtering, our unified approach produces more natural gaze trajectories, scanpath dynamics, and saliency maps than existing attention models, offering valuable insights for the temporal modelling of human attention in dynamic environments.

  </details>



- **\textit{4DSurf}: High-Fidelity Dynamic Scene Surface Reconstruction**  
  Renjie Wu, Hongdong Li, Jose M. Alvarez, Miaomiao Liu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28064v1  
  <details><summary>Abstract</summary>

  This paper addresses the problem of dynamic scene surface reconstruction using Gaussian Splatting (GS), aiming to recover temporally consistent geometry. While existing GS-based dynamic surface reconstruction methods can yield superior reconstruction, they are typically limited to either a single object or objects with only small deformations, struggling to maintain temporally consistent surface reconstruction of large deformations over time. We propose ``\textit{4DSurf}'', a novel and unified framework for generic dynamic surface reconstruction that does not require specifying the number or types of objects in the scene, can handle large surface deformations and temporal inconsistency in reconstruction. The key innovation of our framework is the introduction of Gaussian deformations induced Signed Distance Function Flow Regularization that constrains the motion of Gaussians to align with the evolving surface. To handle large deformations, we introduce an Overlapping Segment Partitioning strategy that divides the sequence into overlapping segments with small deformations and incrementally passes geometric information across segments through the shared overlapping timestep. Experiments on two challenging dynamic scene datasets, Hi4D and CMU Panoptic, demonstrate that our method outperforms state-of-the-art surface reconstruction methods by 49\% and 19\% in Chamfer distance, respectively, and achieves superior temporal consistency under sparse-view settings.

  </details>



- **Event6D: Event-based Novel Object 6D Pose Tracking**  
  Jae-Young Kang, Hoonehee Cho, Taeyeop Lee, Minjun Kang, Bowen Wen, Youngho Kim, Kuk-Jin Yoon  
  _2026-03-30_ · https://arxiv.org/abs/2603.28045v1  
  <details><summary>Abstract</summary>

  Event cameras provide microsecond latency, making them suitable for 6D object pose tracking in fast, dynamic scenes where conventional RGB and depth pipelines suffer from motion blur and large pixel displacements. We introduce EventTrack6D, an event-depth tracking framework that generalizes to novel objects without object-specific training by reconstructing both intensity and depth at arbitrary timestamps between depth frames. Conditioned on the most recent depth measurement, our dual reconstruction recovers dense photometric and geometric cues from sparse event streams. Our EventTrack6D operates at over 120 FPS and maintains temporal consistency under rapid motion. To support training and evaluation, we introduce a comprehensive benchmark suite: a large-scale synthetic dataset for training and two complementary evaluation sets, including real and simulated event datasets. Trained exclusively on synthetic data, EventTrack6D generalizes effectively to real-world scenarios without fine-tuning, maintaining accurate tracking across diverse objects and motion patterns. Our method and datasets validate the effectiveness of event cameras for event-based 6D pose tracking of novel objects. Code and datasets are publicly available at https://chohoonhee.github.io/Event6D.

  </details>



- **RehearsalNeRF: Decoupling Intrinsic Neural Fields of Dynamic Illuminations for Scene Editing**  
  Changyeon Won, Hyunjun Jung, Jungu Cho, Seonmi Park, Chi-Hoon Lee, Hae-Gon Jeon  
  _2026-03-30_ · https://arxiv.org/abs/2603.27948v1  
  <details><summary>Abstract</summary>

  Although there has been significant progress in neural radiance fields, an issue on dynamic illumination changes still remains unsolved. Different from relevant works that parameterize time-variant/-invariant components in scenes, subjects' radiance is highly entangled with their own emitted radiance and lighting colors in spatio-temporal domain. In this paper, we present a new effective method to learn disentangled neural fields under the severe illumination changes, named RehearsalNeRF. Our key idea is to leverage scenes captured under stable lighting like rehearsal stages, easily taken before dynamic illumination occurs, to enforce geometric consistency between the different lighting conditions. In particular, RehearsalNeRF employs a learnable vector for lighting effects which represents illumination colors in a temporal dimension and is used to disentangle projected light colors from scene radiance. Furthermore, our RehearsalNeRF is also able to reconstruct the neural fields of dynamic objects by simply adopting off-the-shelf interactive masks. To decouple the dynamic objects, we propose a new regularization leveraging optical flow, which provides coarse supervision for the color disentanglement. We demonstrate the effectiveness of RehearsalNeRF by showing robust performances on novel view synthesis and scene editing under dynamic illumination conditions. Our source code and video datasets will be publicly available.

  </details>



- **FlashSign: Pose-Free Guidance for Efficient Sign Language Video Generation**  
  Liuzhou Zhang, Zeyu Zhang, Biao Wu, Luyao Tang, Zirui Song, Hongyang He, Renda Han, Guangzhen Yao, Huacan Wang, Ronghao Chen, et al.  
  _2026-03-30_ · https://arxiv.org/abs/2603.27915v1  
  <details><summary>Abstract</summary>

  Sign language plays a crucial role in bridging communication gaps between the deaf and hard-of-hearing communities. However, existing sign language video generation models often rely on complex intermediate representations, which limits their flexibility and efficiency. In this work, we propose a novel pose-free framework for real-time sign language video generation. Our method eliminates the need for intermediate pose representations by directly mapping natural language text to sign language videos using a diffusion-based approach. We introduce two key innovations: (1) a pose-free generative model based on the a state-of-the-art diffusion backbone, which learns implicit text-to-gesture alignments without pose estimation, and (2) a Trainable Sliding Tile Attention (T-STA) mechanism that accelerates inference by exploiting spatio-temporal locality patterns. Unlike previous training-free sparsity approaches, T-STA integrates trainable sparsity into both training and inference, ensuring consistency and eliminating the train-test gap. This approach significantly reduces computational overhead while maintaining high generation quality, making real-time deployment feasible. Our method increases video generation speed by 3.07x without compromising video quality. Our contributions open new avenues for real-time, high-quality, pose-free sign language synthesis, with potential applications in inclusive communication tools for diverse communities. Code: https://github.com/AIGeeksGroup/FlashSign.

  </details>



- **V-CAST: Video Curvature-Aware Spatio-Temporal Pruning for Efficient Video Large Language Models**  
  Xinying Lin, Xuyang Liu, Yiyu Wang, Teng Ma, Wenqi Ren  
  _2026-03-29_ · https://arxiv.org/abs/2603.27650v1  
  <details><summary>Abstract</summary>

  Video large language models (VideoLLMs) show strong capability in video understanding, yet long-context inference is still dominated by massive redundant visual tokens in the prefill stage. We revisit token compression for VideoLLMs under a tight budget and identify a key bottleneck, namely insufficient spatio-temporal information coverage. Existing methods often introduce discontinuous coverage through coarse per-frame allocation or scene segmentation, and token merging can further misalign spatio-temporal coordinates under MRoPE-style discrete (t,h,w) bindings. To address these issues, we propose V-CAST (Video Curvature-Aware Spatio-Temporal Pruning), a training-free, plug-and-play pruning policy for long-context video inference. V-CAST casts token compression as a trajectory approximation problem and introduces a curvature-guided temporal allocation module that routes per-frame token budgets to semantic turns and event boundaries. It further adopts a dual-anchor spatial selection mechanism that preserves high-entropy visual evidence without attention intervention, while keeping retained tokens at their original coordinates to maintain positional alignment. Extensive experiments across multiple VideoLLMs of different architectures and scales demonstrate that V-CAST achieves 98.6% of the original performance, outperforms the second-best method by +1.1% on average, and reduces peak memory and total latency to 86.7% and 86.4% of vanilla Qwen3-VL-8B-Instruct.

  </details>


