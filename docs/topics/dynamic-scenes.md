# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-31 07:42 UTC_

Total papers shown: **10**


---

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



- **HMPDM: A Diffusion Model for Driving Video Prediction with Historical Motion Priors**  
  Ke Li, Tianjia Yang, Kaidi Liang, Xianbiao Hu, Ruwen Qin  
  _2026-03-28_ · https://arxiv.org/abs/2603.27371v1  
  <details><summary>Abstract</summary>

  Video prediction is a useful function for autonomous driving, enabling intelligent vehicles to reliably anticipate how driving scenes will evolve and thereby supporting reasoning and safer planning. However, existing models are constrained by multi-stage training pipelines and remain insufficient in modeling the diverse motion patterns in real driving scenes, leading to degraded temporal consistency and visual quality. To address these challenges, this paper introduces the historical motion priors-informed diffusion model (HMPDM), a video prediction model that leverages historical motion priors to enhance motion understanding and temporal coherence. The proposed deep learning system introduces three key designs: (i) a Temporal-aware Latent Conditioning (TaLC) module for implicit historical motion injection; (ii) a Motion-aware Pyramid Encoder (MaPE) for multi-scale motion representation; (iii) a Self-Conditioning (SC) strategy for stable iterative denoising. Extensive experiments on the Cityscapes and KITTI benchmarks demonstrate that HMPDM outperforms state-of-the-art video prediction methods with efficiency, achieving a 28.2% improvement in FVD on Cityscapes under the same monocular RGB input configuration setting. The implementation codes are publicly available at https://github.com/KELISBU/HMPDM.

  </details>



- **Complet4R: Geometric Complete 4D Reconstruction**  
  Weibang Wang, Kenan Li, Zhuoguang Chen, Yijun Yuan, Hang Zhao  
  _2026-03-28_ · https://arxiv.org/abs/2603.27300v1  
  <details><summary>Abstract</summary>

  We introduce Complet4R, a novel end-to-end framework for Geometric Complete 4D Reconstruction, which aims to recover temporally coherent and geometrically complete reconstruction for dynamic scenes. Our method formalizes the task of Geometric Complete 4D Reconstruction as a unified framework of reconstruction and completion, by directly accumulating full contexts onto each frame. Unlike previous approaches that rely on pairwise reconstruction or local motion estimation, Complet4R utilizes a decoder-only transformer to operate all context globally directly from sequential video input, reconstructing a complete geometry for every single timestamp, including occluded regions visible in other frames. Our method demonstrates the state-of-the-art performance on our proposed benchmark for Geometric Complete 4D Reconstruction and the 3D Point Tracking task. Code will be released to support future research.

  </details>



- **TrackMAE: Video Representation Learning via Track Mask and Predict**  
  Renaud Vandeghen, Fida Mohammad Thoker, Marc Van Droogenbroeck, Bernard Ghanem  
  _2026-03-28_ · https://arxiv.org/abs/2603.27268v1  
  <details><summary>Abstract</summary>

  Masked video modeling (MVM) has emerged as a simple and scalable self-supervised pretraining paradigm, but only encodes motion information implicitly, limiting the encoding of temporal dynamics in the learned representations. As a result, such models struggle on motion-centric tasks that require fine-grained motion awareness. To address this, we propose TrackMAE, a simple masked video modeling paradigm that explicitly uses motion information as a reconstruction signal. In TrackMAE, we use an off-the-shelf point tracker to sparsely track points in the input videos, generating motion trajectories. Furthermore, we exploit the extracted trajectories to improve random tube masking with a motion-aware masking strategy. We enhance video representations learned in both pixel and feature semantic reconstruction spaces by providing a complementary supervision signal in the form of motion targets. We evaluate on six datasets across diverse downstream settings and find that TrackMAE consistently outperforms state-of-the-art video self-supervised learning baselines, learning more discriminative and generalizable representations. Code available at https://github.com/rvandeghen/TrackMAE

  </details>



- **Seeing the Scene Matters: Revealing Forgetting in Video Understanding Models with a Scene-Aware Long-Video Benchmark**  
  Seng Nam Chen, Hao Chen, Chenglam Ho, Xinyu Mao, Jinping Wang, Yu Zhang, Chao Li  
  _2026-03-28_ · https://arxiv.org/abs/2603.27259v1  
  <details><summary>Abstract</summary>

  Long video understanding (LVU) remains a core challenge in multimodal learning. Although recent vision-language models (VLMs) have made notable progress, existing benchmarks mainly focus on either fine-grained perception or coarse summarization, offering limited insight into temporal understanding over long contexts. In this work, we define a scene as a coherent segment of a video in which both visual and semantic contexts remain consistent, aligning with human perception. This leads us to a key question: can current VLMs reason effectively over long, scene-level contexts? To answer this, we introduce a new benchmark, SceneBench, designed to provide scene-level challenges. Our evaluation reveals a sharp drop in accuracy when VLMs attempt to answer scene-level questions, indicating significant forgetting of long-range context. To further validate these findings, we propose Scene Retrieval-Augmented Generation (Scene-RAG), which constructs a dynamic scene memory by retrieving and integrating relevant context across scenes. This Scene-RAG improves VLM performance by +2.50%, confirming that current models still struggle with long-context retention. We hope SceneBench will encourage future research toward VLMs with more robust, human-like video comprehension.

  </details>


