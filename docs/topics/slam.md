# SLAM & Localization

_Updated: 2026-04-12 07:42 UTC_

Total papers shown: **3**


---

- **Novel View Synthesis as Video Completion**  
  Qi Wu, Khiem Vuong, Minsik Jeon, Srinivasa Narasimhan, Deva Ramanan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08500v1  
  <details><summary>Abstract</summary>

  We tackle the problem of sparse novel view synthesis (NVS) using video diffusion models; given $K$ ($\approx 5$) multi-view images of a scene and their camera poses, we predict the view from a target camera pose. Many prior approaches leverage generative image priors encoded via diffusion models. However, models trained on single images lack multi-view knowledge. We instead argue that video models already contain implicit multi-view knowledge and so should be easier to adapt for NVS. Our key insight is to formulate sparse NVS as a low frame-rate video completion task. However, one challenge is that sparse NVS is defined over an unordered set of inputs, often too sparse to admit a meaningful order, so the models should be $\textit{invariant}$ to permutations of that input set. To this end, we present FrameCrafter, which adapts video models (naturally trained with coherent frame orderings) to permutation-invariant NVS through several architectural modifications, including per-frame latent encodings and removal of temporal positional embeddings. Our results suggest that video models can be easily trained to "forget" about time with minimal supervision, producing competitive performance on sparse-view NVS benchmarks. Project page: https://frame-crafter.github.io/

  </details>



- **State and Trajectory Estimation of Tensegrity Robots via Factor Graphs and Chebyshev Polynomials**  
  Edgar Granados, Patrick Meng, Charles Tang, Shrimed Sangani, William R. Johnson, Rebecca Kramer-Bottiglio, Kostas Bekris  
  _2026-04-09_ · https://arxiv.org/abs/2604.08185v1  
  <details><summary>Abstract</summary>

  Tensegrity robots offer compliance and adaptability, but their nonlinear, and underconstrained dynamics make state estimation challenging. Reliable continuous-time estimation of all rigid links is crucial for closed-loop control, system identification, and machine learning; however, conventional methods often fall short. This paper proposes a two-stage approach for robust state or trajectory estimation (i.e., filtering or smoothing) of a cable-driven tensegrity robot. For online state estimation, this work introduces a factor-graph-based method, which fuses measurements from an RGB-D camera with on-board cable length sensors. To the best of the authors' knowledge, this is the first application of factor graphs in this domain. Factor graphs are a natural choice, as they exploit the robot's structural properties and provide effective sensor fusion solutions capable of handling nonlinearities in practice. Both the Mahalanobis distance-based clustering algorithm, used to handle noise, and the Chebyshev polynomial method, used to estimate the most probable velocities and intermediate states, are shown to perform well on simulated and real-world data, compared to an ICP-based algorithm. Results show that the approach provides high fidelity, continuous-time state and trajectory estimates for complex tensegrity robot motions.

  </details>



- **MotionScape: A Large-Scale Real-World Highly Dynamic UAV Video Dataset for World Models**  
  Zile Guo, Zhan Chen, Enze Zhu, Kan Wei, Yongkang Zou, Xiaoxuan Liu, Lei Wang  
  _2026-04-09_ · https://arxiv.org/abs/2604.07991v1  
  <details><summary>Abstract</summary>

  Recent advances in world models have demonstrated strong capabilities in simulating physical reality, making them an increasingly important foundation for embodied intelligence. For UAV agents in particular, accurate prediction of complex 3D dynamics is essential for autonomous navigation and robust decision-making in unconstrained environments. However, under the highly dynamic camera trajectories typical of UAV views, existing world models often struggle to maintain spatiotemporal physical consistency. A key reason lies in the distribution bias of current training data: most existing datasets exhibit restricted 2.5D motion patterns, such as ground-constrained autonomous driving scenes or relatively smooth human-centric egocentric videos, and therefore lack realistic high-dynamic 6-DoF UAV motion priors. To address this gap, we present MotionScape, a large-scale real-world UAV-view video dataset with highly dynamic motion for world modeling. MotionScape contains over 30 hours of 4K UAV-view videos, totaling more than 4.5M frames. This novel dataset features semantically and geometrically aligned training samples, where diverse real-world UAV videos are tightly coupled with accurate 6-DoF camera trajectories and fine-grained natural language descriptions. To build the dataset, we develop an automated multi-stage processing pipeline that integrates CLIP-based relevance filtering, temporal segmentation, robust visual SLAM for trajectory recovery, and large-language-model-driven semantic annotation. Extensive experiments show that incorporating such semantically and geometrically aligned annotations effectively improves the ability of existing world models to simulate complex 3D dynamics and handle large viewpoint shifts, thereby benefiting decision-making and planning for UAV agents in complex environments. The dataset is publicly available at https://github.com/Thelegendzz/MotionScape

  </details>


