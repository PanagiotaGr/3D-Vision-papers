# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-14 07:04 UTC_

Total papers shown: **7**


---

- **DreamVideo-Omni: Omni-Motion Controlled Multi-Subject Video Customization with Latent Identity Reinforcement Learning**  
  Yujie Wei, Xinyu Liu, Shiwei Zhang, Hangjie Yuan, Jinbo Xing, Zhekai Chen, Xiang Wang, Haonan Qiu, Rui Zhao, Yutong Feng, et al.  
  _2026-03-12_ · https://arxiv.org/abs/2603.12257v1  
  <details><summary>Abstract</summary>

  While large-scale diffusion models have revolutionized video synthesis, achieving precise control over both multi-subject identity and multi-granularity motion remains a significant challenge. Recent attempts to bridge this gap often suffer from limited motion granularity, control ambiguity, and identity degradation, leading to suboptimal performance on identity preservation and motion control. In this work, we present DreamVideo-Omni, a unified framework enabling harmonious multi-subject customization with omni-motion control via a progressive two-stage training paradigm. In the first stage, we integrate comprehensive control signals for joint training, encompassing subject appearances, global motion, local dynamics, and camera movements. To ensure robust and precise controllability, we introduce a condition-aware 3D rotary positional embedding to coordinate heterogeneous inputs and a hierarchical motion injection strategy to enhance global motion guidance. Furthermore, to resolve multi-subject ambiguity, we introduce group and role embeddings to explicitly anchor motion signals to specific identities, effectively disentangling complex scenes into independent controllable instances. In the second stage, to mitigate identity degradation, we design a latent identity reward feedback learning paradigm by training a latent identity reward model upon a pretrained video diffusion backbone. This provides motion-aware identity rewards in the latent space, prioritizing identity preservation aligned with human preferences. Supported by our curated large-scale dataset and the comprehensive DreamOmni Bench for multi-subject and omni-motion control evaluation, DreamVideo-Omni demonstrates superior performance in generating high-quality videos with precise controllability.

  </details>



- **Dense Dynamic Scene Reconstruction and Camera Pose Estimation from Multi-View Videos**  
  Shuo Sun, Unal Artan, Malcolm Mielle, Achim J. Lilienthaland, Martin Magnusson  
  _2026-03-12_ · https://arxiv.org/abs/2603.12064v1  
  <details><summary>Abstract</summary>

  We address the challenging problem of dense dynamic scene reconstruction and camera pose estimation from multiple freely moving cameras -- a setting that arises naturally when multiple observers capture a shared event. Prior approaches either handle only single-camera input or require rigidly mounted, pre-calibrated camera rigs, limiting their practical applicability. We propose a two-stage optimization framework that decouples the task into robust camera tracking and dense depth refinement. In the first stage, we extend single-camera visual SLAM to the multi-camera setting by constructing a spatiotemporal connection graph that exploits both intra-camera temporal continuity and inter-camera spatial overlap, enabling consistent scale and robust tracking. To ensure robustness under limited overlap, we introduce a wide-baseline initialization strategy using feed-forward reconstruction models. In the second stage, we refine depth and camera poses by optimizing dense inter- and intra-camera consistency using wide-baseline optical flow. Additionally, we introduce MultiCamRobolab, a new real-world dataset with ground-truth poses from a motion capture system. Finally, we demonstrate that our method significantly outperforms state-of-the-art feed-forward models on both synthetic and real-world benchmarks, while requiring less memory.

  </details>



- **Cross-Resolution Attention Network for High-Resolution PM2.5 Prediction**  
  Ammar Kheder, Helmi Toropainen, Wenqing Peng, Samuel Antão, Zhi-Song Liu, Michael Boy  
  _2026-03-12_ · https://arxiv.org/abs/2603.11725v1  
  <details><summary>Abstract</summary>

  Vision Transformers have achieved remarkable success in spatio-temporal prediction, but their scalability remains limited for ultra-high-resolution, continent-scale domains required in real-world environmental monitoring. A single European air-quality map at 1 km resolution comprises 29 million pixels, far beyond the limits of naive self-attention. We introduce CRAN-PM, a dual-branch Vision Transformer that leverages cross-resolution attention to efficiently fuse global meteorological data (25 km) with local high-resolution PM2.5 at the current time (1 km). Instead of including physically driven factors like temperature and topography as input, we further introduce elevation-aware self-attention and wind-guided cross-attention to force the network to learn physically consistent feature representations for PM2.5 forecasting. CRAN-PM is fully trainable and memory-efficient, generating the complete 29-million-pixel European map in 1.8 seconds on a single GPU. Evaluated on daily PM2.5 forecasting throughout Europe in 2022 (362 days, 2,971 European Environment Agency (EEA) stations), it reduces RMSE by 4.7% at T+1 and 10.7% at T+3 compared to the best single-scale baseline, while reducing bias in complex terrain by 36%.

  </details>



- **DyWeight: Dynamic Gradient Weighting for Few-Step Diffusion Sampling**  
  Tong Zhao, Mingkun Lei, Liangyu Yuan, Yanming Yang, Chenxi Song, Yang Wang, Beier Zhu, Chi Zhang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11607v1  
  <details><summary>Abstract</summary>

  Diffusion Models (DMs) have achieved state-of-the-art generative performance across multiple modalities, yet their sampling process remains prohibitively slow due to the need for hundreds of function evaluations. Recent progress in multi-step ODE solvers has greatly improved efficiency by reusing historical gradients, but existing methods rely on handcrafted coefficients that fail to adapt to the non-stationary dynamics of diffusion sampling. To address this limitation, we propose Dynamic Gradient Weighting (DyWeight), a lightweight, learning-based multi-step solver that introduces a streamlined implicit coupling paradigm. By relaxing classical numerical constraints, DyWeight learns unconstrained time-varying parameters that adaptively aggregate historical gradients while intrinsically scaling the effective step size. This implicit time calibration accurately aligns the solver's numerical trajectory with the model's internal denoising dynamics under large integration steps, avoiding complex decoupled parameterizations and optimizations. Extensive experiments on CIFAR-10, FFHQ, AFHQv2, ImageNet64, LSUN-Bedroom, Stable Diffusion and FLUX.1-dev demonstrate that DyWeight achieves superior visual fidelity and stability with significantly fewer function evaluations, establishing a new state-of-the-art among efficient diffusion solvers. Code is available at https://github.com/Westlake-AGI-Lab/DyWeight

  </details>



- **Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting**  
  Tingxuan Huang, Haowei Zhu, Jun-hai Yong, Hao Pan, Bin Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11543v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic 3D scenes with photorealistic detail and strong temporal coherence remains a significant challenge. Existing Gaussian splatting approaches for dynamic scene modeling often rely on per-frame optimization, which can overfit to instantaneous states instead of capturing underlying motion dynamics. To address this, we present Mango-GS, a multi-frame, node-guided framework for high-fidelity 4D reconstruction. Mango-GS leverages a temporal Transformer to model motion dependencies within a short window of frames, producing temporally consistent deformations. For efficiency, temporal modeling is confined to a sparse set of control nodes. Each node is represented by a decoupled canonical position and a latent code, providing a stable semantic anchor for motion propagation and preventing correspondence drift under large motion. Our framework is trained end-to-end, enhanced by an input masking strategy and two multi-frame losses to improve robustness. Extensive experiments demonstrate that Mango-GS achieves state-of-the-art reconstruction quality and real-time rendering speed, enabling high-fidelity reconstruction and interactive rendering of dynamic scenes.

  </details>



- **Risk-Controllable Multi-View Diffusion for Driving Scenario Generation**  
  Hongyi Lin, Wenxiu Shi, Heye Huang, Dingyi Zhuang, Song Zhang, Yang Liu, Xiaobo Qu, Jinhua Zhao  
  _2026-03-12_ · https://arxiv.org/abs/2603.11534v1  
  <details><summary>Abstract</summary>

  Generating safety-critical driving scenarios is crucial for evaluating and improving autonomous driving systems, but long-tail risky situations are rarely observed in real-world data and difficult to specify through manual scenario design. Existing generative approaches typically treat risk as an after-the-fact label and struggle to maintain geometric consistency in multi-view driving scenes. We present RiskMV-DPO, a general and systematic pipeline for physically-informed, risk-controllable multi-view scenario generation. By integrating target risk levels with physically-grounded risk modeling, we autonomously synthesize diverse and high-stakes dynamic trajectories that serve as explicit geometric anchors for a diffusion-based video generator. To ensure spatial-temporal coherence and geometric fidelity, we introduce a geometry-appearance alignment module and a region-aware direct preference optimization (RA-DPO) strategy with motion-aware masking to focus learning on localized dynamic regions.Experiments on the nuScenes dataset show that RiskMV-DPO can freely generate a wide spectrum of diverse long-tail scenarios while maintaining state-of-the-art visual quality, improving 3D detection mAP from 18.17 to 30.50 and reducing FID to 15.70. Our work shifts the role of world models from passive environment prediction to proactive, risk-controllable synthesis, providing a scalable toolchain for the safety-oriented development of embodied intelligence.

  </details>



- **Are Video Reasoning Models Ready to Go Outside?**  
  Yangfan He, Changgyu Boo, Jaehong Yoon  
  _2026-03-11_ · https://arxiv.org/abs/2603.10652v1  
  <details><summary>Abstract</summary>

  In real-world deployment, vision-language models often encounter disturbances such as weather, occlusion, and camera motion. Under such conditions, their understanding and reasoning degrade substantially, revealing a gap between clean, controlled (i.e., unperturbed) evaluation settings and real-world robustness. To address this limitation, we propose ROVA, a novel training framework that improves robustness by modeling a robustness-aware consistency reward under spatio-temporal corruptions. ROVA introduces a difficulty-aware online training strategy that prioritizes informative samples based on the model's evolving capability. Specifically, it continuously re-estimates sample difficulty via self-reflective evaluation, enabling adaptive training with a robustness-aware consistency reward. We also introduce PVRBench, a new benchmark that injects real-world perturbations into embodied video datasets to assess both accuracy and reasoning quality under realistic disturbances. We evaluate ROVA and baselines on PVRBench, UrbanVideo, and VisBench, where open-source and proprietary models suffer up to 35% and 28% drops in accuracy and reasoning under realistic perturbations. ROVA effectively mitigates performance degradation, boosting relative accuracy by at least 24% and reasoning by over 9% compared with baseline models (QWen2.5/3-VL, InternVL2.5, Embodied-R). These gains transfer to clean standard benchmarks, yielding consistent improvements.

  </details>


