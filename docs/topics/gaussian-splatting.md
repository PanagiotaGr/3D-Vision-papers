# Gaussian Splatting & 3DGS

_Updated: 2026-03-13 07:10 UTC_

Total papers shown: **13**


---

- **AstroSplat: Physics-Based Gaussian Splatting for Rendering and Reconstruction of Small Celestial Bodies**  
  Jennifer Nolan, Travis Driver, John Christian  
  _2026-03-12_ · https://arxiv.org/abs/2603.11969v1  
  <details><summary>Abstract</summary>

  Image-based surface reconstruction and characterization are crucial for missions to small celestial bodies (e.g., asteroids), as it informs mission planning, navigation, and scientific analysis. Recent advances in Gaussian splatting enable high-fidelity neural scene representations but typically rely on a spherical harmonic intensity parameterization that is strictly appearance-based and does not explicitly model material properties or light-surface interactions. We introduce AstroSplat, a physics-based Gaussian splatting framework that integrates planetary reflectance models to improve the autonomous reconstruction and photometric characterization of small-body surfaces from in-situ imagery. The proposed framework is validated on real imagery taken by NASA's Dawn mission, where we demonstrate superior rendering performance and surface reconstruction accuracy compared to the typical spherical harmonic parameterization.

  </details>



- **Mango-GS: Enhancing Spatio-Temporal Consistency in Dynamic Scenes Reconstruction using Multi-Frame Node-Guided 4D Gaussian Splatting**  
  Tingxuan Huang, Haowei Zhu, Jun-hai Yong, Hao Pan, Bin Wang  
  _2026-03-12_ · https://arxiv.org/abs/2603.11543v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic 3D scenes with photorealistic detail and strong temporal coherence remains a significant challenge. Existing Gaussian splatting approaches for dynamic scene modeling often rely on per-frame optimization, which can overfit to instantaneous states instead of capturing underlying motion dynamics. To address this, we present Mango-GS, a multi-frame, node-guided framework for high-fidelity 4D reconstruction. Mango-GS leverages a temporal Transformer to model motion dependencies within a short window of frames, producing temporally consistent deformations. For efficiency, temporal modeling is confined to a sparse set of control nodes. Each node is represented by a decoupled canonical position and a latent code, providing a stable semantic anchor for motion propagation and preventing correspondence drift under large motion. Our framework is trained end-to-end, enhanced by an input masking strategy and two multi-frame losses to improve robustness. Extensive experiments demonstrate that Mango-GS achieves state-of-the-art reconstruction quality and real-time rendering speed, enabling high-fidelity reconstruction and interactive rendering of dynamic scenes.

  </details>



- **Mobile-GS: Real-time Gaussian Splatting for Mobile Devices**  
  Xiaobiao Du, Yida Wang, Kun Zhan, Xin Yu  
  _2026-03-12_ · https://arxiv.org/abs/2603.11531v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful representation for high-quality rendering across a wide range of applications.However, its high computational demands and large storage costs pose significant challenges for deployment on mobile devices. In this work, we propose a mobile-tailored real-time Gaussian Splatting method, dubbed Mobile-GS, enabling efficient inference of Gaussian Splatting on edge devices. Specifically, we first identify alpha blending as the primary computational bottleneck, since it relies on the time-consuming Gaussian depth sorting process. To solve this issue, we propose a depth-aware order-independent rendering scheme that eliminates the need for sorting, thereby substantially accelerating rendering. Although this order-independent rendering improves rendering speed, it may introduce transparency artifacts in regions with overlapping geometry due to the scarcity of rendering order. To address this problem, we propose a neural view-dependent enhancement strategy, enabling more accurate modeling of view-dependent effects conditioned on viewing direction, 3D Gaussian geometry, and appearance attributes. In this way, Mobile-GS can achieve both high-quality and real-time rendering. Furthermore, to facilitate deployment on memory-constrained mobile platforms, we also introduce first-order spherical harmonics distillation, a neural vector quantization technique, and a contribution-based pruning strategy to reduce the number of Gaussian primitives and compress the 3D Gaussian representation with the assistance of neural networks. Extensive experiments demonstrate that our proposed Mobile-GS achieves real-time rendering and compact model size while preserving high visual quality, making it well-suited for mobile applications.

  </details>



- **InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction**  
  Dingqiang Ye, Jiacong Xu, Jianglu Ping, Yuxiang Guo, Chao Fan, Vishal M. Patel  
  _2026-03-11_ · https://arxiv.org/abs/2603.11298v1  
  <details><summary>Abstract</summary>

  High dynamic range (HDR) novel view synthesis (NVS) aims to reconstruct HDR scenes from multi-exposure low dynamic range (LDR) images. Existing HDR pipelines heavily rely on known camera poses, well-initialized dense point clouds, and time-consuming per-scene optimization. Current feed-forward alternatives overlook the HDR problem by assuming exposure-invariant appearance. To bridge this gap, we propose InstantHDR, a feed-forward network that reconstructs 3D HDR scenes from uncalibrated multi-exposure LDR collections in a single forward pass. Specifically, we design a geometry-guided appearance modeling for multi-exposure fusion, and a meta-network for generalizable scene-specific tone mapping. Due to the lack of HDR scene data, we build a pre-training dataset, called HDR-Pretrain, for generalizable feed-forward HDR models, featuring 168 Blender-rendered scenes, diverse lighting types, and multiple camera response functions. Comprehensive experiments show that our InstantHDR delivers comparable synthesis performance to the state-of-the-art optimization-based HDR methods while enjoying $\sim700\times$ and $\sim20\times$ reconstruction speed improvement with our single-forward and post-optimization settings. All code, models, and datasets will be released after the review process.

  </details>



- **Senna-2: Aligning VLM and End-to-End Driving Policy for Consistent Decision Making and Planning**  
  Yuehao Song, Shaoyu Chen, Hao Gao, Yifan Zhu, Weixiang Yue, Jialv Zou, Bo Jiang, Zihao Lu, Yu Wang, Qian Zhang, et al.  
  _2026-03-11_ · https://arxiv.org/abs/2603.11219v1  
  <details><summary>Abstract</summary>

  Vision-language models (VLMs) enhance the planning capability of end-to-end (E2E) driving policy by leveraging high-level semantic reasoning. However, existing approaches often overlook the dual-system consistency between VLM's high-level decision and E2E's low-level planning. As a result, the generated trajectories may misalign with the intended driving decisions, leading to weakened top-down guidance and decision-following ability of the system. To address this issue, we propose Senna-2, an advanced VLM-E2E driving policy that explicitly aligns the two systems for consistent decision-making and planning. Our method follows a consistency-oriented three-stage training paradigm. In the first stage, we conduct driving pre-training to achieve preliminary decision-making and planning, with a decision adapter transmitting VLM decisions to E2E policy in the form of implicit embeddings. In the second stage, we align the VLM and the E2E policy in an open-loop setting. In the third stage, we perform closed-loop alignment via bottom-up Hierarchical Reinforcement Learning in 3DGS environments to reinforce the safety and efficiency. Extensive experiments demonstrate that Senna-2 achieves superior dual-system consistency (19.3% F1 score improvement) and significantly enhances driving safety in both open-loop (5.7% FDE reduction) and closed-loop settings (30.6% AF-CR reduction).

  </details>



- **S2D: Sparse to Dense Lifting for 3D Reconstruction with Minimal Inputs**  
  Yuzhou Ji, Qijian Tian, He Zhu, Xiaoqi Jiang, Guangzhi Cao, Lizhuang Ma, Yuan Xie, Xin Tan  
  _2026-03-11_ · https://arxiv.org/abs/2603.10893v1  
  <details><summary>Abstract</summary>

  Explicit 3D representations have already become an essential medium for 3D simulation and understanding. However, the most commonly used point cloud and 3D Gaussian Splatting (3DGS) each suffer from non-photorealistic rendering and significant degradation under sparse inputs. In this paper, we introduce Sparse to Dense lifting (S2D), a novel pipeline that bridges the two representations and achieves high-quality 3DGS reconstruction with minimal inputs. Specifically, the S2D lifting is two-fold. We first present an efficient one-step diffusion model that lifts sparse point cloud for high-fidelity image artifact fixing. Meanwhile, to reconstruct 3D consistent scenes, we also design a corresponding reconstruction strategy with random sample drop and weighted gradient for robust model fitting from sparse input views to dense novel views. Extensive experiments show that S2D achieves the best consistency in generating novel view guidance and first-tier sparse view reconstruction quality under different input sparsity. By reconstructing stable scenes with the least possible captures among existing methods, S2D enables minimal input requirements for 3DGS applications.

  </details>



- **PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction**  
  Yufei Han, Chu Zhou, Youwei Lyu, Qi Chen, Si Li, Boxin Shi, Yunpeng Jia, Heng Guo, Zhanyu Ma  
  _2026-03-11_ · https://arxiv.org/abs/2603.10801v1  
  <details><summary>Abstract</summary>

  Accurate reconstruction of reflective surfaces remains a fundamental challenge in computer vision, with broad applications in real-time virtual reality and digital content creation. Although 3D Gaussian Splatting (3DGS) enables efficient novel-view rendering with explicit representations, its performance on reflective surfaces still lags behind implicit neural methods, especially in recovering fine geometry and surface normals. To address this gap, we propose PolGS++, a physically-guided polarimetric Gaussian Splatting framework for fast reflective surface reconstruction. Specifically, we integrate a polarized BRDF (pBRDF) model into 3DGS to explicitly decouple diffuse and specular components, providing physically grounded reflectance modeling and stronger geometric cues for reflective surface recovery. Furthermore, we introduce a depth-guided visibility mask acquisition mechanism that enables angle-of-polarization (AoP)-based tangent-space consistency constraints in Gaussian Splatting without costly ray-tracing intersections. This physically guided design improves reconstruction quality and efficiency, requiring only about 10 minutes of training. Extensive experiments on both synthetic and real-world datasets validate the effectiveness of our method.

  </details>



- **Splat2Real: Novel-view Scaling for Physical AI with 3D Gaussian Splatting**  
  Hansol Lim, Jongseong Brad Choi  
  _2026-03-11_ · https://arxiv.org/abs/2603.10638v1  
  <details><summary>Abstract</summary>

  Physical AI faces viewpoint shift between training and deployment, and novel-view robustness is essential for monocular RGB-to-3D perception. We cast Real2Render2Real monocular depth pretraining as imitation-learning-style supervision from a digital twin oracle: a student depth network imitates expert metric depth/visibility rendered from a scene mesh, while 3DGS supplies scalable novel-view observations. We present Splat2Real, centered on novel-view scaling: performance depends more on which views are added than on raw view count. We introduce CN-Coverage, a coverage+novelty curriculum that greedily selects views by geometry gain and an extrapolation penalty, plus a quality-aware guardrail fallback for low-reliability teachers. Across 20 TUM RGB-D sequences with step-matched budgets (N=0 to 2000 additional rendered views, with N unique <= 500 and resampling for larger budgets), naive scaling is unstable; CN-Coverage mitigates worst-case regressions relative to Robot/Coverage policies, and GOL-Gated CN-Coverage provides the strongest medium-high-budget stability with the lowest high-novelty tail error. Downstream control-proxy results versus N provides embodied-relevance evidence by shifting safety/progress trade-offs under viewpoint shift.

  </details>



- **P-GSVC: Layered Progressive 2D Gaussian Splatting for Scalable Image and Video**  
  Longan Wang, Yuang Shi, Wei Tsang Ooi  
  _2026-03-11_ · https://arxiv.org/abs/2603.10551v1  
  <details><summary>Abstract</summary>

  Gaussian splatting has emerged as a competitive explicit representation for image and video reconstruction. In this work, we present P-GSVC, the first layered progressive 2D Gaussian splatting framework that provides a unified solution for scalable Gaussian representation in both images and videos. P-GSVC organizes 2D Gaussian splats into a base layer and successive enhancement layers, enabling coarse-to-fine reconstructions. To effectively optimize this layered representation, we propose a joint training strategy that simultaneously updates Gaussians across layers, aligning their optimization trajectories to ensure inter-layer compatibility and a stable progressive reconstruction. P-GSVC supports scalability in terms of both quality and resolution. Our experiments show that the joint training strategy can gain up to 1.9 dB improvement in PSNR for video and 2.6 dB improvement in PSNR for image when compared to methods that perform sequential layer-wise training. Project page: https://longanwang-cs.github.io/PGSVC-webpage/

  </details>



- **SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning**  
  Jianhe Low, Alexandre Symeonidis-Herzig, Maksym Ivashechkin, Ozge Mercanoglu Sincan, Richard Bowden  
  _2026-03-11_ · https://arxiv.org/abs/2603.10446v2  
  <details><summary>Abstract</summary>

  Generating natural and linguistically accurate sign language avatars remains a formidable challenge. Current Sign Language Production (SLP) frameworks face a stark trade-off: direct text-to-pose models suffer from regression-to-the-mean effects, while dictionary-retrieval methods produce robotic, disjointed transitions. To resolve this, we propose a novel training paradigm that leverages sparse keyframes to capture the true underlying kinematic distribution of human signing. By predicting dense motion from these discrete anchors, our approach mitigates regression-to-the-mean while ensuring fluid articulation. To realize this paradigm at scale, we first introduce FAST, an ultra-efficient sign segmentation model that automatically mines precise temporal boundaries. We then present SignSparK, a large-scale Conditional Flow Matching (CFM) framework that utilizes these extracted anchors to synthesize 3D signing sequences in SMPL-X and MANO spaces. This keyframe-driven formulation also uniquely unlocks Keyframe-to-Pose (KF2P) generation, making precise spatiotemporal editing of signing sequences possible. Furthermore, our adopted reconstruction-based CFM objective also enables high-fidelity synthesis in fewer than ten sampling steps; this allows SignSparK to scale across four distinct sign languages, establishing the largest multilingual SLP framework to date. Finally, by integrating 3D Gaussian Splatting for photorealistic rendering, we demonstrate through extensive evaluation that SignSparK establishes a new state-of-the-art across diverse SLP tasks and multilingual benchmarks.

  </details>



- **4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video**  
  Jin Lyu, Liang An, Pujin Cheng, Yebin Liu, Xiaoying Tang  
  _2026-03-10_ · https://arxiv.org/abs/2603.10125v1  
  <details><summary>Abstract</summary>

  4D reconstruction of equine family (e.g. horses) from monocular video is important for animal welfare. Previous mainstream 4D animal reconstruction methods require joint optimization of motion and appearance over a whole video, which is time-consuming and sensitive to incomplete observation. In this work, we propose a novel framework called 4DEquine by disentangling the 4D reconstruction problem into two sub-problems: dynamic motion reconstruction and static appearance reconstruction. For motion, we introduce a simple yet effective spatio-temporal transformer with a post-optimization stage to regress smooth and pixel-aligned pose and shape sequences from video. For appearance, we design a novel feed-forward network that reconstructs a high-fidelity, animatable 3D Gaussian avatar from as few as a single image. To assist training, we create a large-scale synthetic motion dataset, VarenPoser, which features high-quality surface motions and diverse camera trajectories, as well as a synthetic appearance dataset, VarenTex, comprising realistic multi-view images generated through multi-view diffusion. While training only on synthetic datasets, 4DEquine achieves state-of-the-art performance on real-world APT36K and AiM datasets, demonstrating the superiority of 4DEquine and our new datasets for both geometry and appearance reconstruction. Comprehensive ablation studies validate the effectiveness of both the motion and appearance reconstruction network. Project page: https://luoxue-star.github.io/4DEquine_Project_Page/.

  </details>



- **ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare**  
  Freeman Cheng, Botao Ye, Xueting Li, Junqi You, Fangneng Zhan, Ming-Hsuan Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09968v1  
  <details><summary>Abstract</summary>

  Online novel view synthesis remains challenging, requiring robust scene reconstruction from sequential, often unposed, observations. We present ReCoSplat, an autoregressive feed-forward Gaussian Splatting model supporting posed or unposed inputs, with or without camera intrinsics. While assembling local Gaussians using camera poses scales better than canonical-space prediction, it creates a dilemma during training: using ground-truth poses ensures stability but causes a distribution mismatch when predicted poses are used at inference. To address this, we introduce a Render-and-Compare (ReCo) module. ReCo renders the current reconstruction from the predicted viewpoint and compares it with the incoming observation, providing a stable conditioning signal that compensates for pose errors. To support long sequences, we propose a hybrid KV cache compression strategy combining early-layer truncation with chunk-level selective retention, reducing the KV cache size by over 90% for 100+ frames. ReCoSplat achieves state-of-the-art performance across different input settings on both in- and out-of-distribution benchmarks. Code and pretrained models will be released. Our project page is at https://freemancheng.com/ReCoSplat .

  </details>



- **GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System**  
  Zhiye Tang, Qiudan Zhang, Lei Zhang, Junhui Hou, You Yang, Xu Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09718v1  
  <details><summary>Abstract</summary>

  Recently, the 3D Gaussian splatting (3DGS) technique for real-time radiance field rendering has revolutionized the field of volumetric scene representation, providing users with an immersive experience. But in return, it also poses a large amount of data volume, which is extremely bandwidth-intensive. Cutting-edge researchers have tried to introduce different approaches and construct multiple variants for 3DGS to obtain a more compact scene representation, but it is still challenging for real-time distribution. In this paper, we propose GSStream, a novel volumetric scene streaming system to support 3DGS data format. Specifically, GSStream integrates a collaborative viewport prediction module to better predict users' future behaviors by learning collaborative priors and historical priors from multiple users and users' viewport sequences and a deep reinforcement learning (DRL)-based bitrate adaptation module to tackle the state and action space variability challenge of the bitrate adaptation problem, achieving efficient volumetric scene delivery. Besides, we first build a user viewport trajectory dataset for volumetric scenes to support the training and streaming simulation. Extensive experiments prove that our proposed GSStream system outperforms existing representative volumetric scene streaming systems in visual quality and network usage. Demo video: https://youtu.be/3WEe8PN8yvA.

  </details>


