# Gaussian Splatting & 3DGS

_Updated: 2026-03-12 07:11 UTC_

Total papers shown: **21**


---

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
  _2026-03-11_ · https://arxiv.org/abs/2603.10446v1  
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



- **ProGS: Towards Progressive Coding for 3D Gaussian Splatting**  
  Zhiye Tang, Lingzhuo Liu, Shengjie Jiao, Qiudan Zhang, Junhui Hou, You Yang, Xu Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09703v1  
  <details><summary>Abstract</summary>

  With the emergence of 3D Gaussian Splatting (3DGS), numerous pioneering efforts have been made to address the effective compression issue of massive 3DGS data. 3DGS offers an efficient and scalable representation of 3D scenes by utilizing learnable 3D Gaussians, but the large size of the generated data has posed significant challenges for storage and transmission. Existing methods, however, have been limited by their inability to support progressive coding, a crucial feature in streaming applications with varying bandwidth. To tackle this limitation, this paper introduce a novel approach that organizes 3DGS data into an octree structure, enabling efficient progressive coding. The proposed ProGS is a streaming-friendly codec that facilitates progressive coding for 3D Gaussian splatting, and significantly improves both compression efficiency and visual fidelity. The proposed method incorporates mutual information enhancement mechanisms to mitigate structural redundancy, leveraging the relevance between nodes in the octree hierarchy. By adapting the octree structure and dynamically adjusting the anchor nodes, ProGS ensures scalable data compression without compromising the rendering quality. ProGS achieves a remarkable 45X reduction in file storage compared to the original 3DGS format, while simultaneously improving visual performance by over 10%. This demonstrates that ProGS can provide a robust solution for real-time applications with varying network conditions.

  </details>



- **VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM**  
  Anh Thuan Tran, Jana Kosecka  
  _2026-03-10_ · https://arxiv.org/abs/2603.09673v1  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

  </details>



- **DiffWind: Physics-Informed Differentiable Modeling of Wind-Driven Object Dynamics**  
  Yuanhang Lei, Boming Zhao, Zesong Yang, Xingxuan Li, Tao Cheng, Haocheng Peng, Ru Zhang, Yang Yang, Siyuan Huang, Yujun Shen, et al.  
  _2026-03-10_ · https://arxiv.org/abs/2603.09668v1  
  <details><summary>Abstract</summary>

  Modeling wind-driven object dynamics from video observations is highly challenging due to the invisibility and spatio-temporal variability of wind, as well as the complex deformations of objects. We present DiffWind, a physics-informed differentiable framework that unifies wind-object interaction modeling, video-based reconstruction, and forward simulation. Specifically, we represent wind as a grid-based physical field and objects as particle systems derived from 3D Gaussian Splatting, with their interaction modeled by the Material Point Method (MPM). To recover wind-driven object dynamics, we introduce a reconstruction framework that jointly optimizes the spatio-temporal wind force field and object motion through differentiable rendering and simulation. To ensure physical validity, we incorporate the Lattice Boltzmann Method (LBM) as a physics-informed constraint, enforcing compliance with fluid dynamics laws. Beyond reconstruction, our method naturally supports forward simulation under novel wind conditions and enables new applications such as wind retargeting. We further introduce WD-Objects, a dataset of synthetic and real-world wind-driven scenes. Extensive experiments demonstrate that our method significantly outperforms prior dynamic scene modeling approaches in both reconstruction accuracy and simulation fidelity, opening a new avenue for video-based wind-object interaction modeling.

  </details>



- **X-GS: An Extensible Open Framework Unifying 3DGS Architectures with Downstream Multimodal Models**  
  Yueen Ma, Irwin King  
  _2026-03-10_ · https://arxiv.org/abs/2603.09632v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods are isolated, focusing on specific domains such as online SLAM, semantic enrichment, or 3DGS for unposed images. In this paper, we introduce X-GS, an extensible open framework that unifies a broad range of techniques to enable real-time 3DGS-based online SLAM enriched with semantics, bridging the gap to downstream multimodal models. At the core of X-GS is a highly efficient pipeline called X-GS-Perceiver, capable of taking unposed RGB (or optionally RGB-D) video streams as input to co-optimize geometry and poses, and distill high-dimensional semantic features from vision foundation models into the 3D Gaussians. We achieve real-time performance through a novel online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The semantic 3D Gaussians can then be utilized by vision-language models within the X-GS-Thinker component, enabling downstream tasks such as object detection, zero-shot caption generation, and potentially embodied tasks. Experimental results on real-world datasets showcase the efficacy, efficiency, and newly unlocked multimodal capabilities of the X-GS framework.

  </details>



- **Physics-Driven 3D Gaussian Rendering for Zero-Shot MRI Super-Resolution**  
  Shuting Liu, Lei Zhang, Wei Huang, Zhao Zhang, Zizhou Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09621v1  
  <details><summary>Abstract</summary>

  High-resolution Magnetic Resonance Imaging (MRI) is vital for clinical diagnosis but limited by long acquisition times and motion artifacts. Super-resolution (SR) reconstructs low-resolution scans into high-resolution images, yet existing methods are mutually constrained: paired-data methods achieve efficiency only by relying on costly aligned datasets, while implicit neural representation approaches avoid such data needs at the expense of heavy computation. We propose a zero-shot MRI SR framework using explicit Gaussian representation to balance data requirements and efficiency. MRI-tailored Gaussian parameters embed tissue physical properties, reducing learnable parameters while preserving MR signal fidelity. A physics-grounded volume rendering strategy models MRI signal formation via normalized Gaussian aggregation. Additionally, a brick-based order-independent rasterization scheme enables highly parallel 3D computation, lowering training and inference costs. Experiments on two public MRI datasets show superior reconstruction quality and efficiency, demonstrating the method's potential for clinical MRI SR.

  </details>



- **DenoiseSplat: Feed-Forward Gaussian Splatting for Noisy 3D Scene Reconstruction**  
  Fuzhen Jiang, Zhuoran Li, Yinlin Zhang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09291v1  
  <details><summary>Abstract</summary>

  3D scene reconstruction and novel-view synthesis are fundamental for VR, robotics, and content creation. However, most NeRF and 3D Gaussian Splatting pipelines assume clean inputs and degrade under real noise and artifacts. We therefore propose DenoiseSplat, a feed-forward 3D Gaussian splatting method for noisy multi-view images. We build a large-scale, scene-consistent noisy--clean benchmark on RE10K by injecting Gaussian, Poisson, speckle, and salt-and-pepper noise with controlled intensities. With a lightweight MVSplat-style feed-forward backbone, we train end-to-end using only clean 2D renderings as supervision and no 3D ground truth. On noisy RE10K, DenoiseSplat outperforms vanilla MVSplat and a strong two-stage baseline (IDF + MVSplat) in PSNR/SSIM and LPIPS across noise types and levels.

  </details>



- **Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists**  
  Jiaqi Liu, Zhizhong Han  
  _2026-03-10_ · https://arxiv.org/abs/2603.09277v1  
  <details><summary>Abstract</summary>

  3D Gaussian splatting (3DGS) has become a vital tool for learning a radiance field from multiple posed images. Although 3DGS shows great advantages over NeRF in terms of rendering quality and efficiency, it remains a research challenge to further improve the efficiency of learning 3D Gaussians. To overcome this challenge, we propose novel training strategies and losses to shorten each Gaussian list used to render a pixel, which speeds up the splatting by involving fewer Gaussians along a ray. Specifically, we shrink the size of each Gaussian by resetting their scales regularly, encouraging smaller Gaussians to cover fewer nearby pixels, which shortens the Gaussian lists of pixels. Additionally, we introduce an entropy constraint on the alpha blending procedure to sharpen the weight distribution of Gaussians along each ray, which drives dominant weights larger while making minor weights smaller. As a result, each Gaussian becomes more focused on the pixels where it is dominant, which reduces its impact on nearby pixels, leading to even shorter Gaussian lists. Eventually, we integrate our method into a rendering resolution scheduler which further improves efficiency through progressive resolution increase. We evaluate our method by comparing it with state-of-the-art methods on widely used benchmarks. Our results show significant advantages over others in efficiency without sacrificing rendering quality.

  </details>



- **GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware Vision-Language-Action Models**  
  Md Selim Sarowar, Omer Tariq, Sungho Kim  
  _2026-03-10_ · https://arxiv.org/abs/2603.09079v1  
  <details><summary>Abstract</summary>

  VLA models encode visual observations as 2D patch tokens with no intrinsic geometric structure. We introduce GST-VLA with two contributions. First, the Gaussian Spatial Tokenizer (GST) converts frozen dense depth and frozen semantic patch features into $N_g{=}128$ anisotropic 3D Gaussian primitives, each parameterized by a metric residual mean $μ\in \mathbb{R}^3$, log-scale covariance $\log σ\in \mathbb{R}^3$, and learned opacity $α\in (0,1)$. The covariance eigenstructure encodes local surface orientation, and opacity provides per-primitive geometric confidence, both inaccessible from scalar depth. Spatial attention pooling with learned queries concentrates the fixed token budget on geometrically salient regions rather than distributing uniformly. Second, 3D Depth-Aware Chain-of-Thought (DA-CoT) reasoning supervises four structured intermediate spatial thoughts, covering 3D object grounding, grasp affordance contact geometry, pairwise metric distances, and coarse SE(3) waypoints, as explicit generation targets in the training loss. A cross-attention sublayer at every VLM transformer block provides direct access to the raw 256-primitive Gaussian field during DA-CoT generation. A 300M-parameter flow-matching action expert with mixture-of-experts feedforward sublayers decodes 7-DoF delta action chunks via conditional ODE integration, conditioned on both VLM hidden states and DA-CoT outputs through dual cross-attention. Trained with composite $\mathcal{L}_\mathrm{flow} + \mathcal{L}_\mathrm{CoT} + \mathcal{L}_\mathrm{depth}$ across three progressive stages, GST-VLA achieves 96.4% on LIBERO (+2.0%), and 80.2% on SimplerEnv (+5.4%). Ablations isolate the contribution of each GST component, each DA-CoT thought, and each training stage, confirming independent and synergistic gains concentrated on precision demanding tasks.

  </details>



- **SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training**  
  Jingxing Li, Yongjae Leeand, Deliang Fan  
  _2026-03-09_ · https://arxiv.org/abs/2603.08997v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) achieves real-time novel-view synthesis by optimizing millions of anisotropic Gaussians, yet its training remains expensive, with the backward pass dominating runtime in the post-densification refinement phase. We observe substantial update redundancy in this phase: many sampled views have near-plateaued losses and provide diminishing gradient benefits, but standard training still runs full backpropagation. We propose SkipGS with a novel view-adaptive backward gating mechanism for efficient post-densification training. SkipGS always performs the forward pass to update per-view loss statistics, and selectively skips backward passes when the sampled view's loss is consistent with its recent per-view baseline, while enforcing a minimum backward budget for stable optimization. On Mip-NeRF 360, compared to 3DGS, SkipGS reduces end-to-end training time by 23.1%, driven by a 42.0% reduction in post-densification time, with comparable reconstruction quality. Because it only changes when to backpropagate -- without modifying the renderer, representation, or loss -- SkipGS is plug-and-play and compatible with other complementary efficiency strategies for additive speedups.

  </details>



- **SurgCalib: Gaussian Splatting-Based Hand-Eye Calibration for Robot-Assisted Minimally Invasive Surgery**  
  Zijian Wu, Shuojue Yang, Yu Chung Lee, Eitan Prisman, Yueming Jin, Septimiu E. Salcudean  
  _2026-03-09_ · https://arxiv.org/abs/2603.08983v1  
  <details><summary>Abstract</summary>

  We present a Gaussian Splatting-based framework for hand-eye calibration of the da Vinci surgical robot. In a vision-guided robotic system, accurate estimation of the rigid transformation between the robot base and the camera frame is essential for reliable closed-loop control. For cable-driven surgical robots, this task faces unique challenges. The encoders of surgical instruments often produce inaccurate proprioceptive measurements due to cable stretch and backlash. Conventional hand-eye calibration approaches typically rely on known fiducial patterns and solve the AX = XB formulation. While effective, introducing additional markers into the operating room (OR) environment can violate sterility protocols and disrupt surgical workflows. In this study, we propose SurgCalib, an automatic, markerless framework that has the potential to be used in the OR. SurgCalib first initializes the pose of the surgical instrument using raw kinematic measurements and subsequently refines this pose through a two-phase optimization procedure under the RCM constraint within a Gaussian Splatting-based differentiable rendering pipeline. We evaluate the proposed method on the public dVRK benchmark, SurgPose. The results demonstrate average 2D tool-tip reprojection errors of 12.24 px (2.06 mm) and 11.33 px (1.9 mm), and 3D tool-tip Euclidean distance errors of 5.98 mm and 4.75 mm, for the left and right instruments, respectively.

  </details>



- **Where, What, Why: Toward Explainable 3D-GS Watermarking**  
  Mingshu Cai, Jiajun Li, Osamu Yoshie, Yuya Ieiri, Yixuan Li  
  _2026-03-09_ · https://arxiv.org/abs/2603.08809v1  
  <details><summary>Abstract</summary>

  As 3D Gaussian Splatting becomes the de facto representation for interactive 3D assets, robust yet imperceptible watermarking is critical. We present a representation-native framework that separates where to write from how to preserve quality. A Trio-Experts module operates directly on Gaussian primitives to derive priors for carrier selection, while a Safety and Budget Aware Gate (SBAG) allocates Gaussians to watermark carriers, optimized for bit resilience under perturbation and bitrate budgets, and to visual compensators that are insulated from watermark loss. To maintain fidelity, we introduce a channel-wise group mask that controls gradient propagation for carriers and compensators, thereby limiting Gaussian parameter updates, repairing local artifacts, and preserving high-frequency details without increasing runtime. Our design yields view-consistent watermark persistence and strong robustness against common image distortions such as compression and noise, while achieving a favorable robustness-quality trade-off compared with prior methods. In addition, decoupled finetuning provides per-Gaussian attributions that reveal where the message is carried and why those carriers are selected, enabling auditable explainability. Compared with state-of-the-art methods, our approach achieves a PSNR improvement of +0.83 dB and a bit-accuracy gain of +1.24%.

  </details>



- **ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting**  
  Jordi Muñoz Vicente  
  _2026-03-09_ · https://arxiv.org/abs/2603.08661v1  
  <details><summary>Abstract</summary>

  Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.

  </details>



- **Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction**  
  Zhe Yang, Guoqiang Zhao, Sheng Wu, Kai Luo, Kailun Yang  
  _2026-03-09_ · https://arxiv.org/abs/2603.08503v1  
  <details><summary>Abstract</summary>

  Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at https://github.com/1170632760/Spherical-GOF.

  </details>


