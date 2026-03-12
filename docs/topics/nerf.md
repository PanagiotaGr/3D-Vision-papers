# NeRF & Neural Radiance Fields

_Updated: 2026-03-12 07:11 UTC_

Total papers shown: **11**


---

- **Neural Field Thermal Tomography: A Differentiable Physics Framework for Non-Destructive Evaluation**  
  Tao Zhong, Yixun Hu, Dongzhe Zheng, Aditya Sood, Christine Allen-Blanchette  
  _2026-03-11_ · https://arxiv.org/abs/2603.11045v1  
  <details><summary>Abstract</summary>

  We propose Neural Field Thermal Tomography (NeFTY), a differentiable physics framework for the quantitative 3D reconstruction of material properties from transient surface temperature measurements. While traditional thermography relies on pixel-wise 1D approximations that neglect lateral diffusion, and soft-constrained Physics-Informed Neural Networks (PINNs) often fail in transient diffusion scenarios due to gradient stiffness, NeFTY parameterizes the 3D diffusivity field as a continuous neural field optimized through a rigorous numerical solver. By leveraging a differentiable physics solver, our approach enforces thermodynamic laws as hard constraints while maintaining the memory efficiency required for high-resolution 3D tomography. Our discretize-then-optimize paradigm effectively mitigates the spectral bias and ill-posedness inherent in inverse heat conduction, enabling the recovery of subsurface defects at arbitrary scales. Experimental validation on synthetic data demonstrates that NeFTY significantly improves the accuracy of subsurface defect localization over baselines. Additional details at https://cab-lab-princeton.github.io/nefty/

  </details>



- **ReCoSplat: Autoregressive Feed-Forward Gaussian Splatting Using Render-and-Compare**  
  Freeman Cheng, Botao Ye, Xueting Li, Junqi You, Fangneng Zhan, Ming-Hsuan Yang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09968v1  
  <details><summary>Abstract</summary>

  Online novel view synthesis remains challenging, requiring robust scene reconstruction from sequential, often unposed, observations. We present ReCoSplat, an autoregressive feed-forward Gaussian Splatting model supporting posed or unposed inputs, with or without camera intrinsics. While assembling local Gaussians using camera poses scales better than canonical-space prediction, it creates a dilemma during training: using ground-truth poses ensures stability but causes a distribution mismatch when predicted poses are used at inference. To address this, we introduce a Render-and-Compare (ReCo) module. ReCo renders the current reconstruction from the predicted viewpoint and compares it with the incoming observation, providing a stable conditioning signal that compensates for pose errors. To support long sequences, we propose a hybrid KV cache compression strategy combining early-layer truncation with chunk-level selective retention, reducing the KV cache size by over 90% for 100+ frames. ReCoSplat achieves state-of-the-art performance across different input settings on both in- and out-of-distribution benchmarks. Code and pretrained models will be released. Our project page is at https://freemancheng.com/ReCoSplat .

  </details>



- **ConfCtrl: Enabling Precise Camera Control in Video Diffusion via Confidence-Aware Interpolation**  
  Liudi Yang, George Eskandar, Fengyi Shen, Mohammad Altillawi, Yang Bai, Chi Zhang, Ziyuan Liu, Abhinav Valada  
  _2026-03-10_ · https://arxiv.org/abs/2603.09819v1  
  <details><summary>Abstract</summary>

  We address the challenge of novel view synthesis from only two input images under large viewpoint changes. Existing regression-based methods lack the capacity to reconstruct unseen regions, while camera-guided diffusion models often deviate from intended trajectories due to noisy point cloud projections or insufficient conditioning from camera poses. To address these issues, we propose ConfCtrl, a confidence-aware video interpolation framework that enables diffusion models to follow prescribed camera poses while completing unseen regions. ConfCtrl initializes the diffusion process by combining a confidence-weighted projected point cloud latent with noise as the conditioning input. It then applies a Kalman-inspired predict-update mechanism, treating the projected point cloud as a noisy measurement and using learned residual corrections to balance pose-driven predictions with noisy geometric observations. This allows the model to rely on reliable projections while down-weighting uncertain regions, yielding stable, geometry-aware generation. Experiments on multiple datasets show that ConfCtrl produces geometrically consistent and visually plausible novel views, effectively reconstructing occluded regions under large viewpoint changes.

  </details>



- **GSStream: 3D Gaussian Splatting based Volumetric Scene Streaming System**  
  Zhiye Tang, Qiudan Zhang, Lei Zhang, Junhui Hou, You Yang, Xu Wang  
  _2026-03-10_ · https://arxiv.org/abs/2603.09718v1  
  <details><summary>Abstract</summary>

  Recently, the 3D Gaussian splatting (3DGS) technique for real-time radiance field rendering has revolutionized the field of volumetric scene representation, providing users with an immersive experience. But in return, it also poses a large amount of data volume, which is extremely bandwidth-intensive. Cutting-edge researchers have tried to introduce different approaches and construct multiple variants for 3DGS to obtain a more compact scene representation, but it is still challenging for real-time distribution. In this paper, we propose GSStream, a novel volumetric scene streaming system to support 3DGS data format. Specifically, GSStream integrates a collaborative viewport prediction module to better predict users' future behaviors by learning collaborative priors and historical priors from multiple users and users' viewport sequences and a deep reinforcement learning (DRL)-based bitrate adaptation module to tackle the state and action space variability challenge of the bitrate adaptation problem, achieving efficient volumetric scene delivery. Besides, we first build a user viewport trajectory dataset for volumetric scenes to support the training and streaming simulation. Extensive experiments prove that our proposed GSStream system outperforms existing representative volumetric scene streaming systems in visual quality and network usage. Demo video: https://youtu.be/3WEe8PN8yvA.

  </details>



- **VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM**  
  Anh Thuan Tran, Jana Kosecka  
  _2026-03-10_ · https://arxiv.org/abs/2603.09673v1  
  <details><summary>Abstract</summary>

  Simultaneous Localization and Mapping (SLAM) with 3D Gaussian Splatting (3DGS) enables fast, differentiable rendering and high-fidelity reconstruction across diverse real-world scenes. However, existing 3DGS-SLAM approaches handle measurement reliability implicitly, making pose estimation and global alignment susceptible to drift in low-texture regions, transparent surfaces, or areas with complex reflectance properties. To this end, we introduce VarSplat, an uncertainty-aware 3DGS-SLAM system that explicitly learns per-splat appearance variance. By using the law of total variance with alpha compositing, we then render differentiable per-pixel uncertainty map via efficient, single-pass rasterization. This map guides tracking, submap registration, and loop detection toward focusing on reliable regions and contributes to more stable optimization. Experimental results on Replica (synthetic) and TUM-RGBD, ScanNet, and ScanNet++ (real-world) show that VarSplat improves robustness and achieves competitive or superior tracking, mapping, and novel view synthesis rendering compared to existing studies for dense RGB-D SLAM.

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



- **SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training**  
  Jingxing Li, Yongjae Leeand, Deliang Fan  
  _2026-03-09_ · https://arxiv.org/abs/2603.08997v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) achieves real-time novel-view synthesis by optimizing millions of anisotropic Gaussians, yet its training remains expensive, with the backward pass dominating runtime in the post-densification refinement phase. We observe substantial update redundancy in this phase: many sampled views have near-plateaued losses and provide diminishing gradient benefits, but standard training still runs full backpropagation. We propose SkipGS with a novel view-adaptive backward gating mechanism for efficient post-densification training. SkipGS always performs the forward pass to update per-view loss statistics, and selectively skips backward passes when the sampled view's loss is consistent with its recent per-view baseline, while enforcing a minimum backward budget for stable optimization. On Mip-NeRF 360, compared to 3DGS, SkipGS reduces end-to-end training time by 23.1%, driven by a 42.0% reduction in post-densification time, with comparable reconstruction quality. Because it only changes when to backpropagate -- without modifying the renderer, representation, or loss -- SkipGS is plug-and-play and compatible with other complementary efficiency strategies for additive speedups.

  </details>



- **ImprovedGS+: A High-Performance C++/CUDA Re-Implementation Strategy for 3D Gaussian Splatting**  
  Jordi Muñoz Vicente  
  _2026-03-09_ · https://arxiv.org/abs/2603.08661v1  
  <details><summary>Abstract</summary>

  Recent advancements in 3D Gaussian Splatting (3DGS) have shifted the focus toward balancing reconstruction fidelity with computational efficiency. In this work, we propose ImprovedGS+, a high-performance, low-level reinvention of the ImprovedGS strategy, implemented natively within the LichtFeld-Studio framework. By transitioning from high-level Python logic to hardware-optimized C++/CUDA kernels, we achieve a significant reduction in host-device synchronization and training latency. Our implementation introduces a Long-Axis-Split (LAS) CUDA kernel, custom Laplacian-based importance kernels with Non-Maximum Suppression (NMS) for edge scores, and an adaptive Exponential Scale Scheduler. Experimental results on the Mip-NeRF360 dataset demonstrate that ImprovedGS+ establishes a new Pareto-optimal front for scene reconstruction. Our 1M-budget variant outperforms the state-of-the-art MCMC baseline by achieving a 26.8% reduction in training time (saving 17 minutes per session) and utilizing 13.3% fewer Gaussians while maintaining superior visual quality. Furthermore, our full variant demonstrates a 1.28 dB PSNR increase over the ADC baseline with a 38.4% reduction in parametric complexity. These results validate ImprovedGS+ as a scalable, high-speed solution that upholds the core pillars of Speed, Quality, and Usability within the LichtFeld-Studio ecosystem.

  </details>


