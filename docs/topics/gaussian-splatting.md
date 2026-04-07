# Gaussian Splatting & 3DGS

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **9**


---

- **Free-Range Gaussians: Non-Grid-Aligned Generative 3D Gaussian Reconstruction**  
  Ahan Shabanov, Peter Hedman, Ethan Weber, Zhengqin Li, Denis Rozumny, Gael Le Lan, Naina Dhingra, Lei Luo, Andrea Vedaldi, Christian Richardt, et al.  
  _2026-04-06_ · https://arxiv.org/abs/2604.04874v1  
  <details><summary>Abstract</summary>

  We present Free-Range Gaussians, a multi-view reconstruction method that predicts non-pixel, non-voxel-aligned 3D Gaussians from as few as four images. This is done through flow matching over Gaussian parameters. Our generative formulation of reconstruction allows the model to be supervised with non-grid-aligned 3D data, and enables it to synthesize plausible content in unobserved regions. Thus, it improves on prior methods that produce highly redundant grid-aligned Gaussians, and suffer from holes or blurry conditional means in unobserved regions. To handle the number of Gaussians needed for high-quality results, we introduce a hierarchical patching scheme to group spatially related Gaussians into joint transformer tokens, halving the sequence length while preserving structure. We further propose a timestep-weighted rendering loss during training, and photometric gradient guidance and classifier-free guidance at inference to improve fidelity. Experiments on Objaverse and Google Scanned Objects show consistent improvements over pixel and voxel-aligned methods while using significantly fewer Gaussians, with large gains when input views leave parts of the object unobserved.

  </details>



- **AvatarPointillist: AutoRegressive 4D Gaussian Avatarization**  
  Hongyu Liu, Xuan Wang, Yating Wang, Zijian Wu, Ziyu Wan, Yue Ma, Runtao Liu, Boyao Zhou, Yujun Shen, Qifeng Chen  
  _2026-04-06_ · https://arxiv.org/abs/2604.04787v1  
  <details><summary>Abstract</summary>

  We introduce AvatarPointillist, a novel framework for generating dynamic 4D Gaussian avatars from a single portrait image. At the core of our method is a decoder-only Transformer that autoregressively generates a point cloud for 3D Gaussian Splatting. This sequential approach allows for precise, adaptive construction, dynamically adjusting point density and the total number of points based on the subject's complexity. During point generation, the AR model also jointly predicts per-point binding information, enabling realistic animation. After generation, a dedicated Gaussian decoder converts the points into complete, renderable Gaussian attributes. We demonstrate that conditioning the decoder on the latent features from the AR generator enables effective interaction between stages and markedly improves fidelity. Extensive experiments validate that AvatarPointillist produces high-quality, photorealistic, and controllable avatars. We believe this autoregressive formulation represents a new paradigm for avatar generation, and we will release our code inspire future research.

  </details>



- **3D Gaussian Splatting for Annular Dark Field Scanning Transmission Electron Microscopy Tomography Reconstruction**  
  Beiyuan Zhang, Hesong Li, Ruiwen Shao, Ying Fu  
  _2026-04-06_ · https://arxiv.org/abs/2604.04693v1  
  <details><summary>Abstract</summary>

  Analytical Dark Field Scanning Transmission Electron Microscopy (ADF-STEM) tomography reconstructs nanoscale materials in 3D by integrating multi-view tilt-series images, enabling precise analysis of their structural and compositional features. Although integrating more tilt views improves 3D reconstruction, it requires extended electron exposure that risks damaging dose-sensitive materials and introduces drift and misalignment, making it difficult to balance reconstruction fidelity with sample preservation. In practice, sparse-view acquisition is frequently required, yet conventional ADF-STEM methods degrade under limited views, exhibiting artifacts and reduced structural fidelity. To resolve these issues, in this paper, we adapt 3D GS to this domain with three key components. We first model the local scattering strength as a learnable scalar field, denza, to address the mismatch between 3DGS and ADF-STEM imaging physics. Then we introduce a coefficient $γ$ to stabilize scattering across tilt angles, ensuring consistent denza via scattering view normalization. Finally, We incorporate a loss function that includes a 2D Fourier amplitude term to suppress missing wedge artifacts in sparse-view reconstruction. Experiments on 45-view and 15-view tilt series show that DenZa-Gaussian produces high-fidelity reconstructions and 2D projections that align more closely with original tilts, demonstrating superior robustness under sparse-view conditions.

  </details>



- **PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis**  
  Inseong Choi, Siwoo Lee, Seung-Hun Nam, Soohwan Song  
  _2026-04-06_ · https://arxiv.org/abs/2604.04576v1  
  <details><summary>Abstract</summary>

  Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results.The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.

  </details>



- **GA-GS: Generation-Assisted Gaussian Splatting for Static Scene Reconstruction**  
  Yedong Shen, Shiqi Zhang, Sha Zhang, Yifan Duan, Xinran Zhang, Wenhao Yu, Lu Zhang, Jiajun Deng, Yanyong Zhang  
  _2026-04-06_ · https://arxiv.org/abs/2604.04331v1  
  <details><summary>Abstract</summary>

  Reconstructing static 3D scene from monocular video with dynamic objects is important for numerous applications such as virtual reality and autonomous driving. Current approaches typically rely on background for static scene reconstruction, limiting the ability to recover regions occluded by dynamic objects. In this paper, we propose GA-GS, a Generation-Assisted Gaussian Splatting method for Static Scene Reconstruction. The key innovation of our work lies in leveraging generation to assist in reconstructing occluded regions. We employ a motion-aware module to segment and remove dynamic regions, and thenuse a diffusion model to inpaint the occluded areas, providing pseudo-ground-truth supervision. To balance contributions from real background and generated region, we introduce a learnable authenticity scalar for each Gaussian primitive, which dynamically modulates opacity during splatting for authenticity-aware rendering and supervision. Since no existing dataset provides ground-truth static scene of video with dynamic objects, we construct a dataset named Trajectory-Match, using a fixed-path robot to record each scene with/without dynamic objects, enabling quantitative evaluation in reconstruction of occluded regions. Extensive experiments on both the DAVIS and our dataset show that GA-GS achieves state-of-the-art performance in static scene reconstruction, especially in challenging scenarios with large-scale, persistent occlusions.

  </details>



- **4C4D: 4 Camera 4D Gaussian Splatting**  
  Junsheng Zhou, Zhifan Yang, Liang Han, Wenyuan Zhang, Kanle Shi, Shenkun Xu, Yu-Shen Liu  
  _2026-04-05_ · https://arxiv.org/abs/2604.04063v1  
  <details><summary>Abstract</summary>

  This paper tackles the challenge of recovering 4D dynamic scenes from videos captured by as few as four portable cameras. Learning to model scene dynamics for temporally consistent novel-view rendering is a foundational task in computer graphics, where previous works often require dense multi-view captures using camera arrays of dozens or even hundreds of views. We propose \textbf{4C4D}, a novel framework that enables high-fidelity 4D Gaussian Splatting from video captures of extremely sparse cameras. Our key insight lies that the geometric learning under sparse settings is substantially more difficult than modeling appearance. Driven by this observation, we introduce a Neural Decaying Function on Gaussian opacities for enhancing the geometric modeling capability of 4D Gaussians. This design mitigates the inherent imbalance between geometry and appearance modeling in 4DGS by encouraging the 4DGS gradients to focus more on geometric learning. Extensive experiments across sparse-view datasets with varying camera overlaps show that 4C4D achieves superior performance over prior art. Project page at: https://junshengzhou.github.io/4C4D.

  </details>



- **HOIGS: Human-Object Interaction Gaussian Splatting**  
  Taewoo Kim, Suwoong Yeom, Jaehyun Pyun, Geonho Cha, Dongyoon Wee, Joonsik Nam, Yun-Seong Jeong, Kyeongbo Kong, Suk-Ju Kang  
  _2026-04-05_ · https://arxiv.org/abs/2604.04016v1  
  <details><summary>Abstract</summary>

  Reconstructing dynamic scenes with complex human-object interactions is a fundamental challenge in computer vision and graphics. Existing Gaussian Splatting methods either rely on human pose priors while neglecting dynamic objects, or approximate all motions within a single field, limiting their ability to capture interaction-rich dynamics. To address this gap, we propose Human-Object Interaction Gaussian Splatting (HOIGS), which explicitly models interaction-induced deformation between humans and objects through a cross-attention-based HOI module. Distinct deformation baselines are employed to extract features: HexPlane for humans and Cubic Hermite Spline (CHS) for objects. By integrating these heterogeneous features, HOIGS effectively captures interdependent motions and improves deformation estimation in scenarios involving occlusion, contact, and object manipulation. Comprehensive experiments on multiple datasets demonstrate that our method consistently outperforms state-of-the-art human-centric and 4D Gaussian approaches, highlighting the importance of explicitly modeling human-object interactions for high-fidelity reconstruction.

  </details>



- **M2StyleGS: Multi-Modality 3D Style Transfer with Gaussian Splatting**  
  Xingyu Miao, Xueqi Qiu, Haoran Duan, Yawen Huang, Xian Wu, Jingjing Deng, Yang Long  
  _2026-04-04_ · https://arxiv.org/abs/2604.03773v1  
  <details><summary>Abstract</summary>

  Conventional 3D style transfer methods rely on a fixed reference image to apply artistic patterns to 3D scenes. However, in practical applications such as virtual or augmented reality, users often prefer more flexible inputs, including textual descriptions and diverse imagery. In this work, we introduce a novel real-time styling technique M2StyleGS to generate a sequence of precisely color-mapped views. It utilizes 3D Gaussian Splatting (3DGS) as a 3D presentation and multi-modality knowledge refined by CLIP as a reference style. M2StyleGS resolves the abnormal transformation issue by employing a precise feature alignment, namely subdivisive flow, it strengthens the projection of the mapped CLIP text-visual combination feature to the VGG style feature. In addition, we introduce observation loss, which assists in the stylized scene better matching the reference style during the generation, and suppression loss, which suppresses the offset of reference color information throughout the decoding process. By integrating these approaches, M2StyleGS can employ text or images as references to generate a set of style-enhanced novel views. Our experiments show that M2StyleGS achieves better visual quality and surpasses the previous work by up to 32.92% in terms of consistency.

  </details>



- **CGHair: Compact Gaussian Hair Reconstruction with Card Clustering**  
  Haimin Luo, Srinjay Sarkar, Albert Mosella-Montoro, Francisco Vicente Carrasco, Fernando De la Torre  
  _2026-04-04_ · https://arxiv.org/abs/2604.03716v1  
  <details><summary>Abstract</summary>

  We present a compact pipeline for high-fidelity hair reconstruction from multi-view images. While recent 3D Gaussian Splatting (3DGS) methods achieve realistic results, they often require millions of primitives, leading to high storage and rendering costs. Observing that hair exhibits structural and visual similarities across a hairstyle, we cluster strands into representative hair cards and group these into shared texture codebooks. Our approach integrates this structure with 3DGS rendering, significantly reducing reconstruction time and storage while maintaining comparable visual quality. In addition, we propose a generative prior accelerated method to reconstruct the initial strand geometry from a set of images. Our experiments demonstrate a 4-fold reduction in strand reconstruction time and achieve comparable rendering performance with over 200x lower memory footprint.

  </details>


