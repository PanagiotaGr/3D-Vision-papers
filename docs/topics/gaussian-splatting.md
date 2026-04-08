# Gaussian Splatting & 3DGS

_Updated: 2026-04-08 07:51 UTC_

Total papers shown: **17**


---

- **Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction**  
  Yangyi Xiao, Siting Zhu, Baoquan Yang, Tianchen Deng, Yongbo Chen, Hesheng Wang  
  _2026-04-07_ · https://arxiv.org/abs/2604.05908v1  
  <details><summary>Abstract</summary>

  Multi-traversal scene reconstruction is important for high-fidelity autonomous driving simulation and digital twin construction. This task involves integrating multiple sequences captured from the same geographical area at different times. In this context, a primary challenge is the significant appearance inconsistency across traversals caused by varying illumination and environmental conditions, despite the shared underlying geometry. This paper presents ADM-GS (Appearance Decomposition Gaussian Splatting for Multi-Traversal Reconstruction), a framework that applies an explicit appearance decomposition to the static background to alleviate appearance entanglement across traversals. For the static background, we decompose the appearance into traversal-invariant material, representing intrinsic material properties, and traversal-dependent illumination, capturing lighting variations. Specifically, we propose a neural light field that utilizes a frequency-separated hybrid encoding strategy. By incorporating surface normals and explicit reflection vectors, this design separately captures low-frequency diffuse illumination and high-frequency specular reflections. Quantitative evaluations on the Argoverse 2 and Waymo Open datasets demonstrate the effectiveness of ADM-GS. In multi-traversal experiments, our method achieves a +0.98 dB PSNR improvement over existing latent-based baselines while producing more consistent appearance across traversals. Code will be available at https://github.com/IRMVLab/ADM-GS.

  </details>



- **GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance**  
  Weiqi Zhang, Junsheng Zhou, Haotian Geng, Kanle Shi, Shenkun Xu, Yi Fang, Yu-Shen Liu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05721v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has demonstrated superior performance in rendering efficiency and quality, yet the generation of 3D Gaussians still remains a challenge without proper geometric priors. Existing methods have explored predicting point maps as geometric references for inferring Gaussian primitives, while the unreliable estimated geometries may lead to poor generations. In this work, we introduce GaussianGrow, a novel approach that generates 3D Gaussians by learning to grow them from easily accessible 3D point clouds, naturally enforcing geometric accuracy in Gaussian generation. Specifically, we design a text-guided Gaussian growing scheme that leverages a multi-view diffusion model to synthesize consistent appearances from input point clouds for supervision. To mitigate artifacts caused by fusing neighboring views, we constrain novel views generated at non-preset camera poses identified in overlapping regions across different views. For completing the hard-to-observe regions, we propose to iteratively detect the camera pose by observing the largest un-grown regions in point clouds and inpainting them by inpainting the rendered view with a pretrained 2D diffusion model. The process continues until complete Gaussians are generated. We extensively evaluate GaussianGrow on text-guided Gaussian generation from synthetic and even real-scanned point clouds. Project Page: https://weiqi-zhang.github.io/GaussianGrow

  </details>



- **In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting**  
  Wenhui Xiao, Ethan Goan, Rodrigo Santa Cruz, David Ahmedt-Aristizabal, Olivier Salvado, Clinton Fookes, Leo Lebrat  
  _2026-04-07_ · https://arxiv.org/abs/2604.05715v1  
  <details><summary>Abstract</summary>

  Using accurate depth priors in 3D Gaussian Splatting helps mitigate artifacts caused by sparse training data and textureless surfaces. However, acquiring accurate depth maps requires specialized acquisition systems. Foundation monocular depth estimation models offer a cost-effective alternative, but they suffer from scale ambiguity, multi-view inconsistency, and local geometric inaccuracies, which can degrade rendering performance when applied naively. This paper addresses the challenge of reliably leveraging monocular depth priors for Gaussian Splatting (GS) rendering enhancement. To this end, we introduce a training framework integrating scale-ambiguous and noisy depth priors into geometric supervision. We highlight the importance of learning from weakly aligned depth variations. We introduce a method to isolate ill-posed geometry for selective monocular depth regularization, restricting the propagation of depth inaccuracies into well-reconstructed 3D structures. Extensive experiments across diverse datasets show consistent improvements in geometric accuracy, leading to more faithful depth estimation and higher rendering quality across different GS variants and monocular depth backbones tested.

  </details>



- **3D Smoke Scene Reconstruction Guided by Vision Priors from Multimodal Large Language Models**  
  Xinye Zheng, Fei Wang, Yiqi Nie, Kun Li, Junjie Chen, Jiaqi Zhao, Yanyan Wei, Zhiliang Wu  
  _2026-04-07_ · https://arxiv.org/abs/2604.05687v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D scenes from smoke-degraded multi-view images is particularly difficult because smoke introduces strong scattering effects, view-dependent appearance changes, and severe degradation of cross-view consistency. To address these issues, we propose a framework that integrates visual priors with efficient 3D scene modeling. We employ Nano-Banana-Pro to enhance smoke-degraded images and provide clearer visual observations for reconstruction and develop Smoke-GS, a medium-aware 3D Gaussian Splatting framework for smoke scene reconstruction and restoration-oriented novel view synthesis. Smoke-GS models the scene using explicit 3D Gaussians and introduces a lightweight view-dependent medium branch to capture direction-dependent appearance variations caused by smoke. Our method preserves the rendering efficiency of 3D Gaussian Splatting while improving robustness to smoke-induced degradation. Results demonstrate the effectiveness of our method for generating consistent and visually clear novel views in challenging smoke environments.

  </details>



- **PanopticQuery: Unified Query-Time Reasoning for 4D Scenes**  
  Ruilin Tang, Yang Zhou, Zhong Ye, Wenxi Liu, Yan Huang, Shengfeng He  
  _2026-04-07_ · https://arxiv.org/abs/2604.05638v1  
  <details><summary>Abstract</summary>

  Understanding dynamic 4D environments through natural language queries requires not only accurate scene reconstruction but also robust semantic grounding across space, time, and viewpoints. While recent methods using neural representations have advanced 4D reconstruction, they remain limited in contextual reasoning, especially for complex semantics such as interactions, temporal actions, and spatial relations. A key challenge lies in transforming noisy, view-dependent predictions into globally consistent 4D interpretations. We introduce PanopticQuery, a framework for unified query-time reasoning in 4D scenes. Our approach builds on 4D Gaussian Splatting for high-fidelity dynamic reconstruction and introduces a multi-view semantic consensus mechanism that grounds natural language queries by aggregating 2D semantic predictions across multiple views and time frames. This process filters inconsistent outputs, enforces geometric consistency, and lifts 2D semantics into structured 4D groundings via neural field optimization. To support evaluation, we present Panoptic-L4D, a new benchmark for language-based querying in dynamic scenes. Experiments demonstrate that PanopticQuery sets a new state of the art on complex language queries, effectively handling attributes, actions, spatial relationships, and multi-object interactions. A video demonstration is available in the supplementary materials.

  </details>



- **LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios**  
  Xiang Zhang, Tengfei Wang, Fang Xu, Xin Wang, Zongqian Zhan  
  _2026-04-07_ · https://arxiv.org/abs/2604.05402v1  
  <details><summary>Abstract</summary>

  Visual localization in large-scale UAV scenarios is a critical capability for autonomous systems, yet it remains challenging due to geometric complexity and environmental variations. While 3D Gaussian Splatting (3DGS) has emerged as a promising scene representation, existing 3DGS-based visual localization methods struggle with robust pose initialization and sensitivity to rendering artifacts in large-scale settings. To address these limitations, we propose LSGS-Loc, a novel visual localization pipeline tailored for large-scale 3DGS scenes. Specifically, we introduce a scale-aware pose initialization strategy that combines scene-agnostic relative pose estimation with explicit 3DGS scale constraints, enabling geometrically grounded localization without scene-specific training. Furthermore, in the pose refinement, to mitigate the impact of reconstruction artifacts such as blur and floaters, we develop a Laplacian-based reliability masking mechanism that guides photometric refinement toward high-quality regions. Extensive experiments on large-scale UAV benchmarks demonstrate that our method achieves state-of-the-art accuracy and robustness for unordered image queries, significantly outperforming existing 3DGS-based approaches. Code is available at: https://github.com/xzhang-z/LSGS-Loc

  </details>



- **3DTurboQuant: Training-Free Near-Optimal Quantization for 3D Reconstruction Models**  
  Jae Joong Lee  
  _2026-04-07_ · https://arxiv.org/abs/2604.05366v1  
  <details><summary>Abstract</summary>

  Every existing method for compressing 3D Gaussian Splatting, NeRF, or transformer-based 3D reconstructors requires learning a data-dependent codebook through per-scene fine-tuning. We show this is unnecessary. The parameter vectors that dominate storage in these models, 45-dimensional spherical harmonics in 3DGS and 1024-dimensional key-value vectors in DUSt3R, fall in a dimension range where a single random rotation transforms any input into coordinates with a known Beta distribution. This makes precomputed, data-independent Lloyd-Max quantization near-optimal, within a factor of 2.7 of the information-theoretic lower bound. We develop 3D, deriving (1) a dimension-dependent criterion that predicts which parameters can be quantized and at what bit-width before running any experiment, (2) norm-separation bounds connecting quantization MSE to rendering PSNR per scene, (3) an entry-grouping strategy extending rotation-based quantization to 2-dimensional hash grid features, and (4) a composable pruning-quantization pipeline with a closed-form compression ratio. On NeRF Synthetic, 3DTurboQuant compresses 3DGS by 3.5x with 0.02dB PSNR loss and DUSt3R KV caches by 7.9x with 39.7dB pointmap fidelity. No training, no codebook learning, no calibration data. Compression takes seconds. The code will be released (https://github.com/JaeLee18/3DTurboQuant)

  </details>



- **Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting**  
  Monica Tang, Avideh Zakhor  
  _2026-04-07_ · https://arxiv.org/abs/2604.05316v1  
  <details><summary>Abstract</summary>

  We present an approach for object-level detection and segmentation of target indoor assets in 3D Gaussian Splatting (3DGS) scenes, reconstructed from 360° drone-captured imagery. We introduce a 3D object codebook that jointly leverages mask semantics and spatial information of their corresponding Gaussian primitives to guide multi-view mask association and indoor asset detection. By integrating 2D object detection and segmentation models with semantically and spatially constrained merging procedures, our method aggregates masks from multiple views into coherent 3D object instances. Experiments on two large indoor scenes demonstrate reliable multi-view mask consistency, improving F1 score by 65% over state-of-the-art baselines, and accurate object-level 3D indoor asset detection, achieving an 11% mAP gain over baseline methods.

  </details>



- **SmokeGS-R: Physics-Guided Pseudo-Clean 3DGS for Real-World Multi-View Smoke Restoration**  
  Xueming Fu, Lixia Han  
  _2026-04-07_ · https://arxiv.org/abs/2604.05301v1  
  <details><summary>Abstract</summary>

  Real-world smoke simultaneously attenuates scene radiance, adds airlight, and destabilizes multi-view appearance consistency, making robust 3D reconstruction particularly difficult. We present \textbf{SmokeGS-R}, a practical pipeline developed for the NTIRE 2026 3D Restoration and Reconstruction Track 2 challenge. The key idea is to decouple geometry recovery from appearance correction: we generate physics-guided pseudo-clean supervision with a refined dark channel prior and guided filtering, train a sharp clean-only 3D Gaussian Splatting source model, and then harmonize its renderings with a donor ensemble using geometric-mean reference aggregation, LAB-space Reinhard transfer, and light Gaussian smoothing. On the official challenge testing leaderboard, the final submission achieved \mbox{PSNR $=15.217$} and \mbox{SSIM $=0.666$}. After the public release of RealX3D, we re-evaluated the same frozen result on the seven released challenge scenes without retraining and obtained \mbox{PSNR $=15.209$}, \mbox{SSIM $=0.644$}, and \mbox{LPIPS $=0.551$}, outperforming the strongest official baseline average on the same scenes by $+3.68$ dB PSNR. These results suggest that a geometry-first reconstruction strategy combined with stable post-render appearance harmonization is an effective recipe for real-world multi-view smoke restoration. The code is available at https://github.com/windrise/3drr_Track2_SmokeGS-R.

  </details>



- **Part-Level 3D Gaussian Vehicle Generation with Joint and Hinge Axis Estimation**  
  Shiyao Qian, Yuan Ren, Dongfeng Bai, Bingbing Liu  
  _2026-04-06_ · https://arxiv.org/abs/2604.05070v1  
  <details><summary>Abstract</summary>

  Simulation is essential for autonomous driving, yet current frameworks often model vehicles as rigid assets and fail to capture part-level articulation. With perception algorithms increasingly leveraging dynamics such as wheel steering or door opening, realistic simulation requires animatable vehicle representations. Existing CAD-based pipelines are limited by library coverage and fixed templates, preventing faithful reconstruction of in-the-wild instances. We propose a generative framework that, from a single image or sparse multi-view input, synthesizes an animatable 3D Gaussian vehicle. Our method addresses two challenges: (i) large 3D asset generators are optimized for static quality but not articulation, leading to distortions at part boundaries when animated; and (ii) segmentation alone cannot provide the kinematic parameters required for motion. To overcome this, we introduce a part-edge refinement module that enforces exclusive Gaussian ownership and a kinematic reasoning head that predicts joint positions and hinge axes of movable parts. Together, these components enable faithful part-aware simulation, bridging the gap between static generation and animatable vehicle models.

  </details>



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
  _2026-04-06_ · https://arxiv.org/abs/2604.04576v2  
  <details><summary>Abstract</summary>

  Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results. The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.

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


