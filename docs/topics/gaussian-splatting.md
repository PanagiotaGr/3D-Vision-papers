# Gaussian Splatting & 3DGS

_Updated: 2026-03-19 07:12 UTC_

Total papers shown: **12**


---

- **AHOY! Animatable Humans under Occlusion from YouTube Videos with Gaussian Splatting and Video Diffusion Priors**  
  Aymen Mir, Riza Alp Guler, Xiangjun Tang, Peter Wonka, Gerard Pons-Moll  
  _2026-03-18_ · https://arxiv.org/abs/2603.17975v1  
  <details><summary>Abstract</summary>

  We present AHOY, a method for reconstructing complete, animatable 3D Gaussian avatars from in-the-wild monocular video despite heavy occlusion. Existing methods assume unoccluded input-a fully visible subject, often in a canonical pose-excluding the vast majority of real-world footage where people are routinely occluded by furniture, objects, or other people. Reconstructing from such footage poses fundamental challenges: large body regions may never be observed, and multi-view supervision per pose is unavailable. We address these challenges with four contributions: (i) a hallucination-as-supervision pipeline that uses identity-finetuned diffusion models to generate dense supervision for previously unobserved body regions; (ii) a two-stage canonical-to-pose-dependent architecture that bootstraps from sparse observations to full pose-dependent Gaussian maps; (iii) a map-pose/LBS-pose decoupling that absorbs multi-view inconsistencies from the generated data; (iv) a head/body split supervision strategy that preserves facial identity. We evaluate on YouTube videos and on multi-view capture data with significant occlusion and demonstrate state-of-the-art reconstruction quality. We also demonstrate that the resulting avatars are robust enough to be animated with novel poses and composited into 3DGS scenes captured using cell-phone video. Our project page is available at https://miraymen.github.io/ahoy/

  </details>



- **CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image**  
  Yizheng Song, Yiyu Zhuang, Qipeng Xu, Haixiang Wang, Jiahe Zhu, Jing Tian, Siyu Zhu, Hao Zhu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17779v1  
  <details><summary>Abstract</summary>

  Single-view 3D human reconstruction has garnered significant attention in recent years. Despite numerous advancements, prior research has concentrated on reconstructing 3D models from clear, close-up images of individual subjects, often yielding subpar results in the more prevalent multi-person scenarios. Reconstructing 3D human crowd models is a highly intricate task, laden with challenges such as: 1) extensive occlusions, 2) low clarity, and 3) numerous and various appearances. To address this task, we propose CrowdGaussian, a unified framework that directly reconstructs multi-person 3D Gaussian Splatting (3DGS) representations from single-image inputs. To handle occlusions, we devise a self-supervised adaptation pipeline that enables the pretrained large human model to reconstruct complete 3D humans with plausible geometry and appearance from heavily occluded inputs. Furthermore, we introduce Self-Calibrated Learning (SCL). This training strategy enables single-step diffusion models to adaptively refine coarse renderings to optimal quality by blending identity-preserving samples with clean/corrupted image pairs. The outputs can be distilled back to enhance the quality of multi-person 3DGS representations. Extensive experiments demonstrate that CrowdGaussian generates photorealistic, geometrically coherent reconstructions of multi-person scenes.

  </details>



- **TAPESTRY: From Geometry to Appearance via Consistent Turntable Videos**  
  Yan Zeng, Haoran Jiang, Kaixin Yao, Qixuan Zhang, Longwen Zhang, Lan Xu, Jingyi Yu  
  _2026-03-18_ · https://arxiv.org/abs/2603.17735v1  
  <details><summary>Abstract</summary>

  Automatically generating photorealistic and self-consistent appearances for untextured 3D models is a critical challenge in digital content creation. The advancement of large-scale video generation models offers a natural approach: directly synthesizing 360-degree turntable videos (TTVs), which can serve not only as high-quality dynamic previews but also as an intermediate representation to drive texture synthesis and neural rendering. However, existing general-purpose video diffusion models struggle to maintain strict geometric consistency and appearance stability across the full range of views, making their outputs ill-suited for high-quality 3D reconstruction. To this end, we introduce TAPESTRY, a framework for generating high-fidelity TTVs conditioned on explicit 3D geometry. We reframe the 3D appearance generation task as a geometry-conditioned video diffusion problem: given a 3D mesh, we first render and encode multi-modal geometric features to constrain the video generation process with pixel-level precision, thereby enabling the creation of high-quality and consistent TTVs. Building upon this, we also design a method for downstream reconstruction tasks from the TTV input, featuring a multi-stage pipeline with 3D-Aware Inpainting. By rotating the model and performing a context-aware secondary generation, this pipeline effectively completes self-occluded regions to achieve full surface coverage. The videos generated by TAPESTRY are not only high-quality dynamic previews but also serve as a reliable, 3D-aware intermediate representation that can be seamlessly back-projected into UV textures or used to supervise neural rendering methods like 3DGS. This enables the automated creation of production-ready, complete 3D assets from untextured meshes. Experimental results demonstrate that our method outperforms existing approaches in both video consistency and final reconstruction quality.

  </details>



- **ReLaGS: Relational Language Gaussian Splatting**  
  Yaxu Xie, Abdalla Arafa, Alireza Javanmardi, Christen Millerdurai, Jia Cheng Hu, Shaoxiang Wang, Alain Pagani, Didier Stricker  
  _2026-03-18_ · https://arxiv.org/abs/2603.17605v1  
  <details><summary>Abstract</summary>

  Achieving unified 3D perception and reasoning across tasks such as segmentation, retrieval, and relation understanding remains challenging, as existing methods are either object-centric or rely on costly training for inter-object reasoning. We present a novel framework that constructs a hierarchical language-distilled Gaussian scene and its 3D semantic scene graph without scene-specific training. A Gaussian pruning mechanism refines scene geometry, while a robust multi-view language alignment strategy aggregates noisy 2D features into accurate 3D object embeddings. On top of this hierarchy, we build an open-vocabulary 3D scene graph with Vision Language derived annotations and Graph Neural Network-based relational reasoning. Our approach enables efficient and scalable open-vocabulary 3D reasoning by jointly modeling hierarchical semantics and inter/intra-object relationships, validated across tasks including open-vocabulary segmentation, scene graph generation, and relation-guided retrieval. Project page: https://dfki-av.github.io/ReLaGS/

  </details>



- **UniSem: Generalizable Semantic 3D Reconstruction from Sparse Unposed Images**  
  Guibiao Liao, Qian Ren, Kaimin Liao, Hua Wang, Zhi Chen, Luchao Wang, Yaohua Tang  
  _2026-03-18_ · https://arxiv.org/abs/2603.17519v1  
  <details><summary>Abstract</summary>

  Semantic-aware 3D reconstruction from sparse, unposed images remains challenging for feed-forward 3D Gaussian Splatting (3DGS). Existing methods often predict an over-complete set of Gaussian primitives under sparse-view supervision, leading to unstable geometry and inferior depth quality. Meanwhile, they rely solely on 2D segmenter features for semantic lifting, which provides weak 3D-level and limited generalizable supervision, resulting in incomplete 3D semantics in novel scenes. To address these issues, we propose UniSem, a unified framework that jointly improves depth accuracy and semantic generalization via two key components. First, Error-aware Gaussian Dropout (EGD) performs error-guided capacity control by suppressing redundancy-prone Gaussians using rendering error cues, producing meaningful, geometrically stable Gaussian representations for improved depth estimation. Second, we introduce a Mix-training Curriculum (MTC) that progressively blends 2D segmenter-lifted semantics with the model's own emergent 3D semantic priors, implemented with object-level prototype alignment to enhance semantic coherence and completeness. Extensive experiments on ScanNet and Replica show that UniSem achieves superior performance in depth prediction and open-vocabulary 3D segmentation across varying numbers of input views. Notably, with 16-view inputs, UniSem reduces depth Rel by 15.2% and improves open-vocabulary segmentation mAcc by 3.7% over strong baselines.

  </details>



- **Adaptive Anchor Policies for Efficient 4D Gaussian Streaming**  
  Ashim Dahal, Rabab Abdelfattah, Nick Rahimi  
  _2026-03-18_ · https://arxiv.org/abs/2603.17227v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction with Gaussian Splatting has enabled efficient streaming for real-time rendering and free-viewpoint video. However, most pipelines rely on fixed anchor selection such as Farthest Point Sampling (FPS), typically using 8,192 anchors regardless of scene complexity, which over-allocates computation under strict budgets. We propose Efficient Gaussian Streaming (EGS), a plug-in, budget-aware anchor sampler that replaces FPS with a reinforcement-learned policy while keeping the Gaussian streaming reconstruction backbone unchanged. The policy jointly selects an anchor budget and a subset of informative anchors under discrete constraints, balancing reconstruction quality and runtime using spatial features of the Gaussian representation. We evaluate EGS in two settings: fast rendering, which prioritizes runtime efficiency, and high-quality refinement, which enables additional optimization. Experiments on dynamic multi-view datasets show consistent improvements in the quality--efficiency trade-off over FPS sampling. On unseen data, in fast rendering at 256 anchors ($32\times$ fewer than 8,192), EGS improves PSNR by $+0.52$--$0.61$\,dB while running $1.29$--$1.35\times$ faster than IGS@8192 (N3DV and MeetingRoom). In high-quality refinement, EGS remains competitive with the full-anchor baseline at substantially lower anchor budgets. \emph{Code and pretrained checkpoints will be released upon acceptance.} \keywords{4D Gaussian Splatting \and 4D Gaussian Streaming \and Reinforcement Learning}

  </details>



- **SMAL-pets: SMAL Based Avatars of Pets from Single Image**  
  Piotr Borycki, Joanna Waczyńska, Yizhe Zhu, Yongqiang Gao, Przemysław Spurek  
  _2026-03-17_ · https://arxiv.org/abs/2603.17131v1  
  <details><summary>Abstract</summary>

  Creating high-fidelity, animatable 3D dog avatars remains a formidable challenge in computer vision. Unlike human digital doubles, animal reconstruction faces a critical shortage of large-scale, annotated datasets for specialized applications. Furthermore, the immense morphological diversity across species, breeds, and crosses, which varies significantly in size, proportions, and features, complicates the generalization of existing models. Current reconstruction methods often struggle to capture realistic fur textures. Additionally, ensuring these avatars are fully editable and capable of performing complex, naturalistic movements typically necessitates labor-intensive manual mesh manipulation and expert rigging. This paper introduces SMAL-pets, a comprehensive framework that generates high-quality, editable animal avatars from a single input image. Our approach bridges the gap between reconstruction and generative modeling by leveraging a hybrid architecture. Our method integrates 3D Gaussian Splatting with the SMAL parametric model to provide a representation that is both visually high-fidelity and anatomically grounded. We introduce a multimodal editing suite that enables users to refine the avatar's appearance and execute complex animations through direct textual prompts. By allowing users to control both the aesthetic and behavioral aspects of the model via natural language, SMAL-pets provides a flexible, robust tool for animation and virtual reality.

  </details>



- **M^3: Dense Matching Meets Multi-View Foundation Models for Monocular Gaussian Splatting SLAM**  
  Kerui Ren, Guanghao Li, Changjian Jiang, Yingxiang Xu, Tao Lu, Linning Xu, Junting Dong, Jiangmiao Pang, Mulin Yu, Bo Dai  
  _2026-03-17_ · https://arxiv.org/abs/2603.16844v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from uncalibrated monocular video remains challenging, as it requires both high-precision pose estimation and computationally efficient online refinement in dynamic environments. While coupling 3D foundation models with SLAM frameworks is a promising paradigm, a critical bottleneck persists: most multi-view foundation models estimate poses in a feed-forward manner, yielding pixel-level correspondences that lack the requisite precision for rigorous geometric optimization. To address this, we present M^3, which augments the Multi-view foundation model with a dedicated Matching head to facilitate fine-grained dense correspondences and integrates it into a robust Monocular Gaussian Splatting SLAM. M^3 further enhances tracking stability by incorporating dynamic area suppression and cross-inference intrinsic alignment. Extensive experiments on diverse indoor and outdoor benchmarks demonstrate state-of-the-art accuracy in both pose estimation and scene reconstruction. Notably, M^3 reduces ATE RMSE by 64.3% compared to VGGT-SLAM 2.0 and outperforms ARTDECO by 2.11 dB in PSNR on the ScanNet++ dataset.

  </details>



- **Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty**  
  Mangyu Kong, Jaewon Lee, Seongwon Lee, Euntai Kim  
  _2026-03-17_ · https://arxiv.org/abs/2603.16538v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently emerged as a powerful scene representation and is increasingly used for visual localization and pose refinement. However, despite its high-quality differentiable rendering, the robustness of 3DGS-based pose refinement remains highly sensitive to both the initial camera pose and the reconstructed geometry. In this work, we take a closer look at these limitations and identify two major sources of uncertainty: (i) pose prior uncertainty, which often arises from regression or retrieval models that output a single deterministic estimate, and (ii) geometric uncertainty, caused by imperfections in the 3DGS reconstruction that propagate errors into PnP solvers. Such uncertainties can distort reprojection geometry and destabilize optimization, even when the rendered appearance still looks plausible. To address these uncertainties, we introduce a relocalization framework that combines Monte Carlo pose sampling with Fisher Information-based PnP optimization. Our method explicitly accounts for both pose and geometric uncertainty and requires no retraining or additional supervision. Across diverse indoor and outdoor benchmarks, our approach consistently improves localization accuracy and significantly increases stability under pose and depth noise.

  </details>



- **ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars**  
  Kaiwen Song, Jinkai Cui, Juyong Zhang  
  _2026-03-17_ · https://arxiv.org/abs/2603.16447v1  
  <details><summary>Abstract</summary>

  In practical real-time XR and telepresence applications, network and computing resources fluctuate frequently. Therefore, a progressive 3D representation is needed. To this end, we propose ProgressiveAvatars, a progressive avatar representation built on a hierarchy of 3D Gaussians grown by adaptive implicit subdivision on a template mesh. 3D Gaussians are defined in face-local coordinates to remain animatable under varying expressions and head motion across multiple detail levels. The hierarchy expands when screen-space signals indicate a lack of detail, allocating resources to important areas. Leveraging importance ranking, ProgressiveAvatars supports incremental loading and rendering, adding new Gaussians as they arrive while preserving previous content, thus achieving smooth quality improvements across varying bandwidths. ProgressiveAvatars enables progressive delivery and progressive rendering under fluctuating network bandwidth and varying compute and memory resources.

  </details>



- **Leveling3D: Leveling Up 3D Reconstruction with Feed-Forward 3D Gaussian Splatting and Geometry-Aware Generation**  
  Yiming Huang, Baixiang Huang, Beilei Cui, Chi Kit Ng, Long Bai, Hongliang Ren  
  _2026-03-17_ · https://arxiv.org/abs/2603.16211v1  
  <details><summary>Abstract</summary>

  Feed-forward 3D reconstruction has revolutionized 3D vision, providing a powerful baseline for downstream tasks such as novel-view synthesis with 3D Gaussian Splatting. Previous works explore fixing the corrupted rendering results with a diffusion model. However, they lack geometric concern and fail at filling the missing area on the extrapolated view. In this work, we introduce Leveling3D, a novel pipeline that integrates feed-forward 3D reconstruction with geometrical-consistent generation to enable holistic simultaneous reconstruction and generation. We propose a geometry-aware leveling adapter, a lightweight technique that aligns internal knowledge in the diffusion model with the geometry prior from the feed-forward model. The leveling adapter enables generation on the artifact area of the extrapolated novel views caused by underconstrained regions of the 3D representation. Specifically, to learn a more diverse distributed generation, we introduce the palette filtering strategy for training, and a test-time masking refinement to prevent messy boundaries along the fixing regions. More importantly, the enhanced extrapolated novel views from Leveling3D could be used as the inputs for feed-forward 3DGS, leveling up the 3D reconstruction. We achieve SOTA performance on public datasets, including tasks such as novel-view synthesis and depth estimation.

  </details>



- **NanoGS: Training-Free Gaussian Splat Simplification**  
  Butian Xiong, Rong Liu, Tiantian Zhou, Meida Chen, Zhiwen Fan, Andrew Feng  
  _2026-03-17_ · https://arxiv.org/abs/2603.16103v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splat (3DGS) enables high-fidelity, real-time novel view synthesis by representing scenes with large sets of anisotropic primitives, but often requires millions of Splats, incurring significant storage and transmission costs. Most existing compression methods rely on GPU-intensive post-training optimization with calibrated images, limiting practical deployment. We introduce NanoGS, a training-free and lightweight framework for Gaussian Splat simplification. Instead of relying on image-based rendering supervision, NanoGS formulates simplification as local pairwise merging over a sparse spatial graph. The method approximates a pair of Gaussians with a single primitive using mass preserved moment matching and evaluates merge quality through a principled merge cost between the original mixture and its approximation. By restricting merge candidates to local neighborhoods and selecting compatible pairs efficiently, NanoGS produces compact Gaussian representations while preserving scene structure and appearance. NanoGS operates directly on existing Gaussian Splat models, runs efficiently on CPU, and preserves the standard 3DGS parameterization, enabling seamless integration with existing rendering pipelines. Experiments demonstrate that NanoGS substantially reduces primitive count while maintaining high rendering fidelity, providing an efficient and practical solution for Gaussian Splat simplification. Our project website is available at https://saliteta.github.io/NanoGS/.

  </details>


