# NeRF & Neural Radiance Fields

_Updated: 2026-03-05 07:08 UTC_

Total papers shown: **9**


---

- **BLOCK: An Open-Source Bi-Stage MLLM Character-to-Skin Pipeline for Minecraft**  
  Hengquan Guo  
  _2026-03-04_ · https://arxiv.org/abs/2603.03964v1  
  <details><summary>Abstract</summary>

  We present \textbf{BLOCK}, an open-source bi-stage character-to-skin pipeline that generates pixel-perfect Minecraft skins from arbitrary character concepts. BLOCK decomposes the problem into (i) a \textbf{3D preview synthesis stage} driven by a large multimodal model (MLLM) with a carefully designed prompt-and-reference template, producing a consistent dual-panel (front/back) oblique-view Minecraft-style preview; and (ii) a \textbf{skin decoding stage} based on a fine-tuned FLUX.2 model that translates the preview into a skin atlas image. We further propose \textbf{EvolveLoRA}, a progressive LoRA curriculum (text-to-image $\rightarrow$ image-to-image $\rightarrow$ preview-to-skin) that initializes each phase from the previous adapter to improve stability and efficiency. BLOCK is released with all prompt templates and fine-tuned weights to support reproducible character-to-skin generation.

  </details>



- **WSI-INR: Implicit Neural Representations for Lesion Segmentation in Whole-Slide Images**  
  Yunheng Wu, Wenqi Huang, Liangyi Wang, Masahiro Oda, Yuichiro Hayashi, Daniel Rueckert, Kensaku Mori  
  _2026-03-04_ · https://arxiv.org/abs/2603.03749v1  
  <details><summary>Abstract</summary>

  Whole-slide images (WSIs) are fundamental for computational pathology, where accurate lesion segmentation is critical for clinical decision making. Existing methods partition WSIs into discrete patches, disrupting spatial continuity and treating multi-resolution views as independent samples, which leads to spatially fragmented segmentation and reduced robustness to resolution variations. To address the issues, we propose WSI-INR, a novel patch-free framework based on Implicit Neural Representations (INRs). WSI-INR models the WSI as a continuous implicit function mapping spatial coordinates directly to tissue semantics features, outputting segmentation results while preserving intrinsic spatial information across the entire slide. In the WSI-INR, we incorporate multi-resolution hash grid encoding to regard different resolution levels as varying sampling densities of the same continuous tissue, achieving a consistent feature representation across resolutions. In addition, by jointly training a shared INR decoder, WSI-INR can capture general priors across different cases. Experimental results showed that WSI-INR maintains robust segmentation performance across resolutions; at Base/4, our resolution-specific optimization improves Dice score by +26.11%, while U-Net and TransUNet decrease by 54.28% and 36.18%, respectively. Crucially, this work enables INRs to segment highly heterogeneous pathological lesions beyond structurally consistent anatomical tissues, offering a fresh perspective for pathological analysis.

  </details>



- **VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats**  
  Alessio Mazzucchelli, Ivan Ojeda-Martin, Fernando Rivas-Manzaneque, Elena Garces, Adrian Penate-Sanchez, Francesc Moreno-Noguer  
  _2026-03-03_ · https://arxiv.org/abs/2603.02986v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.

  </details>



- **Generalized non-exponential Gaussian splatting**  
  Sébastien Speierer, Adrian Jarabo  
  _2026-03-03_ · https://arxiv.org/abs/2603.02887v2  
  <details><summary>Abstract</summary>

  In this work we generalize 3D Gaussian splatting (3DGS) to a wider family of physically-based alpha-blending operators. 3DGS has become the standard de-facto for radiance field rendering and reconstruction, given its flexibility and efficiency. At its core, it is based on alpha-blending sorted semitransparent primitives, which in the limit converges to the classic radiative transfer function with exponential transmittance. Inspired by recent research on non-exponential radiative transfer, we generalize the image formation model of 3DGS to non-exponential regimes. Based on this generalization, we use a quadratic transmittance to define sub-linear, linear, and super-linear versions of 3DGS, which exhibit faster-than-exponential decay. We demonstrate that these new non-exponential variants achieve similar quality than the original 3DGS but significantly reduce the number of overdraws, which result on speed-ups of up to $4\times$ in complex real-world captures, on a ray-tracing-based renderer.

  </details>



- **Multimodal-Prior-Guided Importance Sampling for Hierarchical Gaussian Splatting in Sparse-View Novel View Synthesis**  
  Kaiqiang Xiong, Zhanke Wang, Ronggang Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02866v1  
  <details><summary>Abstract</summary>

  We present multimodal-prior-guided importance sampling as the central mechanism for hierarchical 3D Gaussian Splatting (3DGS) in sparse-view novel view synthesis. Our sampler fuses complementary cues { -- } photometric rendering residuals, semantic priors, and geometric priors { -- } to produce a robust, local recoverability estimate that directly drives where to inject fine Gaussians. Built around this sampling core, our framework comprises (1) a coarse-to-fine Gaussian representation that encodes global shape with a stable coarse layer and selectively adds fine primitives where the multimodal metric indicates recoverable detail; and (2) a geometric-aware sampling and retention policy that concentrates refinement on geometrically critical and complex regions while protecting newly added primitives in underconstrained areas from premature pruning. By prioritizing regions supported by consistent multimodal evidence rather than raw residuals alone, our method alleviates overfitting texture-induced errors and suppresses noise from pose/appearance inconsistencies. Experiments on diverse sparse-view benchmarks demonstrate state-of-the-art reconstructions, with up to +0.3 dB PSNR on DTU.

  </details>



- **R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild**  
  Margherita Lea Corona, Wieland Morgenstern, Peter Eisert, Anna Hilsmann  
  _2026-03-03_ · https://arxiv.org/abs/2603.02801v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has established itself as a leading technique for 3D reconstruction and novel view synthesis of static scenes, achieving outstanding rendering quality and fast training. However, the method does not explicitly model the scene illumination, making it unsuitable for relighting tasks. Furthermore, 3DGS struggles to reconstruct scenes captured in the wild by unconstrained photo collections featuring changing lighting conditions. In this paper, we present R3GW, a novel method that learns a relightable 3DGS representation of an outdoor scene captured in the wild. Our approach separates the scene into a relightable foreground and a non-reflective background (the sky), using two distinct sets of Gaussians. R3GW models view-dependent lighting effects in the foreground reflections by combining Physically Based Rendering with the 3DGS scene representation in a varying illumination setting. We evaluate our method quantitatively and qualitatively on the NeRF-OSR dataset, offering state-of-the-art performance and enhanced support for physically-based relighting of unconstrained scenes. Our method synthesizes photorealistic novel views under arbitrary illumination conditions. Additionally, our representation of the sky mitigates depth reconstruction artifacts, improving rendering quality at the sky-foreground boundary

  </details>



- **Neural Electromagnetic Fields for High-Resolution Material Parameter Reconstruction**  
  Zhe Chen, Peilin Zheng, Wenshuo Chen, Xiucheng Wang, Yutao Yue, Nan Cheng  
  _2026-03-03_ · https://arxiv.org/abs/2603.02582v1  
  <details><summary>Abstract</summary>

  Creating functional Digital Twins, simulatable 3D replicas of the real world, is a central challenge in computer vision. Current methods like NeRF produce visually rich but functionally incomplete twins. The key barrier is the lack of underlying material properties (e.g., permittivity, conductivity). Acquiring this information for every point in a scene via non-contact, non-invasive sensing is a primary goal, but it demands solving a notoriously ill-posed physical inversion problem. Standard remote signals, like images and radio frequencies (RF), deeply entangle the unknown geometry, ambient field, and target materials. We introduce NEMF, a novel framework for dense, non-invasive physical inversion designed to build functional digital twins. Our key insight is a systematic disentanglement strategy. NEMF leverages high-fidelity geometry from images as a powerful anchor, which first enables the resolution of the ambient field. By constraining both geometry and field using only non-invasive data, the original ill-posed problem transforms into a well-posed, physics-supervised learning task. This transformation unlocks our core inversion module: a decoder. Guided by ambient RF signals and a differentiable layer incorporating physical reflection models, it learns to explicitly output a continuous, spatially-varying field of the scene's underlying material parameters. We validate our framework on high-fidelity synthetic datasets. Experiments show our non-invasive inversion reconstructs these material maps with high accuracy, and the resulting functional twin enables high-fidelity physical simulation. This advance moves beyond passive visual replicas, enabling the creation of truly functional and simulatable models of the physical world.

  </details>



- **SemGS: Feed-Forward Semantic 3D Gaussian Splatting from Sparse Views for Generalizable Scene Understanding**  
  Sheng Ye, Zhen-Hui Dong, Ruoyu Fan, Tian Lv, Yong-Jin Liu  
  _2026-03-03_ · https://arxiv.org/abs/2603.02548v1  
  <details><summary>Abstract</summary>

  Semantic understanding of 3D scenes is essential for robots to operate effectively and safely in complex environments. Existing methods for semantic scene reconstruction and semantic-aware novel view synthesis often rely on dense multi-view inputs and require scene-specific optimization, limiting their practicality and scalability in real-world applications. To address these challenges, we propose SemGS, a feed-forward framework for reconstructing generalizable semantic fields from sparse image inputs. SemGS uses a dual-branch architecture to extract color and semantic features, where the two branches share shallow CNN layers, allowing semantic reasoning to leverage textural and structural cues in color appearance. We also incorporate a camera-aware attention mechanism into the feature extractor to explicitly model geometric relationships between camera viewpoints. The extracted features are decoded into dual-Gaussians that share geometric consistency while preserving branch-specific attributes, and further rasterized to synthesize semantic maps under novel viewpoints. Additionally, we introduce a regional smoothness loss to enhance semantic coherence. Experiments show that SemGS achieves state-of-the-art performance on benchmark datasets, while providing rapid inference and strong generalization capabilities across diverse synthetic and real-world scenarios.

  </details>



- **OnlineX: Unified Online 3D Reconstruction and Understanding with Active-to-Stable State Evolution**  
  Chong Xia, Fangfu Liu, Yule Wang, Yize Pang, Yueqi Duan  
  _2026-03-02_ · https://arxiv.org/abs/2603.02134v2  
  <details><summary>Abstract</summary>

  Recent advances in generalizable 3D Gaussian Splatting (3DGS) have enabled rapid 3D scene reconstruction within seconds, eliminating the need for per-scene optimization. However, existing methods primarily follow an offline reconstruction paradigm, lacking the capacity for continuous reconstruction, which limits their applicability to online scenarios such as robotics and VR/AR. In this paper, we introduce OnlineX, a feed-forward framework that reconstructs both 3D visual appearance and language fields in an online manner using only streaming images. A key challenge in online formulation is the cumulative drift issue, which is rooted in the fundamental conflict between two opposing roles of the memory state: an active role that constantly refreshes to capture high-frequency local geometry, and a stable role that conservatively accumulates and preserves the long-term global structure. To address this, we introduce a decoupled active-to-stable state evolution paradigm. Our framework decouples the memory state into a dedicated active state and a persistent stable state, and then cohesively fuses the information from the former into the latter to achieve both fidelity and stability. Moreover, we jointly model visual appearance and language fields and incorporate an implicit Gaussian fusion module to enhance reconstruction quality. Experiments on mainstream datasets demonstrate that our method consistently outperforms prior work in novel view synthesis and semantic understanding, showcasing robust performance across input sequences of varying lengths with real-time inference speed.

  </details>


