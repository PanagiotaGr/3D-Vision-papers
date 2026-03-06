# Gaussian Splatting & 3DGS

_Updated: 2026-03-06 07:06 UTC_

Total papers shown: **11**


---

- **SSR-GS: Separating Specular Reflection in Gaussian Splatting for Glossy Surface Reconstruction**  
  Ningjing Fan, Yiqun Wang  
  _2026-03-05_ · https://arxiv.org/abs/2603.05152v1  
  <details><summary>Abstract</summary>

  In recent years, 3D Gaussian splatting (3DGS) has achieved remarkable progress in novel view synthesis. However, accurately reconstructing glossy surfaces under complex illumination remains challenging, particularly in scenes with strong specular reflections and multi-surface interreflections. To address this issue, we propose SSR-GS, a specular reflection modeling framework for glossy surface reconstruction. Specifically, we introduce a prefiltered Mip-Cubemap to model direct specular reflections efficiently, and propose an IndiASG module to capture indirect specular reflections. Furthermore, we design Visual Geometry Priors (VGP) that couple a reflection-aware visual prior via a reflection score (RS) to downweight the photometric loss contribution of reflection-dominated regions, with geometry priors derived from VGGT, including progressively decayed depth supervision and transformed normal constraints. Extensive experiments on both synthetic and real-world datasets demonstrate that SSR-GS achieves state-of-the-art performance in glossy surface reconstruction.

  </details>



- **GloSplat: Joint Pose-Appearance Optimization for Faster and More Accurate 3D Reconstruction**  
  Tianyu Xiong, Rui Li, Linjie Li, Jiaqi Yang  
  _2026-03-05_ · https://arxiv.org/abs/2603.04847v1  
  <details><summary>Abstract</summary>

  Feature extraction, matching, structure from motion (SfM), and novel view synthesis (NVS) have traditionally been treated as separate problems with independent optimization objectives. We present GloSplat, a framework that performs \emph{joint pose-appearance optimization} during 3D Gaussian Splatting training. Unlike prior joint optimization methods (BARF, NeRF--, 3RGS) that rely purely on photometric gradients for pose refinement, GloSplat preserves \emph{explicit SfM feature tracks} as first-class entities throughout training: track 3D points are maintained as separate optimizable parameters from Gaussian primitives, providing persistent geometric anchors via a reprojection loss that operates alongside photometric supervision. This architectural choice prevents early-stage pose drift while enabling fine-grained refinement -- a capability absent in photometric-only approaches. We introduce two pipeline variants: (1) \textbf{GloSplat-F}, a COLMAP-free variant using retrieval-based pair selection for efficient reconstruction, and (2) \textbf{GloSplat-A}, an exhaustive matching variant for maximum quality. Both employ global SfM initialization followed by joint photometric-geometric optimization during 3DGS training. Experiments demonstrate that GloSplat-F achieves state-of-the-art among COLMAP-free methods while GloSplat-A surpasses all COLMAP-based baselines.

  </details>



- **DSA-SRGS: Super-Resolution Gaussian Splatting for Dynamic Sparse-View DSA Reconstruction**  
  Shiyu Zhang, Zhicong Wu, Huangxuan Zhao, Zhentao Liu, Lei Chen, Yong Luo, Lefei Zhang, Zhiming Cui, Ziwen Ke, Bo Du  
  _2026-03-05_ · https://arxiv.org/abs/2603.04770v1  
  <details><summary>Abstract</summary>

  Digital subtraction angiography (DSA) is a key imaging technique for the auxiliary diagnosis and treatment of cerebrovascular diseases. Recent advancements in gaussian splatting and dynamic neural representations have enabled robust 3D vessel reconstruction from sparse dynamic inputs. However, these methods are fundamentally constrained by the resolution of input projections, where performing naive upsampling to enhance rendering resolution inevitably results in severe blurring and aliasing artifacts. Such lack of super-resolution capability prevents the reconstructed 4D models from recovering fine-grained vascular details and intricate branching structures, which restricts their application in precision diagnosis and treatment. To solve this problem, this paper proposes DSA-SRGS, the first super-resolution gaussian splatting framework for dynamic sparse-view DSA reconstruction. Specifically, we introduce a Multi-Fidelity Texture Learning Module that integrates high-quality priors from a fine-tuned DSA-specific super-resolution model, into the 4D reconstruction optimization. To mitigate potential hallucination artifacts from pseudo-labels, this module employs a Confidence-Aware Strategy to adaptively weight supervision signals between the original low-resolution projections and the generated high-resolution pseudo-labels. Furthermore, we develop Radiative Sub-Pixel Densification, an adaptive strategy that leverages gradient accumulation from high-resolution sub-pixel sampling to refine the 4D radiative gaussian kernels. Extensive experiments on two clinical DSA datasets demonstrate that DSA-SRGS significantly outperforms state-of-the-art methods in both quantitative metrics and qualitative visual fidelity.

  </details>



- **Gaussian Wardrobe: Compositional 3D Gaussian Avatars for Free-Form Virtual Try-On**  
  Zhiyi Chen, Hsuan-I Ho, Tianjian Jiang, Jie Song, Manuel Kaufmann, Chen Guo  
  _2026-03-04_ · https://arxiv.org/abs/2603.04290v2  
  <details><summary>Abstract</summary>

  We introduce Gaussian Wardrobe, a novel framework to digitalize compositional 3D neural avatars from multi-view videos. Existing methods for 3D neural avatars typically treat the human body and clothing as an inseparable entity. However, this paradigm fails to capture the dynamics of complex free-form garments and limits the reuse of clothing across different individuals. To overcome these problems, we develop a novel, compositional 3D Gaussian representation to build avatars from multiple layers of free-form garments. The core of our method is decomposing neural avatars into bodies and layers of shape-agnostic neural garments. To achieve this, our framework learns to disentangle each garment layer from multi-view videos and canonicalizes it into a shape-independent space. In experiments, our method models photorealistic avatars with high-fidelity dynamics, achieving new state-of-the-art performance on novel pose synthesis benchmarks. In addition, we demonstrate that the learned compositional garments contribute to a versatile digital wardrobe, enabling a practical virtual try-on application where clothing can be freely transferred to new subjects. Project page: https://ait.ethz.ch/gaussianwardrobe

  </details>



- **EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding**  
  Seungjun Lee, Zihan Wang, Yunsong Wang, Gim Hee Lee  
  _2026-03-04_ · https://arxiv.org/abs/2603.04254v1  
  <details><summary>Abstract</summary>

  Understanding a 3D scene immediately with its exploration is essential for embodied tasks, where an agent must construct and comprehend the 3D scene in an online and nearly real-time manner. In this study, we propose EmbodiedSplat, an online feed-forward 3DGS for open-vocabulary scene understanding that enables simultaneous online 3D reconstruction and 3D semantic understanding from the streaming images. Unlike existing open-vocabulary 3DGS methods which are typically restricted to either offline or per-scene optimization setting, our objectives are two-fold: 1) Reconstructs the semantic-embedded 3DGS of the entire scene from over 300 streaming images in an online manner. 2) Highly generalizable to novel scenes with feed-forward design and supports nearly real-time 3D semantic reconstruction when combined with real-time 2D models. To achieve these objectives, we propose an Online Sparse Coefficients Field with a CLIP Global Codebook where it binds the 2D CLIP embeddings to each 3D Gaussian while minimizing memory consumption and preserving the full semantic generalizability of CLIP. Furthermore, we generate 3D geometric-aware CLIP features by aggregating the partial point cloud of 3DGS through 3D U-Net to compensate the 3D geometric prior to 2D-oriented language embeddings. Extensive experiments on diverse indoor datasets, including ScanNet, ScanNet++, and Replica, demonstrate both the effectiveness and efficiency of our method. Check out our project page in https://0nandon.github.io/EmbodiedSplat/.

  </details>



- **DM-CFO: A Diffusion Model for Compositional 3D Tooth Generation with Collision-Free Optimization**  
  Yan Tian, Pengcheng Xue, Weiping Ding, Mahmoud Hassaballah, Karen Egiazarian, Aura Conci, Abdulkadir Sengur, Leszek Rutkowski  
  _2026-03-04_ · https://arxiv.org/abs/2603.03602v1  
  <details><summary>Abstract</summary>

  The automatic design of a 3D tooth model plays a crucial role in dental digitization. However, current approaches face challenges in compositional 3D tooth generation because both the layouts and shapes of missing teeth need to be optimized.In addition, collision conflicts are often omitted in 3D Gaussian-based compositional 3D generation, where objects may intersect with each other due to the absence of explicit geometric information on the object surfaces. Motivated by graph generation through diffusion models and collision detection using 3D Gaussians, we propose an approach named DM-CFO for compositional tooth generation, where the layout of missing teeth is progressively restored during the denoising phase under both text and graph constraints. Then, the Gaussian parameters of each layout-guided tooth and the entire jaw are alternately updated using score distillation sampling (SDS). Furthermore, a regularization term based on the distances between the 3D Gaussians of neighboring teeth and the anchor tooth is introduced to penalize tooth intersections. Experimental results on three tooth-design datasets demonstrate that our approach significantly improves the multiview consistency and realism of the generated teeth compared with existing methods. Project page: https://amateurc.github.io/CF-3DTeeth/.

  </details>



- **VIRGi: View-dependent Instant Recoloring of 3D Gaussians Splats**  
  Alessio Mazzucchelli, Ivan Ojeda-Martin, Fernando Rivas-Manzaneque, Elena Garces, Adrian Penate-Sanchez, Francesc Moreno-Noguer  
  _2026-03-03_ · https://arxiv.org/abs/2603.02986v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has recently transformed the fields of novel view synthesis and 3D reconstruction due to its ability to accurately model complex 3D scenes and its unprecedented rendering performance. However, a significant challenge persists: the absence of an efficient and photorealistic method for editing the appearance of the scene's content. In this paper we introduce VIRGi, a novel approach for rapidly editing the color of scenes modeled by 3DGS while preserving view-dependent effects such as specular highlights. Key to our method are a novel architecture that separates color into diffuse and view-dependent components, and a multi-view training strategy that integrates image patches from multiple viewpoints. Improving over the conventional single-view batch training, our 3DGS representation provides more accurate reconstruction and serves as a solid representation for the recoloring task. For 3DGS recoloring, we then introduce a rapid scheme requiring only one manually edited image of the scene from the end-user. By fine-tuning the weights of a single MLP, alongside a module for single-shot segmentation of the editable area, the color edits are seamlessly propagated to the entire scene in just two seconds, facilitating real-time interaction and providing control over the strength of the view-dependent effects. An exhaustive validation on diverse datasets demonstrates significant quantitative and qualitative advancements over competitors based on Neural Radiance Fields representations.

  </details>



- **Articulation in Motion: Prior-free Part Mobility Analysis for Articulated Objects By Dynamic-Static Disentanglement**  
  Hao Ai, Wenjie Chang, Jianbo Jiao, Ales Leonardis, Ofek Eyal  
  _2026-03-03_ · https://arxiv.org/abs/2603.02910v1  
  <details><summary>Abstract</summary>

  Articulated objects are ubiquitous in daily life. Our goal is to achieve a high-quality reconstruction, segmentation of independent moving parts, and analysis of articulation. Recent methods analyse two different articulation states and perform per-point part segmentation, optimising per-part articulation using cross-state correspondences, given a priori knowledge of the number of parts. Such assumptions greatly limit their applications and performance. Their robustness is reduced when objects cannot be clearly visible in both states. To address these issues, in this paper, we present a new framework, Articulation in Motion (AiM). We infer part-level decomposition, articulation kinematics, and reconstruct an interactive 3D digital replica from a user-object interaction video and a start-state scan. We propose a dual-Gaussian scene representation that is learned from an initial 3DGS scan of the object and a video that shows the movement of separate parts. It uses motion cues to segment the object into parts and assign articulation joints. Subsequently, a robust, sequential RANSAC is employed to achieve part mobility analysis without any part-level structural priors, which clusters moving primitives into rigid parts and estimates kinematics while automatically determining the number of parts. The proposed approach separates the object into parts, each represented as a 3D Gaussian set, enabling high-quality rendering. Our approach yields higher quality part segmentation than previous methods, without prior knowledge. Extensive experimental analysis on both simple and complex objects validates the effectiveness and strong generalisation ability of our approach. Project page: https://haoai-1997.github.io/AiM/.

  </details>



- **Intrinsic Geometry-Appearance Consistency Optimization for Sparse-View Gaussian Splatting**  
  Kaiqiang Xiong, Rui Peng, Jiahao Wu, Zhanke Wang, Jie Liang, Xiaoyun Zheng, Feng Gao, Ronggang Wang  
  _2026-03-03_ · https://arxiv.org/abs/2603.02893v1  
  <details><summary>Abstract</summary>

  3D human reconstruction from a single image is a challenging problem and has been exclusively studied in the literature. Recently, some methods have resorted to diffusion models for guidance, optimizing a 3D representation via Score Distillation Sampling(SDS) or generating a back-view image for facilitating reconstruction. However, these methods tend to produce unsatisfactory artifacts (\textit{e.g.} flattened human structure or over-smoothing results caused by inconsistent priors from multiple views) and struggle with real-world generalization in the wild. In this work, we present \emph{MVD-HuGaS}, enabling free-view 3D human rendering from a single image via a multi-view human diffusion model. We first generate multi-view images from the single reference image with an enhanced multi-view diffusion model, which is well fine-tuned on high-quality 3D human datasets to incorporate 3D geometry priors and human structure priors. To infer accurate camera poses from the sparse generated multi-view images for reconstruction, an alignment module is introduced to facilitate joint optimization of 3D Gaussians and camera poses. Furthermore, we propose a depth-based Facial Distortion Mitigation module to refine the generated facial regions, thereby improving the overall fidelity of the reconstruction. Finally, leveraging the refined multi-view images, along with their accurate camera poses, MVD-HuGaS optimizes the 3D Gaussians of the target human for high-fidelity free-view renderings. Extensive experiments on Thuman2.0 and 2K2K datasets show that the proposed MVD-HuGaS achieves state-of-the-art performance on single-view 3D human rendering.

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


