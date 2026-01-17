# Neural Rendering & View Synthesis

_Updated: 2026-01-17 06:45 UTC_

Total papers shown: **3**


---

- **WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments**  
  Xuweiyi Chen, Wentao Zhou, Zezhou Cheng  
  _2026-01-15_ · https://arxiv.org/abs/2601.10716v1  
  <details><summary>Abstract</summary>

  We present WildRayZer, a self-supervised framework for novel view synthesis (NVS) in dynamic environments where both the camera and objects move. Dynamic content breaks the multi-view consistency that static NVS models rely on, leading to ghosting, hallucinated geometry, and unstable pose estimation. WildRayZer addresses this by performing an analysis-by-synthesis test: a camera-only static renderer explains rigid structure, and its residuals reveal transient regions. From these residuals, we construct pseudo motion masks, distill a motion estimator, and use it to mask input tokens and gate loss gradients so supervision focuses on cross-view background completion. To enable large-scale training and evaluation, we curate Dynamic RealEstate10K (D-RE10K), a real-world dataset of 15K casually captured dynamic sequences, and D-RE10K-iPhone, a paired transient and clean benchmark for sparse-view transient-aware NVS. Experiments show that WildRayZer consistently outperforms optimization-based and feed-forward baselines in both transient-region removal and full-frame NVS quality with a single feed-forward pass.

  </details>



- **Lunar-G2R: Geometry-to-Reflectance Learning for High-Fidelity Lunar BRDF Estimation**  
  Clementine Grethen, Nicolas Menga, Roland Brochard, Geraldine Morin, Simone Gasparini, Jeremy Lebreton, Manuel Sanchez Gestido  
  _2026-01-15_ · https://arxiv.org/abs/2601.10449v1  
  <details><summary>Abstract</summary>

  We address the problem of estimating realistic, spatially varying reflectance for complex planetary surfaces such as the lunar regolith, which is critical for high-fidelity rendering and vision-based navigation. Existing lunar rendering pipelines rely on simplified or spatially uniform BRDF models whose parameters are difficult to estimate and fail to capture local reflectance variations, limiting photometric realism. We propose Lunar-G2R, a geometry-to-reflectance learning framework that predicts spatially varying BRDF parameters directly from a lunar digital elevation model (DEM), without requiring multi-view imagery, controlled illumination, or dedicated reflectance-capture hardware at inference time. The method leverages a U-Net trained with differentiable rendering to minimize photometric discrepancies between real orbital images and physically based renderings under known viewing and illumination geometry. Experiments on a geographically held-out region of the Tycho crater show that our approach reduces photometric error by 38 % compared to a state-of-the-art baseline, while achieving higher PSNR and SSIM and improved perceptual similarity, capturing fine-scale reflectance variations absent from spatially uniform models. To our knowledge, this is the first method to infer a spatially varying reflectance model directly from terrain geometry.

  </details>



- **A$^2$TG: Adaptive Anisotropic Textured Gaussians for Efficient 3D Scene Representation**  
  Sheng-Chi Hsu, Ting-Yu Yen, Shih-Hsuan Hung, Hung-Kuo Chu  
  _2026-01-14_ · https://arxiv.org/abs/2601.09243v1  
  <details><summary>Abstract</summary>

  Gaussian Splatting has emerged as a powerful representation for high-quality, real-time 3D scene rendering. While recent works extend Gaussians with learnable textures to enrich visual appearance, existing approaches allocate a fixed square texture per primitive, leading to inefficient memory usage and limited adaptability to scene variability. In this paper, we introduce adaptive anisotropic textured Gaussians (A$^2$TG), a novel representation that generalizes textured Gaussians by equipping each primitive with an anisotropic texture. Our method employs a gradient-guided adaptive rule to jointly determine texture resolution and aspect ratio, enabling non-uniform, detail-aware allocation that aligns with the anisotropic nature of Gaussian splats. This design significantly improves texture efficiency, reducing memory consumption while enhancing image quality. Experiments on multiple benchmark datasets demonstrate that A TG consistently outperforms fixed-texture Gaussian Splatting methods, achieving comparable rendering fidelity with substantially lower memory requirements.

  </details>


