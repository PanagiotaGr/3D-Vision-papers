# NeRF & Neural Radiance Fields

_Updated: 2026-01-16 13:08 UTC_

Total papers shown: **3**


---

- **WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments**  
  Xuweiyi Chen, Wentao Zhou, Zezhou Cheng  
  _2026-01-15_ · https://arxiv.org/abs/2601.10716v1  
  <details><summary>Abstract</summary>

  We present WildRayZer, a self-supervised framework for novel view synthesis (NVS) in dynamic environments where both the camera and objects move. Dynamic content breaks the multi-view consistency that static NVS models rely on, leading to ghosting, hallucinated geometry, and unstable pose estimation. WildRayZer addresses this by performing an analysis-by-synthesis test: a camera-only static renderer explains rigid structure, and its residuals reveal transient regions. From these residuals, we construct pseudo motion masks, distill a motion estimator, and use it to mask input tokens and gate loss gradients so supervision focuses on cross-view background completion. To enable large-scale training and evaluation, we curate Dynamic RealEstate10K (D-RE10K), a real-world dataset of 15K casually captured dynamic sequences, and D-RE10K-iPhone, a paired transient and clean benchmark for sparse-view transient-aware NVS. Experiments show that WildRayZer consistently outperforms optimization-based and feed-forward baselines in both transient-region removal and full-frame NVS quality with a single feed-forward pass.

  </details>



- **Variable Basis Mapping for Real-Time Volumetric Visualization**  
  Qibiao Li, Yuxuan Wang, Youcheng Cai, Huangsheng Du, Ligang Liu  
  _2026-01-14_ · https://arxiv.org/abs/2601.09417v1  
  <details><summary>Abstract</summary>

  Real-time visualization of large-scale volumetric data remains challenging, as direct volume rendering and voxel-based methods suffer from prohibitively high computational cost. We propose Variable Basis Mapping (VBM), a framework that transforms volumetric fields into 3D Gaussian Splatting (3DGS) representations through wavelet-domain analysis. First, we precompute a compact Wavelet-to-Gaussian Transition Bank that provides optimal Gaussian surrogates for canonical wavelet atoms across multiple scales. Second, we perform analytical Gaussian construction that maps discrete wavelet coefficients directly to 3DGS parameters using a closed-form, mathematically principled rule. Finally, a lightweight image-space fine-tuning stage further refines the representation to improve rendering fidelity. Experiments on diverse datasets demonstrate that VBM significantly accelerates convergence and enhances rendering quality, enabling real-time volumetric visualization.

  </details>



- **RAVEN: Erasing Invisible Watermarks via Novel View Synthesis**  
  Fahad Shamshad, Nils Lukas, Karthik Nandakumar  
  _2026-01-13_ · https://arxiv.org/abs/2601.08832v1  
  <details><summary>Abstract</summary>

  Invisible watermarking has become a critical mechanism for authenticating AI-generated image content, with major platforms deploying watermarking schemes at scale. However, evaluating the vulnerability of these schemes against sophisticated removal attacks remains essential to assess their reliability and guide robust design. In this work, we expose a fundamental vulnerability in invisible watermarks by reformulating watermark removal as a view synthesis problem. Our key insight is that generating a perceptually consistent alternative view of the same semantic content, akin to re-observing a scene from a shifted perspective, naturally removes the embedded watermark while preserving visual fidelity. This reveals a critical gap: watermarks robust to pixel-space and frequency-domain attacks remain vulnerable to semantic-preserving viewpoint transformations. We introduce a zero-shot diffusion-based framework that applies controlled geometric transformations in latent space, augmented with view-guided correspondence attention to maintain structural consistency during reconstruction. Operating on frozen pre-trained models without detector access or watermark knowledge, our method achieves state-of-the-art watermark suppression across 15 watermarking methods--outperforming 14 baseline attacks while maintaining superior perceptual quality across multiple datasets.

  </details>


