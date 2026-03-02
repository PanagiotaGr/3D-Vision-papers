# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **2**


---

- **UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images**  
  Junhwa Hur, Charles Herrmann, Songyou Peng, Philipp Henzler, Zeyu Ma, Todd Zickler, Deqing Sun  
  _2026-02-27_ · https://arxiv.org/abs/2602.24290v1  
  <details><summary>Abstract</summary>

  Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/

  </details>



- **Spatio-Temporal Garment Reconstruction Using Diffusion Mapping via Pattern Coordinates**  
  Yingxuan You, Ren Li, Corentin Dumery, Cong Cao, Hao Li, Pascal Fua  
  _2026-02-27_ · https://arxiv.org/abs/2602.24043v1  
  <details><summary>Abstract</summary>

  Reconstructing 3D clothed humans from monocular images and videos is a fundamental problem with applications in virtual try-on, avatar creation, and mixed reality. Despite significant progress in human body recovery, accurately reconstructing garment geometry, particularly for loose-fitting clothing, remains an open challenge. We propose a unified framework for high-fidelity 3D garment reconstruction from both single images and video sequences. Our approach combines Implicit Sewing Patterns (ISP) with a generative diffusion model to learn expressive garment shape priors in 2D UV space. Leveraging these priors, we introduce a mapping model that establishes correspondences between image pixels, UV pattern coordinates, and 3D geometry, enabling accurate and detailed garment reconstruction from single images. We further extend this formulation to dynamic reconstruction by introducing a spatio-temporal diffusion scheme with test-time guidance to enforce long-range temporal consistency. We also develop analytic projection-based constraints that preserve image-aligned geometry in visible regions while enforcing coherent completion in occluded areas over time. Although trained exclusively on synthetically simulated cloth data, our method generalizes well to real-world imagery and consistently outperforms existing approaches on both tight- and loose-fitting garments. The reconstructed garments preserve fine geometric detail while exhibiting realistic dynamic motion, supporting downstream applications such as texture editing, garment retargeting, and animation.

  </details>


