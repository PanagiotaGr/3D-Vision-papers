# Neural Rendering & View Synthesis

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **3**


---

- **UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images**  
  Junhwa Hur, Charles Herrmann, Songyou Peng, Philipp Henzler, Zeyu Ma, Todd Zickler, Deqing Sun  
  _2026-02-27_ · https://arxiv.org/abs/2602.24290v1  
  <details><summary>Abstract</summary>

  Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/

  </details>



- **HumanOrbit: 3D Human Reconstruction as 360° Orbit Generation**  
  Keito Suzuki, Kunyao Chen, Lei Wang, Bang Du, Runfa Blark Li, Peng Liu, Ning Bi, Truong Nguyen  
  _2026-02-27_ · https://arxiv.org/abs/2602.24148v1  
  <details><summary>Abstract</summary>

  We present a method for generating a full 360° orbit video around a person from a single input image. Existing methods typically adapt image-based diffusion models for multi-view synthesis, but yield inconsistent results across views and with the original identity. In contrast, recent video diffusion models have demonstrated their ability in generating photorealistic results that align well with the given prompts. Inspired by these results, we propose HumanOrbit, a video diffusion model for multi-view human image generation. Our approach enables the model to synthesize continuous camera rotations around the subject, producing geometrically consistent novel views while preserving the appearance and identity of the person. Using the generated multi-view frames, we further propose a reconstruction pipeline that recovers a textured mesh of the subject. Experimental results validate the effectiveness of HumanOrbit for multi-view image generation and that the reconstructed 3D models exhibit superior completeness and fidelity compared to those from state-of-the-art baselines.

  </details>



- **DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer**  
  Yuxuan Zhang, Katarína Tóthová, Zian Wang, Kangxue Yin, Haithem Turki, Riccardo de Lutio, Yen-Yu Chang, Or Litany, Sanja Fidler, Zan Gojcic  
  _2026-02-27_ · https://arxiv.org/abs/2602.24096v1  
  <details><summary>Abstract</summary>

  Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.

  </details>


