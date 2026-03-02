# Gaussian Splatting & 3DGS

_Updated: 2026-03-02 07:14 UTC_

Total papers shown: **5**


---

- **UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images**  
  Junhwa Hur, Charles Herrmann, Songyou Peng, Philipp Henzler, Zeyu Ma, Todd Zickler, Deqing Sun  
  _2026-02-27_ · https://arxiv.org/abs/2602.24290v1  
  <details><summary>Abstract</summary>

  Dense 4D reconstruction from unposed images remains a critical challenge, with current methods relying on slow test-time optimization or fragmented, task-specific feedforward models. We introduce UFO-4D, a unified feedforward framework to reconstruct a dense, explicit 4D representation from just a pair of unposed images. UFO-4D directly estimates dynamic 3D Gaussian Splats, enabling the joint and consistent estimation of 3D geometry, 3D motion, and camera pose in a feedforward manner. Our core insight is that differentiably rendering multiple signals from a single Dynamic 3D Gaussian representation offers major training advantages. This approach enables a self-supervised image synthesis loss while tightly coupling appearance, depth, and motion. Since all modalities share the same geometric primitives, supervising one inherently regularizes and improves the others. This synergy overcomes data scarcity, allowing UFO-4D to outperform prior work by up to 3 times in joint geometry, motion, and camera pose estimation. Our representation also enables high-fidelity 4D interpolation across novel views and time. Please visit our project page for visual results: https://ufo-4d.github.io/

  </details>



- **GeoDiff4D: Geometry-Aware Diffusion for 4D Head Avatar Reconstruction**  
  Chao Xu, Xiaochen Zhao, Xiang Deng, Jingxiang Sun, Zhuo Su, Donglin Di, Yebin Liu  
  _2026-02-27_ · https://arxiv.org/abs/2602.24161v1  
  <details><summary>Abstract</summary>

  Reconstructing photorealistic and animatable 4D head avatars from a single portrait image remains a fundamental challenge in computer vision. While diffusion models have enabled remarkable progress in image and video generation for avatar reconstruction, existing methods primarily rely on 2D priors and struggle to achieve consistent 3D geometry. We propose a novel framework that leverages geometry-aware diffusion to learn strong geometry priors for high-fidelity head avatar reconstruction. Our approach jointly synthesizes portrait images and corresponding surface normals, while a pose-free expression encoder captures implicit expression representations. Both synthesized images and expression latents are incorporated into 3D Gaussian-based avatars, enabling photorealistic rendering with accurate geometry. Extensive experiments demonstrate that our method substantially outperforms state-of-the-art approaches in visual quality, expression fidelity, and cross-identity generalization, while supporting real-time rendering.

  </details>



- **Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives**  
  Haoran Wang, Guoxi Huang, Fan Zhang, David Bull, Nantheera Anantrasirichai  
  _2026-02-27_ · https://arxiv.org/abs/2602.24136v1  
  <details><summary>Abstract</summary>

  Recent significant advances in 3D scene representation have been driven by 3D Gaussian Splatting (3DGS), which has enabled real-time rendering with photorealistic quality. 3DGS often requires a large number of primitives to achieve high fidelity, leading to redundant representations and high resource consumption, thereby limiting its scalability for complex or large-scale scenes. Consequently, effective pruning strategies and more expressive primitives that can reduce redundancy while preserving visual quality are crucial for practical deployment. We propose an efficient, integrated reconstruction-aware pruning strategy that adaptively determines pruning timing and refining intervals based on reconstruction quality, thus reducing model size while enhancing rendering quality. Moreover, we introduce a 3D Difference-of-Gaussians primitive that jointly models both positive and negative densities in a single primitive, improving the expressiveness of Gaussians under compact configurations. Our method significantly improves model compactness, achieving up to 90\% reduction in Gaussian-count while delivering visual quality that is similar to, or in some cases better than, that produced by state-of-the-art methods. Code will be made publicly available.

  </details>



- **DiffusionHarmonizer: Bridging Neural Reconstruction and Photorealistic Simulation with Online Diffusion Enhancer**  
  Yuxuan Zhang, Katarína Tóthová, Zian Wang, Kangxue Yin, Haithem Turki, Riccardo de Lutio, Yen-Yu Chang, Or Litany, Sanja Fidler, Zan Gojcic  
  _2026-02-27_ · https://arxiv.org/abs/2602.24096v1  
  <details><summary>Abstract</summary>

  Simulation is essential to the development and evaluation of autonomous robots such as self-driving vehicles. Neural reconstruction is emerging as a promising solution as it enables simulating a wide variety of scenarios from real-world data alone in an automated and scalable way. However, while methods such as NeRF and 3D Gaussian Splatting can produce visually compelling results, they often exhibit artifacts particularly when rendering novel views, and fail to realistically integrate inserted dynamic objects, especially when they were captured from different scenes. To overcome these limitations, we introduce DiffusionHarmonizer, an online generative enhancement framework that transforms renderings from such imperfect scenes into temporally consistent outputs while improving their realism. At its core is a single-step temporally-conditioned enhancer that is converted from a pretrained multi-step image diffusion model, capable of running in online simulators on a single GPU. The key to training it effectively is a custom data curation pipeline that constructs synthetic-real pairs emphasizing appearance harmonization, artifact correction, and lighting realism. The result is a scalable system that significantly elevates simulation fidelity in both research and production environments.

  </details>



- **SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting**  
  Xiang Feng, Xiangbo Wang, Tieshi Zhong, Chengkai Wang, Yiting Zhao, Tianxiang Xu, Zhenzhong Kuang, Feiwei Qin, Xuefei Yin, Yanming Zhu  
  _2026-02-27_ · https://arxiv.org/abs/2602.24020v1  
  <details><summary>Abstract</summary>

  3D super-resolution (3DSR) aims to reconstruct high-resolution (HR) 3D scenes from low-resolution (LR) multi-view images. Existing methods rely on dense LR inputs and per-scene optimization, which restricts the high-frequency priors for constructing HR 3D Gaussian Splatting (3DGS) to those inherited from pretrained 2D super-resolution (2DSR) models. This severely limits reconstruction fidelity, cross-scene generalization, and real-time usability. We propose to reformulate 3DSR as a direct feed-forward mapping from sparse LR views to HR 3DGS representations, enabling the model to autonomously learn 3D-specific high-frequency geometry and appearance from large-scale, multi-scene data. This fundamentally changes how 3DSR acquires high-frequency knowledge and enables robust generalization to unseen scenes. Specifically, we introduce SR3R, a feed-forward framework that directly predicts HR 3DGS representations from sparse LR views via the learned mapping network. To further enhance reconstruction fidelity, we introduce Gaussian offset learning and feature refinement, which stabilize reconstruction and sharpen high-frequency details. SR3R is plug-and-play and can be paired with any feed-forward 3DGS reconstruction backbone: the backbone provides an LR 3DGS scaffold, and SR3R upscales it to an HR 3DGS. Extensive experiments across three 3D benchmarks demonstrate that SR3R surpasses state-of-the-art (SOTA) 3DSR methods and achieves strong zero-shot generalization, even outperforming SOTA per-scene optimization methods on unseen scenes.

  </details>


