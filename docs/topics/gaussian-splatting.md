# Gaussian Splatting & 3DGS

_Updated: 2026-02-24 07:14 UTC_

Total papers shown: **7**


---

- **tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction**  
  Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, Yiwei Hu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20160v1  
  <details><summary>Abstract</summary>

  We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.

  </details>



- **Augmented Radiance Field: A General Framework for Enhanced Gaussian Splatting**  
  Yixin Yang, Bojian Wu, Yang Zhou, Hui Huang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19916v1  
  <details><summary>Abstract</summary>

  Due to the real-time rendering performance, 3D Gaussian Splatting (3DGS) has emerged as the leading method for radiance field reconstruction. However, its reliance on spherical harmonics for color encoding inherently limits its ability to separate diffuse and specular components, making it challenging to accurately represent complex reflections. To address this, we propose a novel enhanced Gaussian kernel that explicitly models specular effects through view-dependent opacity. Meanwhile, we introduce an error-driven compensation strategy to improve rendering quality in existing 3DGS scenes. Our method begins with 2D Gaussian initialization and then adaptively inserts and optimizes enhanced Gaussian kernels, ultimately producing an augmented radiance field. Experiments demonstrate that our method not only surpasses state-of-the-art NeRF methods in rendering performance but also achieves greater parameter efficiency. Project page at: https://xiaoxinyyx.github.io/augs.

  </details>



- **One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image**  
  Pengfei Wang, Liyi Chen, Zhiyuan Ma, Yanjun Guo, Guowen Zhang, Lei Zhang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19766v1  
  <details><summary>Abstract</summary>

  Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Code and models will be released.

  </details>



- **RAP: Fast Feedforward Rendering-Free Attribute-Guided Primitive Importance Score Prediction for Efficient 3D Gaussian Splatting Processing**  
  Kaifa Yang, Qi Yang, Yiling Xu, Zhu Li  
  _2026-02-23_ · https://arxiv.org/abs/2602.19753v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a leading technology for high-quality 3D scene reconstruction. However, the iterative refinement and densification process leads to the generation of a large number of primitives, each contributing to the reconstruction to a substantially different extent. Estimating primitive importance is thus crucial, both for removing redundancy during reconstruction and for enabling efficient compression and transmission. Existing methods typically rely on rendering-based analyses, where each primitive is evaluated through its contribution across multiple camera viewpoints. However, such methods are sensitive to the number and selection of views, rely on specialized differentiable rasterizers, and have long calculation times that grow linearly with view count, making them difficult to integrate as plug-and-play modules and limiting scalability and generalization. To address these issues, we propose RAP, a fast feedforward rendering-free attribute-guided method for efficient importance score prediction in 3DGS. RAP infers primitive significance directly from intrinsic Gaussian attributes and local neighborhood statistics, avoiding rendering-based or visibility-dependent computations. A compact MLP predicts per-primitive importance scores using rendering loss, pruning-aware loss, and significance distribution regularization. After training on a small set of scenes, RAP generalizes effectively to unseen data and can be seamlessly integrated into reconstruction, compression, and transmission pipelines. Our code is publicly available at https://github.com/yyyykf/RAP.

  </details>



- **DefenseSplat: Enhancing the Robustness of 3D Gaussian Splatting via Frequency-Aware Filtering**  
  Yiran Qiao, Yiren Lu, Yunlai Zhou, Rui Yang, Linlin Hou, Yu Yin, Jing Ma  
  _2026-02-22_ · https://arxiv.org/abs/2602.19323v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has emerged as a powerful paradigm for real-time and high-fidelity 3D reconstruction from posed images. However, recent studies reveal its vulnerability to adversarial corruptions in input views, where imperceptible yet consistent perturbations can drastically degrade rendering quality, increase training and rendering time, and inflate memory usage, even leading to server denial-of-service. In our work, to mitigate this issue, we begin by analyzing the distinct behaviors of adversarial perturbations in the low- and high-frequency components of input images using wavelet transforms. Based on this observation, we design a simple yet effective frequency-aware defense strategy that reconstructs training views by filtering high-frequency noise while preserving low-frequency content. This approach effectively suppresses adversarial artifacts while maintaining the authenticity of the original scene. Notably, it does not significantly impair training on clean data, achieving a desirable trade-off between robustness and performance on clean inputs. Through extensive experiments under a wide range of attack intensities on multiple benchmarks, we demonstrate that our method substantially enhances the robustness of 3DGS without access to clean ground-truth supervision. By highlighting and addressing the overlooked vulnerabilities of 3D Gaussian Splatting, our work paves the way for more robust and secure 3D reconstructions.

  </details>



- **PhysConvex: Physics-Informed 3D Dynamic Convex Radiance Fields for Reconstruction and Simulation**  
  Dan Wang, Xinrui Cui, Serge Belongie, Ravi Ramamoorthi  
  _2026-02-21_ · https://arxiv.org/abs/2602.18886v1  
  <details><summary>Abstract</summary>

  Reconstructing and simulating dynamic 3D scenes with both visual realism and physical consistency remains a fundamental challenge. Existing neural representations, such as NeRFs and 3DGS, excel in appearance reconstruction but struggle to capture complex material deformation and dynamics. We propose PhysConvex, a Physics-informed 3D Dynamic Convex Radiance Field that unifies visual rendering and physical simulation. PhysConvex represents deformable radiance fields using physically grounded convex primitives governed by continuum mechanics. We introduce a boundary-driven dynamic convex representation that models deformation through vertex and surface dynamics, capturing spatially adaptive, non-uniform deformation, and evolving boundaries. To efficiently simulate complex geometries and heterogeneous materials, we further develop a reduced-order convex simulation that advects dynamic convex fields using neural skinning eigenmodes as shape- and material-aware deformation bases with time-varying reduced DOFs under Newtonian dynamics. Convex dynamics also offers compact, gap-free volumetric coverage, enhancing both geometric efficiency and simulation fidelity. Experiments demonstrate that PhysConvex achieves high-fidelity reconstruction of geometry, appearance, and physical properties from videos, outperforming existing methods.

  </details>



- **Spatial-Temporal State Propagation Autoregressive Model for 4D Object Generation**  
  Liying Yang, Jialun Liu, Jiakui Hu, Chenhao Guan, Haibin Huang, Fangqiu Yi, Chi Zhang, Yanyan Liang  
  _2026-02-21_ · https://arxiv.org/abs/2602.18830v1  
  <details><summary>Abstract</summary>

  Generating high-quality 4D objects with spatial-temporal consistency is still formidable. Existing diffusion-based methods often struggle with spatial-temporal inconsistency, as they fail to leverage outputs from all previous timesteps to guide the generation at the current timestep. Therefore, we propose a Spatial-Temporal State Propagation AutoRegressive Model (4DSTAR), which generates 4D objects maintaining temporal-spatial consistency. 4DSTAR formulates the generation problem as the prediction of tokens that represent the 4D object. It consists of two key components: (1) The dynamic spatial-temporal state propagation autoregressive model (STAR) is proposed, which achieves spatial-temporal consistent generation. Unlike standard autoregressive models, STAR divides prediction tokens into groups based on timesteps. It models long-term dependencies by propagating spatial-temporal states from previous groups and utilizes these dependencies to guide generation at the next timestep. To this end, a spatial-temporal container is proposed, which dynamically updating the effective spatial-temporal state features from all historical groups, then updated features serve as conditional features to guide the prediction of the next token group. (2) The 4D VQ-VAE is proposed, which implicitly encodes the 4D structure into discrete space and decodes the discrete tokens predicted by STAR into temporally coherent dynamic 3D Gaussians. Experiments demonstrate that 4DSTAR generates spatial-temporal consistent 4D objects, and achieves performance competitive with diffusion models.

  </details>


