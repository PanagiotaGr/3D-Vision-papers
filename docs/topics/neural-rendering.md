# Neural Rendering & View Synthesis

_Updated: 2026-02-24 07:14 UTC_

Total papers shown: **5**


---

- **tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction**  
  Chen Wang, Hao Tan, Wang Yifan, Zhiqin Chen, Yuheng Liu, Kalyan Sunkavalli, Sai Bi, Lingjie Liu, Yiwei Hu  
  _2026-02-23_ · https://arxiv.org/abs/2602.20160v1  
  <details><summary>Abstract</summary>

  We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model's capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.

  </details>



- **SemanticNVS: Improving Semantic Scene Understanding in Generative Novel View Synthesis**  
  Xinya Chen, Christopher Wewer, Jiahao Xie, Xinting Hu, Jan Eric Lenssen  
  _2026-02-23_ · https://arxiv.org/abs/2602.20079v1  
  <details><summary>Abstract</summary>

  We present SemanticNVS, a camera-conditioned multi-view diffusion model for novel view synthesis (NVS), which improves generation quality and consistency by integrating pre-trained semantic feature extractors. Existing NVS methods perform well for views near the input view, however, they tend to generate semantically implausible and distorted images under long-range camera motion, revealing severe degradation. We speculate that this degradation is due to current models failing to fully understand their conditioning or intermediate generated scene content. Here, we propose to integrate pre-trained semantic feature extractors to incorporate stronger scene semantics as conditioning to achieve high-quality generation even at distant viewpoints. We investigate two different strategies, (1) warped semantic features and (2) an alternating scheme of understanding and generation at each denoising step. Experimental results on multiple datasets demonstrate the clear qualitative and quantitative (4.69%-15.26% in FID) improvement over state-of-the-art alternatives.

  </details>



- **Learning Positive-Incentive Point Sampling in Neural Implicit Fields for Object Pose Estimation**  
  Yifei Shi, Boyan Wan, Xin Xu, Kai Xu  
  _2026-02-23_ · https://arxiv.org/abs/2602.19937v1  
  <details><summary>Abstract</summary>

  Learning neural implicit fields of 3D shapes is a rapidly emerging field that enables shape representation at arbitrary resolutions. Due to the flexibility, neural implicit fields have succeeded in many research areas, including shape reconstruction, novel view image synthesis, and more recently, object pose estimation. Neural implicit fields enable learning dense correspondences between the camera space and the object's canonical space-including unobserved regions in camera space-significantly boosting object pose estimation performance in challenging scenarios like highly occluded objects and novel shapes. Despite progress, predicting canonical coordinates for unobserved camera-space regions remains challenging due to the lack of direct observational signals. This necessitates heavy reliance on the model's generalization ability, resulting in high uncertainty. Consequently, densely sampling points across the entire camera space may yield inaccurate estimations that hinder the learning process and compromise performance. To alleviate this problem, we propose a method combining an SO(3)-equivariant convolutional implicit network and a positive-incentive point sampling (PIPS) strategy. The SO(3)-equivariant convolutional implicit network estimates point-level attributes with SO(3)-equivariance at arbitrary query locations, demonstrating superior performance compared to most existing baselines. The PIPS strategy dynamically determines sampling locations based on the input, thereby boosting the network's accuracy and training efficiency. Our method outperforms the state-of-the-art on three pose estimation datasets. Notably, it demonstrates significant improvements in challenging scenarios, such as objects captured with unseen pose, high occlusion, novel geometry, and severe noise.

  </details>



- **One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image**  
  Pengfei Wang, Liyi Chen, Zhiyuan Ma, Yanjun Guo, Guowen Zhang, Lei Zhang  
  _2026-02-23_ · https://arxiv.org/abs/2602.19766v1  
  <details><summary>Abstract</summary>

  Generating explorable 3D scenes from a single image is a highly challenging problem in 3D vision. Existing methods struggle to support free exploration, often producing severe geometric distortions and noisy artifacts when the viewpoint moves far from the original perspective. We introduce \textbf{One2Scene}, an effective framework that decomposes this ill-posed problem into three tractable sub-tasks to enable immersive explorable scene generation. We first use a panorama generator to produce anchor views from a single input image as initialization. Then, we lift these 2D anchors into an explicit 3D geometric scaffold via a generalizable, feed-forward Gaussian Splatting network. Instead of treating the panorama as a single image for reconstruction, we project it into multiple sparse anchor views and reformulate the reconstruction task as multi-view stereo matching, which allows us to leverage robust geometric priors learned from large-scale multi-view datasets. A bidirectional feature fusion module is used to enforce cross-view consistency, yielding an efficient and geometrically reliable scaffold. Finally, the scaffold serves as a strong prior for a novel view generator to produce photorealistic and geometrically accurate views at arbitrary cameras. By explicitly conditioning on a 3D-consistent scaffold to perform reconstruction, One2Scene works stably under large camera motions, supporting immersive scene exploration. Extensive experiments show that One2Scene substantially outperforms state-of-the-art methods in panorama depth estimation, feed-forward 360° reconstruction, and explorable 3D scene generation. Code and models will be released.

  </details>



- **SceneTok: A Compressed, Diffusable Token Space for 3D Scenes**  
  Mohammad Asim, Christopher Wewer, Jan Eric Lenssen  
  _2026-02-21_ · https://arxiv.org/abs/2602.18882v1  
  <details><summary>Abstract</summary>

  We present SceneTok, a novel tokenizer for encoding view sets of scenes into a compressed and diffusable set of unstructured tokens. Existing approaches for 3D scene representation and generation commonly use 3D data structures or view-aligned fields. In contrast, we introduce the first method that encodes scene information into a small set of permutation-invariant tokens that is disentangled from the spatial grid. The scene tokens are predicted by a multi-view tokenizer given many context views and rendered into novel views by employing a light-weight rectified flow decoder. We show that the compression is 1-3 orders of magnitude stronger than for other representations while still reaching state-of-the-art reconstruction quality. Further, our representation can be rendered from novel trajectories, including ones deviating from the input trajectory, and we show that the decoder gracefully handles uncertainty. Finally, the highly-compressed set of unstructured latent scene tokens enables simple and efficient scene generation in 5 seconds, achieving a much better quality-speed trade-off than previous paradigms.

  </details>


