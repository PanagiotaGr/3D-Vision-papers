# Neural Rendering & View Synthesis

_Updated: 2026-01-21 06:54 UTC_

Total papers shown: **4**


---

- **One-Shot Refiner: Boosting Feed-forward Novel View Synthesis via One-Step Diffusion**  
  Yitong Dong, Qi Zhang, Minchao Jiang, Zhiqiang Wu, Qingnan Fan, Ying Feng, Huaqi Zhang, Hujun Bao, Guofeng Zhang  
  _2026-01-20_ · https://arxiv.org/abs/2601.14161v1  
  <details><summary>Abstract</summary>

  We present a novel framework for high-fidelity novel view synthesis (NVS) from sparse images, addressing key limitations in recent feed-forward 3D Gaussian Splatting (3DGS) methods built on Vision Transformer (ViT) backbones. While ViT-based pipelines offer strong geometric priors, they are often constrained by low-resolution inputs due to computational costs. Moreover, existing generative enhancement methods tend to be 3D-agnostic, resulting in inconsistent structures across views, especially in unseen regions. To overcome these challenges, we design a Dual-Domain Detail Perception Module, which enables handling high-resolution images without being limited by the ViT backbone, and endows Gaussians with additional features to store high-frequency details. We develop a feature-guided diffusion network, which can preserve high-frequency details during the restoration process. We introduce a unified training strategy that enables joint optimization of the ViT-based geometric backbone and the diffusion-based refinement module. Experiments demonstrate that our method can maintain superior generation quality across multiple datasets.

  </details>



- **ParkingTwin: Training-Free Streaming 3D Reconstruction for Parking-Lot Digital Twins**  
  Xinhao Liu, Yu Wang, Xiansheng Guo, Gordon Owusu Boateng, Yu Cao, Haonan Si, Xingchen Guo, Nirwan Ansari  
  _2026-01-20_ · https://arxiv.org/abs/2601.13706v1  
  <details><summary>Abstract</summary>

  High-fidelity parking-lot digital twins provide essential priors for path planning, collision checking, and perception validation in Automated Valet Parking (AVP). Yet robot-oriented reconstruction faces a trilemma: sparse forward-facing views cause weak parallax and ill-posed geometry; dynamic occlusions and extreme lighting hinder stable texture fusion; and neural rendering typically needs expensive offline optimization, violating edge-side streaming constraints. We propose ParkingTwin, a training-free, lightweight system for online streaming 3D reconstruction. First, OSM-prior-driven geometric construction uses OpenStreetMap semantic topology to directly generate a metric-consistent TSDF, replacing blind geometric search with deterministic mapping and avoiding costly optimization. Second, geometry-aware dynamic filtering employs a quad-modal constraint field (normal/height/depth consistency) to reject moving vehicles and transient occlusions in real time. Third, illumination-robust fusion in CIELAB decouples luminance and chromaticity via adaptive L-channel weighting and depth-gradient suppression, reducing seams under abrupt lighting changes. ParkingTwin runs at 30+ FPS on an entry-level GTX 1660. On a 68,000 m^2 real-world dataset, it achieves SSIM 0.87 (+16.0%), delivers about 15x end-to-end speedup, and reduces GPU memory by 83.3% compared with state-of-the-art 3D Gaussian Splatting (3DGS) that typically requires high-end GPUs (RTX 4090D). The system outputs explicit triangle meshes compatible with Unity/Unreal digital-twin pipelines. Project page: https://mihoutao-liu.github.io/ParkingTwin/

  </details>



- **GaussExplorer: 3D Gaussian Splatting for Embodied Exploration and Reasoning**  
  Kim Yu-Ji, Dahye Lee, Kim Jun-Seong, GeonU Kim, Nam Hyeon-Woo, Yongjin Kwon, Yu-Chiang Frank Wang, Jaesung Choe, Tae-Hyun Oh  
  _2026-01-19_ · https://arxiv.org/abs/2601.13132v1  
  <details><summary>Abstract</summary>

  We present GaussExplorer, a framework for embodied exploration and reasoning built on 3D Gaussian Splatting (3DGS). While prior approaches to language-embedded 3DGS have made meaningful progress in aligning simple text queries with Gaussian embeddings, they are generally optimized for relatively simple queries and struggle to interpret more complex, compositional language queries. Alternative studies based on object-centric RGB-D structured memories provide spatial grounding but are constrained by pre-fixed viewpoints. To address these issues, GaussExplorer introduces Vision-Language Models (VLMs) on top of 3DGS to enable question-driven exploration and reasoning within 3D scenes. We first identify pre-captured images that are most correlated with the query question, and subsequently adjust them into novel viewpoints to more accurately capture visual information for better reasoning by VLMs. Experiments show that ours outperforms existing methods on several benchmarks, demonstrating the effectiveness of integrating VLM-based reasoning with 3DGS for embodied tasks.

  </details>



- **CSGaussian: Progressive Rate-Distortion Compression and Segmentation for 3D Gaussian Splatting**  
  Yu-Jen Tseng, Chia-Hao Kao, Jing-Zhong Chen, Alessandro Gnutti, Shao-Yuan Lo, Yen-Yu Lin, Wen-Hsiao Peng  
  _2026-01-19_ · https://arxiv.org/abs/2601.12814v1  
  <details><summary>Abstract</summary>

  We present the first unified framework for rate-distortion-optimized compression and segmentation of 3D Gaussian Splatting (3DGS). While 3DGS has proven effective for both real-time rendering and semantic scene understanding, prior works have largely treated these tasks independently, leaving their joint consideration unexplored. Inspired by recent advances in rate-distortion-optimized 3DGS compression, this work integrates semantic learning into the compression pipeline to support decoder-side applications--such as scene editing and manipulation--that extend beyond traditional scene reconstruction and view synthesis. Our scheme features a lightweight implicit neural representation-based hyperprior, enabling efficient entropy coding of both color and semantic attributes while avoiding costly grid-based hyperprior as seen in many prior works. To facilitate compression and segmentation, we further develop compression-guided segmentation learning, consisting of quantization-aware training to enhance feature separability and a quality-aware weighting mechanism to suppress unreliable Gaussian primitives. Extensive experiments on the LERF and 3D-OVS datasets demonstrate that our approach significantly reduces transmission cost while preserving high rendering quality and strong segmentation performance.

  </details>


