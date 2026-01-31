# NeRF & Neural Radiance Fields

_Updated: 2026-01-31 06:56 UTC_

Total papers shown: **3**


---

- **From Implicit Ambiguity to Explicit Solidity: Diagnosing Interior Geometric Degradation in Neural Radiance Fields for Dense 3D Scene Understanding**  
  Jiangsan Zhao, Jakob Geipel, Kryzysztof Kusnierek  
  _2026-01-29_ · https://arxiv.org/abs/2601.21421v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRFs) have emerged as a powerful paradigm for multi-view reconstruction, complementing classical photogrammetric pipelines based on Structure-from-Motion (SfM) and Multi-View Stereo (MVS). However, their reliability for quantitative 3D analysis in dense, self-occluding scenes remains poorly understood. In this study, we identify a fundamental failure mode of implicit density fields under heavy occlusion, which we term Interior Geometric Degradation (IGD). We show that transmittance-based volumetric optimization satisfies photometric supervision by reconstructing hollow or fragmented structures rather than solid interiors, leading to systematic instance undercounting. Through controlled experiments on synthetic datasets with increasing occlusion, we demonstrate that state-of-the-art mask-supervised NeRFs saturate at approximately 89% instance recovery in dense scenes, despite improved surface coherence and mask quality. To overcome this limitation, we introduce an explicit geometric pipeline based on Sparse Voxel Rasterization (SVRaster), initialized from SfM feature geometry. By projecting 2D instance masks onto an explicit voxel grid and enforcing geometric separation via recursive splitting, our approach preserves physical solidity and achieves a 95.8% recovery rate in dense clusters. A sensitivity analysis using degraded segmentation masks further shows that explicit SfM-based geometry is substantially more robust to supervision failure, recovering 43% more instances than implicit baselines. These results demonstrate that explicit geometric priors are a prerequisite for reliable quantitative analysis in highly self-occluding 3D scenes.

  </details>



- **Lightweight High-Fidelity Low-Bitrate Talking Face Compression for 3D Video Conference**  
  Jianglong Li, Jun Xu, Bingcong Lu, Zhengxue Cheng, Hongwei Hu, Ronghua Wu, Li Song  
  _2026-01-29_ · https://arxiv.org/abs/2601.21269v1  
  <details><summary>Abstract</summary>

  The demand for immersive and interactive communication has driven advancements in 3D video conferencing, yet achieving high-fidelity 3D talking face representation at low bitrates remains a challenge. Traditional 2D video compression techniques fail to preserve fine-grained geometric and appearance details, while implicit neural rendering methods like NeRF suffer from prohibitive computational costs. To address these challenges, we propose a lightweight, high-fidelity, low-bitrate 3D talking face compression framework that integrates FLAME-based parametric modeling with 3DGS neural rendering. Our approach transmits only essential facial metadata in real time, enabling efficient reconstruction with a Gaussian-based head model. Additionally, we introduce a compact representation and compression scheme, including Gaussian attribute compression and MLP optimization, to enhance transmission efficiency. Experimental results demonstrate that our method achieves superior rate-distortion performance, delivering high-quality facial rendering at extremely low bitrates, making it well-suited for real-time 3D video conferencing applications.

  </details>



- **FreeFix: Boosting 3D Gaussian Splatting via Fine-Tuning-Free Diffusion Models**  
  Hongyu Zhou, Zisen Shao, Sheng Miao, Pan Wang, Dongfeng Bai, Bingbing Liu, Yiyi Liao  
  _2026-01-28_ · https://arxiv.org/abs/2601.20857v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields and 3D Gaussian Splatting have advanced novel view synthesis, yet still rely on dense inputs and often degrade at extrapolated views. Recent approaches leverage generative models, such as diffusion models, to provide additional supervision, but face a trade-off between generalization and fidelity: fine-tuning diffusion models for artifact removal improves fidelity but risks overfitting, while fine-tuning-free methods preserve generalization but often yield lower fidelity. We introduce FreeFix, a fine-tuning-free approach that pushes the boundary of this trade-off by enhancing extrapolated rendering with pretrained image diffusion models. We present an interleaved 2D-3D refinement strategy, showing that image diffusion models can be leveraged for consistent refinement without relying on costly video diffusion models. Furthermore, we take a closer look at the guidance signal for 2D refinement and propose a per-pixel confidence mask to identify uncertain regions for targeted improvement. Experiments across multiple datasets show that FreeFix improves multi-frame consistency and achieves performance comparable to or surpassing fine-tuning-based methods, while retaining strong generalization ability.

  </details>


