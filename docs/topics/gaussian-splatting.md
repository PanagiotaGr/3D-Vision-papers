# Gaussian Splatting & 3DGS

_Updated: 2026-01-27 06:53 UTC_

Total papers shown: **5**


---

- **Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting**  
  Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  _2026-01-26_ · https://arxiv.org/abs/2601.18633v1  
  <details><summary>Abstract</summary>

  Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works. Our project code and supplementary documents are public available at https://github.com/stonewalking/Splat-portrait.

  </details>



- **LoD-Structured 3D Gaussian Splatting for Streaming Video Reconstruction**  
  Xinhui Liu, Can Wang, Lei Liu, Zhenghao Chen, Wei Jiang, Wei Wang, Dong Xu  
  _2026-01-26_ · https://arxiv.org/abs/2601.18475v1  
  <details><summary>Abstract</summary>

  Free-Viewpoint Video (FVV) reconstruction enables photorealistic and interactive 3D scene visualization; however, real-time streaming is often bottlenecked by sparse-view inputs, prohibitive training costs, and bandwidth constraints. While recent 3D Gaussian Splatting (3DGS) has advanced FVV due to its superior rendering speed, Streaming Free-Viewpoint Video (SFVV) introduces additional demands for rapid optimization, high-fidelity reconstruction under sparse constraints, and minimal storage footprints. To bridge this gap, we propose StreamLoD-GS, an LoD-based Gaussian Splatting framework designed specifically for SFVV. Our approach integrates three core innovations: 1) an Anchor- and Octree-based LoD-structured 3DGS with a hierarchical Gaussian dropout technique to ensure efficient and stable optimization while maintaining high-quality rendering; 2) a GMM-based motion partitioning mechanism that separates dynamic and static content, refining dynamic regions while preserving background stability; and 3) a quantized residual refinement framework that significantly reduces storage requirements without compromising visual fidelity. Extensive experiments demonstrate that StreamLoD-GS achieves competitive or state-of-the-art performance in terms of quality, efficiency, and storage.

  </details>



- **Geometry-Grounded Gaussian Splatting**  
  Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, Ping Tan  
  _2026-01-25_ · https://arxiv.org/abs/2601.17835v1  
  <details><summary>Abstract</summary>

  Gaussian Splatting (GS) has demonstrated impressive quality and efficiency in novel view synthesis. However, shape extraction from Gaussian primitives remains an open problem. Due to inadequate geometry parameterization and approximation, existing shape reconstruction methods suffer from poor multi-view consistency and are sensitive to floaters. In this paper, we present a rigorous theoretical derivation that establishes Gaussian primitives as a specific type of stochastic solids. This theoretical framework provides a principled foundation for Geometry-Grounded Gaussian Splatting by enabling the direct treatment of Gaussian primitives as explicit geometric representations. Using the volumetric nature of stochastic solids, our method efficiently renders high-quality depth maps for fine-grained geometry extraction. Experiments show that our method achieves the best shape reconstruction results among all Gaussian Splatting-based methods on public datasets.

  </details>



- **Advancing Structured Priors for Sparse-Voxel Surface Reconstruction**  
  Ting-Hsun Chi, Chu-Rong Chen, Chi-Tun Hsu, Hsuan-Ting Lin, Sheng-Yu Huang, Cheng Sun, Yu-Chiang Frank Wang  
  _2026-01-25_ · https://arxiv.org/abs/2601.17720v1  
  <details><summary>Abstract</summary>

  Reconstructing accurate surfaces with radiance fields has progressed rapidly, yet two promising explicit representations, 3D Gaussian Splatting and sparse-voxel rasterization, exhibit complementary strengths and weaknesses. 3D Gaussian Splatting converges quickly and carries useful geometric priors, but surface fidelity is limited by its point-like parameterization. Sparse-voxel rasterization provides continuous opacity fields and crisp geometry, but its typical uniform dense-grid initialization slows convergence and underutilizes scene structure. We combine the advantages of both by introducing a voxel initialization method that places voxels at plausible locations and with appropriate levels of detail, yielding a strong starting point for per-scene optimization. To further enhance depth consistency without blurring edges, we propose refined depth geometry supervision that converts multi-view cues into direct per-ray depth regularization. Experiments on standard benchmarks demonstrate improvements over prior methods in geometric accuracy, better fine-structure recovery, and more complete surfaces, while maintaining fast convergence.

  </details>



- **PocketGS: On-Device Training of 3D Gaussian Splatting for High Perceptual Modeling**  
  Wenzhi Guo, Guangchi Fang, Shu Yang, Bing Wang  
  _2026-01-24_ · https://arxiv.org/abs/2601.17354v1  
  <details><summary>Abstract</summary>

  Efficient and high-fidelity 3D scene modeling is a long-standing pursuit in computer graphics. While recent 3D Gaussian Splatting (3DGS) methods achieve impressive real-time modeling performance, they rely on resource-unconstrained training assumptions that fail on mobile devices, which are limited by minute-scale training budgets and hardware-available peak-memory. We present PocketGS, a mobile scene modeling paradigm that enables on-device 3DGS training under these tightly coupled constraints while preserving high perceptual fidelity. Our method resolves the fundamental contradictions of standard 3DGS through three co-designed operators: G builds geometry-faithful point-cloud priors; I injects local surface statistics to seed anisotropic Gaussians, thereby reducing early conditioning gaps; and T unrolls alpha compositing with cached intermediates and index-mapped gradient scattering for stable mobile backpropagation. Collectively, these operators satisfy the competing requirements of training efficiency, memory compactness, and modeling fidelity. Extensive experiments demonstrate that PocketGS is able to outperform the powerful mainstream workstation 3DGS baseline to deliver high-quality reconstructions, enabling a fully on-device, practical capture-to-rendering workflow.

  </details>


