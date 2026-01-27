# NeRF & Neural Radiance Fields

_Updated: 2026-01-27 06:53 UTC_

Total papers shown: **7**


---

- **Splat-Portrait: Generalizing Talking Heads with Gaussian Splatting**  
  Tong Shi, Melonie de Almeida, Daniela Ivanova, Nicolas Pugeault, Paul Henderson  
  _2026-01-26_ · https://arxiv.org/abs/2601.18633v1  
  <details><summary>Abstract</summary>

  Talking Head Generation aims at synthesizing natural-looking talking videos from speech and a single portrait image. Previous 3D talking head generation methods have relied on domain-specific heuristics such as warping-based facial motion representation priors to animate talking motions, yet still produce inaccurate 3D avatar reconstructions, thus undermining the realism of generated animations. We introduce Splat-Portrait, a Gaussian-splatting-based method that addresses the challenges of 3D head reconstruction and lip motion synthesis. Our approach automatically learns to disentangle a single portrait image into a static 3D reconstruction represented as static Gaussian Splatting, and a predicted whole-image 2D background. It then generates natural lip motion conditioned on input audio, without any motion driven priors. Training is driven purely by 2D reconstruction and score-distillation losses, without 3D supervision nor landmarks. Experimental results demonstrate that Splat-Portrait exhibits superior performance on talking head generation and novel view synthesis, achieving better visual quality compared to previous works. Our project code and supplementary documents are public available at https://github.com/stonewalking/Splat-portrait.

  </details>



- **PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction**  
  Isaac Deutsch, Nicolas Moënne-Loccoz, Gavriel State, Zan Gojcic  
  _2026-01-26_ · https://arxiv.org/abs/2601.18336v1  
  <details><summary>Abstract</summary>

  Multi-view 3D reconstruction methods remain highly sensitive to photometric inconsistencies arising from camera optical characteristics and variations in image signal processing (ISP). Existing mitigation strategies such as per-frame latent variables or affine color corrections lack physical grounding and generalize poorly to novel views. We propose the Physically-Plausible ISP (PPISP) correction module, which disentangles camera-intrinsic and capture-dependent effects through physically based and interpretable transformations. A dedicated PPISP controller, trained on the input views, predicts ISP parameters for novel viewpoints, analogous to auto exposure and auto white balance in real cameras. This design enables realistic and fair evaluation on novel views without access to ground-truth images. PPISP achieves SoTA performance on standard benchmarks, while providing intuitive control and supporting the integration of metadata when available. The source code is available at: https://github.com/nv-tlabs/ppisp

  </details>



- **MV-SAM: Multi-view Promptable Segmentation using Pointmap Guidance**  
  Yoonwoo Jeong, Cheng Sun, Yu-Chiang Frank Wang, Minsu Cho, Jaesung Choe  
  _2026-01-25_ · https://arxiv.org/abs/2601.17866v1  
  <details><summary>Abstract</summary>

  Promptable segmentation has emerged as a powerful paradigm in computer vision, enabling users to guide models in parsing complex scenes with prompts such as clicks, boxes, or textual cues. Recent advances, exemplified by the Segment Anything Model (SAM), have extended this paradigm to videos and multi-view images. However, the lack of 3D awareness often leads to inconsistent results, necessitating costly per-scene optimization to enforce 3D consistency. In this work, we introduce MV-SAM, a framework for multi-view segmentation that achieves 3D consistency using pointmaps -- 3D points reconstructed from unposed images by recent visual geometry models. Leveraging the pixel-point one-to-one correspondence of pointmaps, MV-SAM lifts images and prompts into 3D space, eliminating the need for explicit 3D networks or annotated 3D data. Specifically, MV-SAM extends SAM by lifting image embeddings from its pretrained encoder into 3D point embeddings, which are decoded by a transformer using cross-attention with 3D prompt embeddings. This design aligns 2D interactions with 3D geometry, enabling the model to implicitly learn consistent masks across views through 3D positional embeddings. Trained on the SA-1B dataset, our method generalizes well across domains, outperforming SAM2-Video and achieving comparable performance with per-scene optimization baselines on NVOS, SPIn-NeRF, ScanNet++, uCo3D, and DL3DV benchmarks. Code will be released.

  </details>



- **Geometry-Grounded Gaussian Splatting**  
  Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, Ping Tan  
  _2026-01-25_ · https://arxiv.org/abs/2601.17835v1  
  <details><summary>Abstract</summary>

  Gaussian Splatting (GS) has demonstrated impressive quality and efficiency in novel view synthesis. However, shape extraction from Gaussian primitives remains an open problem. Due to inadequate geometry parameterization and approximation, existing shape reconstruction methods suffer from poor multi-view consistency and are sensitive to floaters. In this paper, we present a rigorous theoretical derivation that establishes Gaussian primitives as a specific type of stochastic solids. This theoretical framework provides a principled foundation for Geometry-Grounded Gaussian Splatting by enabling the direct treatment of Gaussian primitives as explicit geometric representations. Using the volumetric nature of stochastic solids, our method efficiently renders high-quality depth maps for fine-grained geometry extraction. Experiments show that our method achieves the best shape reconstruction results among all Gaussian Splatting-based methods on public datasets.

  </details>



- **Learning Sewing Patterns via Latent Flow Matching of Implicit Fields**  
  Cong Cao, Ren Li, Corentin Dumery, Hao Li  
  _2026-01-25_ · https://arxiv.org/abs/2601.17740v1  
  <details><summary>Abstract</summary>

  Sewing patterns define the structural foundation of garments and are essential for applications such as fashion design, fabrication, and physical simulation. Despite progress in automated pattern generation, accurately modeling sewing patterns remains difficult due to the broad variability in panel geometry and seam arrangements. In this work, we introduce a sewing pattern modeling method based on an implicit representation. We represent each panel using a signed distance field that defines its boundary and an unsigned distance field that identifies seam endpoints, and encode these fields into a continuous latent space that enables differentiable meshing. A latent flow matching model learns distributions over panel combinations in this representation, and a stitching prediction module recovers seam relations from extracted edge segments. This formulation allows accurate modeling and generation of sewing patterns with complex structures. We further show that it can be used to estimate sewing patterns from images with improved accuracy relative to existing approaches, and supports applications such as pattern completion and refitting, providing a practical tool for digital fashion design.

  </details>



- **Advancing Structured Priors for Sparse-Voxel Surface Reconstruction**  
  Ting-Hsun Chi, Chu-Rong Chen, Chi-Tun Hsu, Hsuan-Ting Lin, Sheng-Yu Huang, Cheng Sun, Yu-Chiang Frank Wang  
  _2026-01-25_ · https://arxiv.org/abs/2601.17720v1  
  <details><summary>Abstract</summary>

  Reconstructing accurate surfaces with radiance fields has progressed rapidly, yet two promising explicit representations, 3D Gaussian Splatting and sparse-voxel rasterization, exhibit complementary strengths and weaknesses. 3D Gaussian Splatting converges quickly and carries useful geometric priors, but surface fidelity is limited by its point-like parameterization. Sparse-voxel rasterization provides continuous opacity fields and crisp geometry, but its typical uniform dense-grid initialization slows convergence and underutilizes scene structure. We combine the advantages of both by introducing a voxel initialization method that places voxels at plausible locations and with appropriate levels of detail, yielding a strong starting point for per-scene optimization. To further enhance depth consistency without blurring edges, we propose refined depth geometry supervision that converts multi-view cues into direct per-ray depth regularization. Experiments on standard benchmarks demonstrate improvements over prior methods in geometric accuracy, better fine-structure recovery, and more complete surfaces, while maintaining fast convergence.

  </details>



- **NeRF-MIR: Towards High-Quality Restoration of Masked Images with Neural Radiance Fields**  
  Xianliang Huang, Zhizhou Zhong, Shuhang Chen, Yi Xu, Juhong Guan, Shuigeng Zhou  
  _2026-01-24_ · https://arxiv.org/abs/2601.17350v1  
  <details><summary>Abstract</summary>

  Neural Radiance Fields (NeRF) have demonstrated remarkable performance in novel view synthesis. However, there is much improvement room on restoring 3D scenes based on NeRF from corrupted images, which are common in natural scene captures and can significantly impact the effectiveness of NeRF. This paper introduces NeRF-MIR, a novel neural rendering approach specifically proposed for the restoration of masked images, demonstrating the potential of NeRF in this domain. Recognizing that randomly emitting rays to pixels in NeRF may not effectively learn intricate image textures, we propose a \textbf{P}atch-based \textbf{E}ntropy for \textbf{R}ay \textbf{E}mitting (\textbf{PERE}) strategy to distribute emitted rays properly. This enables NeRF-MIR to fuse comprehensive information from images of different views. Additionally, we introduce a \textbf{P}rogressively \textbf{I}terative \textbf{RE}storation (\textbf{PIRE}) mechanism to restore the masked regions in a self-training process. Furthermore, we design a dynamically-weighted loss function that automatically recalibrates the loss weights for masked regions. As existing datasets do not support NeRF-based masked image restoration, we construct three masked datasets to simulate corrupted scenarios. Extensive experiments on real data and constructed datasets demonstrate the superiority of NeRF-MIR over its counterparts in masked image restoration.

  </details>


