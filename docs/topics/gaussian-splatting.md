# Gaussian Splatting & 3DGS

_Updated: 2026-01-31 06:56 UTC_

Total papers shown: **8**


---

- **PLANING: A Loosely Coupled Triangle-Gaussian Framework for Streaming 3D Reconstruction**  
  Changjian Jiang, Kerui Ren, Xudong Li, Kaiwen Song, Linning Xu, Tao Lu, Junting Dong, Yu Zhang, Bo Dai, Mulin Yu  
  _2026-01-29_ · https://arxiv.org/abs/2601.22046v1  
  <details><summary>Abstract</summary>

  Streaming reconstruction from monocular image sequences remains challenging, as existing methods typically favor either high-quality rendering or accurate geometry, but rarely both. We present PLANING, an efficient on-the-fly reconstruction framework built on a hybrid representation that loosely couples explicit geometric primitives with neural Gaussians, enabling geometry and appearance to be modeled in a decoupled manner. This decoupling supports an online initialization and optimization strategy that separates geometry and appearance updates, yielding stable streaming reconstruction with substantially reduced structural redundancy. PLANING improves dense mesh Chamfer-L2 by 18.52% over PGSR, surpasses ARTDECO by 1.31 dB PSNR, and reconstructs ScanNetV2 scenes in under 100 seconds, over 5x faster than 2D Gaussian Splatting, while matching the quality of offline per-scene optimization. Beyond reconstruction quality, the structural clarity and computational efficiency of \modelname~make it well suited for a broad range of downstream applications, such as enabling large-scale scene modeling and simulation-ready environments for embodied AI. Project page: https://city-super.github.io/PLANING/ .

  </details>



- **Hybrid Foveated Path Tracing with Peripheral Gaussians for Immersive Anatomy**  
  Constantin Kleinbeck, Luisa Theelke, Hannah Schieber, Ulrich Eck, Rüdiger von Eisenhart-Rothe, Daniel Roth  
  _2026-01-29_ · https://arxiv.org/abs/2601.22026v1  
  <details><summary>Abstract</summary>

  Volumetric medical imaging offers great potential for understanding complex pathologies. Yet, traditional 2D slices provide little support for interpreting spatial relationships, forcing users to mentally reconstruct anatomy into three dimensions. Direct volumetric path tracing and VR rendering can improve perception but are computationally expensive, while precomputed representations, like Gaussian Splatting, require planning ahead. Both approaches limit interactive use. We propose a hybrid rendering approach for high-quality, interactive, and immersive anatomical visualization. Our method combines streamed foveated path tracing with a lightweight Gaussian Splatting approximation of the periphery. The peripheral model generation is optimized with volume data and continuously refined using foveal renderings, enabling interactive updates. Depth-guided reprojection further improves robustness to latency and allows users to balance fidelity with refresh rate. We compare our method against direct path tracing and Gaussian Splatting. Our results highlight how their combination can preserve strengths in visual quality while re-generating the peripheral model in under a second, eliminating extensive preprocessing and approximations. This opens new options for interactive medical visualization.

  </details>



- **Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring**  
  Borja Carrillo-Perez, Felix Sattler, Angel Bueno Rodriguez, Maurice Stephan, Sarah Barnes  
  _2026-01-29_ · https://arxiv.org/abs/2601.21786v1  
  <details><summary>Abstract</summary>

  Three-dimensional (3D) reconstruction of ships is an important part of maritime monitoring, allowing improved visualization, inspection, and decision-making in real-world monitoring environments. However, most state-ofthe-art 3D reconstruction methods require multi-view supervision, annotated 3D ground truth, or are computationally intensive, making them impractical for real-time maritime deployment. In this work, we present an efficient pipeline for single-view 3D reconstruction of real ships by training entirely on synthetic data and requiring only a single view at inference. Our approach uses the Splatter Image network, which represents objects as sparse sets of 3D Gaussians for rapid and accurate reconstruction from single images. The model is first fine-tuned on synthetic ShapeNet vessels and further refined with a diverse custom dataset of 3D ships, bridging the domain gap between synthetic and real-world imagery. We integrate a state-of-the-art segmentation module based on YOLOv8 and custom preprocessing to ensure compatibility with the reconstruction network. Postprocessing steps include real-world scaling, centering, and orientation alignment, followed by georeferenced placement on an interactive web map using AIS metadata and homography-based mapping. Quantitative evaluation on synthetic validation data demonstrates strong reconstruction fidelity, while qualitative results on real maritime images from the ShipSG dataset confirm the potential for transfer to operational maritime settings. The final system provides interactive 3D inspection of real ships without requiring real-world 3D annotations. This pipeline provides an efficient, scalable solution for maritime monitoring and highlights a path toward real-time 3D ship visualization in practical applications. Interactive demo: https://dlr-mi.github.io/ship3d-demo/.

  </details>



- **Mesh Splatting for End-to-end Multiview Surface Reconstruction**  
  Ruiqi Zhang, Jiacheng Wu, Jie Chen  
  _2026-01-29_ · https://arxiv.org/abs/2601.21400v1  
  <details><summary>Abstract</summary>

  Surfaces are typically represented as meshes, which can be extracted from volumetric fields via meshing or optimized directly as surface parameterizations. Volumetric representations occupy 3D space and have a large effective receptive field along rays, enabling stable and efficient optimization via volumetric rendering; however, subsequent meshing often produces overly dense meshes and introduces accumulated errors. In contrast, pure surface methods avoid meshing but capture only boundary geometry with a single-layer receptive field, making it difficult to learn intricate geometric details and increasing reliance on priors (e.g., shading or normals). We bridge this gap by differentiably turning a surface representation into a volumetric one, enabling end-to-end surface reconstruction via volumetric rendering to model complex geometries. Specifically, we soften a mesh into multiple semi-transparent layers that remain differentiable with respect to the base mesh, endowing it with a controllable 3D receptive field. Combined with a splatting-based renderer and a topology-control strategy, our method can be optimized in about 20 minutes to achieve accurate surface reconstruction while substantially improving mesh quality.

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



- **GRTX: Efficient Ray Tracing for 3D Gaussian-Based Rendering**  
  Junseo Lee, Sangyun Jeon, Jungi Lee, Junyong Park, Jaewoong Sim  
  _2026-01-28_ · https://arxiv.org/abs/2601.20429v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has gained widespread adoption across diverse applications due to its exceptional rendering performance and visual quality. While most existing methods rely on rasterization to render Gaussians, recent research has started investigating ray tracing approaches to overcome the fundamental limitations inherent in rasterization. However, current Gaussian ray tracing methods suffer from inefficiencies such as bloated acceleration structures and redundant node traversals, which greatly degrade ray tracing performance. In this work, we present GRTX, a set of software and hardware optimizations that enable efficient ray tracing for 3D Gaussian-based rendering. First, we introduce a novel approach for constructing streamlined acceleration structures for Gaussian primitives. Our key insight is that anisotropic Gaussians can be treated as unit spheres through ray space transformations, which substantially reduces BVH size and traversal overhead. Second, we propose dedicated hardware support for traversal checkpointing within ray tracing units. This eliminates redundant node visits during multi-round tracing by resuming traversal from checkpointed nodes rather than restarting from the root node in each subsequent round. Our evaluation shows that GRTX significantly improves ray tracing performance compared to the baseline ray tracing method with a negligible hardware cost.

  </details>



- **GVGS: Gaussian Visibility-Aware Multi-View Geometry for Accurate Surface Reconstruction**  
  Mai Su, Qihan Yu, Zhongtao Wang, Yilong Li, Chengwei Pan, Yisong Chen, Guoping Wang  
  _2026-01-28_ · https://arxiv.org/abs/2601.20331v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting enables efficient optimization and high-quality rendering, yet accurate surface reconstruction remains challenging. Prior methods improve surface reconstruction by refining Gaussian depth estimates, either via multi-view geometric consistency or through monocular depth priors. However, multi-view constraints become unreliable under large geometric discrepancies, while monocular priors suffer from scale ambiguity and local inconsistency, ultimately leading to inaccurate Gaussian depth supervision. To address these limitations, we introduce a Gaussian visibility-aware multi-view geometric consistency constraint that aggregates the visibility of shared Gaussian primitives across views, enabling more accurate and stable geometric supervision. In addition, we propose a progressive quadtree-calibrated Monocular depth constraint that performs block-wise affine calibration from coarse to fine spatial scales, mitigating the scale ambiguity of depth priors while preserving fine-grained surface details. Extensive experiments on DTU and TNT datasets demonstrate consistent improvements in geometric accuracy over prior Gaussian-based and implicit surface reconstruction methods. Codes are available at an anonymous repository: https://github.com/GVGScode/GVGS.

  </details>


