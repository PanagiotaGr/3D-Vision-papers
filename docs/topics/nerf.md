# NeRF & Neural Radiance Fields

_Updated: 2026-03-27 07:34 UTC_

Total papers shown: **10**


---

- **Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting**  
  Yixing Lao, Xuyang Bai, Xiaoyang Wu, Nuoyuan Yan, Zixin Luo, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Shiwei Li, Hengshuang Zhao  
  _2026-03-26_ · https://arxiv.org/abs/2603.25745v1  
  <details><summary>Abstract</summary>

  Existing feed-forward 3D Gaussian Splatting methods predict pixel-aligned primitives, leading to a quadratic growth in primitive count as resolution increases. This fundamentally limits their scalability, making high-resolution synthesis such as 4K intractable. We introduce LGTM (Less Gaussians, Texture More), a feed-forward framework that overcomes this resolution scaling barrier. By predicting compact Gaussian primitives coupled with per-primitive textures, LGTM decouples geometric complexity from rendering resolution. This approach enables high-fidelity 4K novel view synthesis without per-scene optimization, a capability previously out of reach for feed-forward methods, all while using significantly fewer Gaussian primitives. Project page: https://yxlao.github.io/lgtm/

  </details>



- **MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data**  
  Zhekai Chen, Yuqing Wang, Manyuan Zhang, Xihui Liu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25319v1  
  <details><summary>Abstract</summary>

  Generating images conditioned on multiple visual references is critical for real-world applications such as multi-subject composition, narrative illustration, and novel view synthesis, yet current models suffer from severe performance degradation as the number of input references grows. We identify the root cause as a fundamental data bottleneck: existing datasets are dominated by single- or few-reference pairs and lack the structured, long-context supervision needed to learn dense inter-reference dependencies. To address this, we introduce MacroData, a large-scale dataset of 400K samples, each containing up to 10 reference images, systematically organized across four complementary dimensions -- Customization, Illustration, Spatial reasoning, and Temporal dynamics -- to provide comprehensive coverage of the multi-reference generation space. Recognizing the concurrent absence of standardized evaluation protocols, we further propose MacroBench, a benchmark of 4,000 samples that assesses generative coherence across graded task dimensions and input scales. Extensive experiments show that fine-tuning on MacroData yields substantial improvements in multi-reference generation, and ablation studies further reveal synergistic benefits of cross-task co-training and effective strategies for handling long-context complexity. The dataset and benchmark will be publicly released.

  </details>



- **ViewSplat: View-Adaptive Dynamic Gaussian Splatting for Feed-Forward Synthesis**  
  Moonyeon Jeong, Seunggi Min, Suhyeon Lee, Hongje Seong  
  _2026-03-26_ · https://arxiv.org/abs/2603.25265v1  
  <details><summary>Abstract</summary>

  We present ViewSplat, a view-adaptive 3D Gaussian splatting network for novel view synthesis from unposed images. While recent feed-forward 3D Gaussian splatting has significantly accelerated 3D scene reconstruction by bypassing per-scene optimization, a fundamental fidelity gap remains. We attribute this bottleneck to the limited capacity of single-step feed-forward networks to regress static Gaussian primitives that satisfy all viewpoints. To address this limitation, we shift the paradigm from static primitive regression to view-adaptive dynamic splatting. Instead of a rigid Gaussian representation, our pipeline learns a view-adaptable latent representation. Specifically, ViewSplat initially predicts base Gaussian primitives alongside the weights of dynamic MLPs. During rendering, these MLPs take target view coordinates as input and predict view-dependent residual updates for each Gaussian attribute (i.e., 3D position, scale, rotation, opacity, and color). This mechanism, which we term view-adaptive dynamic splatting, allows each primitive to rectify initial estimation errors, effectively capturing high-fidelity appearances. Extensive experiments demonstrate that ViewSplat achieves state-of-the-art fidelity while maintaining fast inference (17 FPS) and real-time rendering (154 FPS).

  </details>



- **AirSplat: Alignment and Rating for Robust Feed-Forward 3D Gaussian Splatting**  
  Minh-Quan Viet Bui, Jaeho Moon, Munchurl Kim  
  _2026-03-26_ · https://arxiv.org/abs/2603.25129v1  
  <details><summary>Abstract</summary>

  While 3D Vision Foundation Models (3DVFMs) have demonstrated remarkable zero-shot capabilities in visual geometry estimation, their direct application to generalizable novel view synthesis (NVS) remains challenging. In this paper, we propose AirSplat, a novel training framework that effectively adapts the robust geometric priors of 3DVFMs into high-fidelity, pose-free NVS. Our approach introduces two key technical contributions: (1) Self-Consistent Pose Alignment (SCPA), a training-time feedback loop that ensures pixel-aligned supervision to resolve pose-geometry discrepancy; and (2) Rating-based Opacity Matching (ROM), which leverages the local 3D geometry consistency knowledge from a sparse-view NVS teacher model to filter out degraded primitives. Experimental results on large-scale benchmarks demonstrate that our method significantly outperforms state-of-the-art pose-free NVS approaches in reconstruction quality. Our AirSplat highlights the potential of adapting 3DVFMs to enable simultaneous visual geometry estimation and high-quality view synthesis.

  </details>



- **Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos**  
  Xuankai Zhang, Junjin Xiao, Shangwei Huang, Wei-shi Zheng, Qing Zhang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25058v1  
  <details><summary>Abstract</summary>

  We present an approach for high-quality dynamic Gaussian Splatting from monocular videos. To this end, we in this work go one step further beyond previous methods to explicitly model continuous position and orientation deformation of dynamic Gaussians, using an SE(3) B-spline motion bases with a compact set of control points. To improve computational efficiency while enhancing the ability to model complex motions, an adaptive control mechanism is devised to dynamically adjust the number of motion bases and control points. Besides, we develop a soft segment reconstruction strategy to mitigate long-interval motion interference, and employ a multi-view diffusion model to provide multi-view cues for avoiding overfitting to training views. Extensive experiments demonstrate that our method outperforms state-of-the-art methods in novel view synthesis. Our code is available at https://github.com/hhhddddddd/se3bsplinegs.

  </details>



- **GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator**  
  Liyuan Zhu, Manjunath Narayana, Michal Stary, Will Hutchcroft, Gordon Wetzstein, Iro Armeni  
  _2026-03-26_ · https://arxiv.org/abs/2603.25053v1  
  <details><summary>Abstract</summary>

  We present GaussFusion, a novel approach for improving 3D Gaussian splatting (3DGS) reconstructions in the wild through geometry-informed video generation. GaussFusion mitigates common 3DGS artifacts, including floaters, flickering, and blur caused by camera pose errors, incomplete coverage, and noisy geometry initialization. Unlike prior RGB-based approaches limited to a single reconstruction pipeline, our method introduces a geometry-informed video-to-video generator that refines 3DGS renderings across both optimization-based and feed-forward methods. Given an existing reconstruction, we render a Gaussian primitive video buffer encoding depth, normals, opacity, and covariance, which the generator refines to produce temporally coherent, artifact-free frames. We further introduce an artifact synthesis pipeline that simulates diverse degradation patterns, ensuring robustness and generalization. GaussFusion achieves state-of-the-art performance on novel-view synthesis benchmarks, and an efficient variant runs in real time at 21 FPS while maintaining similar performance, enabling interactive 3D applications.

  </details>



- **GeoNDC: A Queryable Neural Data Cube for Planetary-Scale Earth Observation**  
  Jianbo Qi, Mengyao Li, Baogui Jiang, Yidan Chen, Qiao Wang  
  _2026-03-26_ · https://arxiv.org/abs/2603.25037v1  
  <details><summary>Abstract</summary>

  Satellite Earth observation has accumulated massive spatiotemporal archives essential for monitoring environmental change, yet these remain organized as discrete raster files, making them costly to store, transmit, and query. We present GeoNDC, a queryable neural data cube that encodes planetary-scale Earth observation data as a continuous spatiotemporal implicit neural field, enabling on-demand queries and continuous-time reconstruction without full decompression. Experiments on a 20-year global MODIS MCD43A4 reflectance record (7 bands, 5\,km, 8-day sampling) show that the learned representation supports direct spatiotemporal queries on consumer hardware. On Sentinel-2 imagery (10\,m), continuous temporal parameterization recovers cloud-free dynamics with high fidelity ($R^2 > 0.85$) under simulated 2-km cloud occlusion. On HiGLASS biophysical products (LAI and FPAR), GeoNDC attains near-perfect accuracy ($R^2 > 0.98$). The representation compresses the 20-year MODIS archive to 0.44\,GB -- approximately 95:1 relative to an optimized Int16 baseline -- with high spectral fidelity (mean $R^2 > 0.98$, mean RMSE $= 0.021$). These results suggest GeoNDC offers a unified AI-native representation for planetary-scale Earth observation, complementing raw archives with a compact, analysis-ready data layer integrating query, reconstruction, and compression in a single framework.

  </details>



- **Few TensoRF: Enhance the Few-shot on Tensorial Radiance Fields**  
  Thanh-Hai Le, Hoang-Hau Tran, Trong-Nghia Vu  
  _2026-03-26_ · https://arxiv.org/abs/2603.25008v1  
  <details><summary>Abstract</summary>

  This paper presents Few TensoRF, a 3D reconstruction framework that combines TensorRF's efficient tensor based representation with FreeNeRF's frequency driven few shot regularization. Using TensorRF to significantly accelerate rendering speed and introducing frequency and occlusion masks, the method improves stability and reconstruction quality under sparse input views. Experiments on the Synthesis NeRF benchmark show that Few TensoRF method improves the average PSNR from 21.45 dB (TensorRF) to 23.70 dB, with the fine tuned version reaching 24.52 dB, while maintaining TensorRF's fast \(\approx10-15\) minute training time. Experiments on the THuman 2.0 dataset further demonstrate competitive performance in human body reconstruction, achieving 27.37 - 34.00 dB with only eight input images. These results highlight Few TensoRF as an efficient and data effective solution for real-time 3D reconstruction across diverse scenes.

  </details>



- **Accurate Point Measurement in 3DGS -- A New Alternative to Traditional Stereoscopic-View Based Measurements**  
  Deyan Deng, Rongjun Qin  
  _2026-03-25_ · https://arxiv.org/abs/2603.24716v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized real-time rendering with its state-of-the-art novel view synthesis, but its utility for accurate geometric measurement remains underutilized. Compared to multi-view stereo (MVS) point clouds or meshes, 3DGS rendered views present superior visual quality and completeness. However, current point measurement methods still rely on demanding stereoscopic workstations or direct picking on often-incomplete and inaccurate 3D meshes. As a novel view synthesizer, 3DGS renders exact source views and smoothly interpolates in-between views. This allows users to intuitively pick congruent points across different views while operating 3DGS models. By triangulating these congruent points, one can precisely generate 3D point measurements. This approach mimics traditional stereoscopic measurement but is significantly less demanding: it requires neither a stereo workstation nor specialized operator stereoscopic capability. Furthermore, it enables multi-view intersection (more than two views) for higher measurement accuracy. We implemented a web-based application to demonstrate this proof-of-concept (PoC). Using several UAV aerial datasets, we show this PoC allows users to successfully perform highly accurate point measurements, achieving accuracy matching or exceeding traditional stereoscopic methods on standard hardware. Specifically, our approach significantly outperforms direct mesh-based measurements. Quantitatively, our method achieves RMSEs in the 1-2 cm range on well-defined points. More critically, on challenging thin structures where mesh-based RMSE was 0.062 m, our method achieved 0.037 m. On sharp corners poorly reconstructed in the mesh, our method successfully measured all points with a 0.013 m RMSE, whereas the mesh method failed entirely. Code is available at: https://github.com/GDAOSU/3dgs_measurement_tool.

  </details>



- **SpectralSplats: Robust Differentiable Tracking via Spectral Moment Supervision**  
  Avigail Cohen Rimon, Amir Mann, Mirela Ben Chen, Or Litany  
  _2026-03-25_ · https://arxiv.org/abs/2603.24036v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) enables real-time, photorealistic novel view synthesis, making it a highly attractive representation for model-based video tracking. However, leveraging the differentiability of the 3DGS renderer "in the wild" remains notoriously fragile. A fundamental bottleneck lies in the compact, local support of the Gaussian primitives. Standard photometric objectives implicitly rely on spatial overlap; if severe camera misalignment places the rendered object outside the target's local footprint, gradients strictly vanish, leaving the optimizer stranded. We introduce SpectralSplats, a robust tracking framework that resolves this "vanishing gradient" problem by shifting the optimization objective from the spatial to the frequency domain. By supervising the rendered image via a set of global complex sinusoidal features (Spectral Moments), we construct a global basin of attraction, ensuring that a valid, directional gradient toward the target exists across the entire image domain, even when pixel overlap is completely nonexistent. To harness this global basin without introducing periodic local minima associated with high frequencies, we derive a principled Frequency Annealing schedule from first principles, gracefully transitioning the optimizer from global convexity to precise spatial alignment. We demonstrate that SpectralSplats acts as a seamless, drop-in replacement for spatial losses across diverse deformation parameterizations (from MLPs to sparse control points), successfully recovering complex deformations even from severely misaligned initializations where standard appearance-based tracking catastrophically fails.

  </details>


