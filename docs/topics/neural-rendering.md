# Neural Rendering & View Synthesis

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



- **DRoPS: Dynamic 3D Reconstruction of Pre-Scanned Objects**  
  Narek Tumanyan, Samuel Rota Bulò, Denis Rozumny, Lorenzo Porzi, Adam Harley, Tali Dekel, Peter Kontschieder, Jonathon Luiten  
  _2026-03-25_ · https://arxiv.org/abs/2603.24770v1  
  <details><summary>Abstract</summary>

  Dynamic scene reconstruction from casual videos has seen recent remarkable progress. Numerous approaches have attempted to overcome the ill-posedness of the task by distilling priors from 2D foundational models and by imposing hand-crafted regularization on the optimized motion. However, these methods struggle to reconstruct scenes from extreme novel viewpoints, especially when highly articulated motions are present. In this paper, we present DRoPS, a novel approach that leverages a static pre-scan of the dynamic object as an explicit geometric and appearance prior. While existing state-of-the-art methods fail to fully exploit the pre-scan, DRoPS leverages our novel setup to effectively constrain the solution space and ensure geometrical consistency throughout the sequence. The core of our novelty is twofold: first, we establish a grid-structured and surface-aligned model by organizing Gaussian primitives into pixel grids anchored to the object surface. Second, by leveraging the grid structure of our primitives, we parameterize motion using a CNN conditioned on those grids, injecting strong implicit regularization and correlating the motion of nearby points. Extensive experiments demonstrate that our method significantly outperforms the current state of the art in rendering quality and 3D tracking accuracy.

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



- **FilterGS: Traversal-Free Parallel Filtering and Adaptive Shrinking for Large-Scale LoD 3D Gaussian Splatting**  
  Yixian Wang, Haolin Yu, Jiadong Tang, Yu Gao, Xihan Wang, Yufeng Yue, Yi Yang  
  _2026-03-25_ · https://arxiv.org/abs/2603.23891v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting has revolutionized neural rendering with real-time performance. However, scaling this approach to large scenes using Level-of-Detail methods faces critical challenges: inefficient serial traversal consuming over 60\% of rendering time, and redundant Gaussian-tile pairs that incur unnecessary processing overhead. To address these limitations, we introduce FilterGS, featuring a parallel filtering mechanism with two complementary filters that select Gaussian elements efficiently without tree traversal. Additionally, we propose a novel GTC metric that quantifies the redundancy of Gaussian-tile key-value pairs. Based on this metric, we introduce a scene-adaptive Gaussian shrinking strategy that effectively reduces redundant pairs. Extensive experiments demonstrate that FilterGS achieves state-of-the-art rendering speeds while maintaining competitive visual quality across multiple large-scale datasets. Project page: https://github.com/xenon-w/FilterGS

  </details>


