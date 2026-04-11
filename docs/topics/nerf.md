# NeRF & Neural Radiance Fields

_Updated: 2026-04-11 07:15 UTC_

Total papers shown: **5**


---

- **Novel View Synthesis as Video Completion**  
  Qi Wu, Khiem Vuong, Minsik Jeon, Srinivasa Narasimhan, Deva Ramanan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08500v1  
  <details><summary>Abstract</summary>

  We tackle the problem of sparse novel view synthesis (NVS) using video diffusion models; given $K$ ($\approx 5$) multi-view images of a scene and their camera poses, we predict the view from a target camera pose. Many prior approaches leverage generative image priors encoded via diffusion models. However, models trained on single images lack multi-view knowledge. We instead argue that video models already contain implicit multi-view knowledge and so should be easier to adapt for NVS. Our key insight is to formulate sparse NVS as a low frame-rate video completion task. However, one challenge is that sparse NVS is defined over an unordered set of inputs, often too sparse to admit a meaningful order, so the models should be $\textit{invariant}$ to permutations of that input set. To this end, we present FrameCrafter, which adapts video models (naturally trained with coherent frame orderings) to permutation-invariant NVS through several architectural modifications, including per-frame latent encodings and removal of temporal positional embeddings. Our results suggest that video models can be easily trained to "forget" about time with minimal supervision, producing competitive performance on sparse-view NVS benchmarks. Project page: https://frame-crafter.github.io/

  </details>



- **SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction**  
  Chensheng Dai, Shengjun Zhang, Min Chen, Yueqi Duan  
  _2026-04-09_ · https://arxiv.org/abs/2604.08370v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has demonstrated impressive performance in 3D scene reconstruction. Beyond novel view synthesis, it shows great potential for multi-view surface reconstruction. Existing methods employ optimization-based reconstruction pipelines that achieve precise and complete surface extractions. However, these approaches typically require dense input views and high time consumption for per-scene optimization. To address these limitations, we propose SurfelSplat, a feed-forward framework that generates efficient and generalizable pixel-aligned Gaussian surfel representations from sparse-view images. We observe that conventional feed-forward structures struggle to recover accurate geometric attributes of Gaussian surfels because the spatial frequency of pixel-aligned primitives exceeds Nyquist sampling rates. Therefore, we propose a cross-view feature aggregation module based on the Nyquist sampling theorem. Specifically, we first adapt the geometric forms of Gaussian surfels with spatial sampling rate-guided low-pass filters. We then project the filtered surfels across all input views to obtain cross-view feature correlations. By processing these correlations through a specially designed feature fusion network, we can finally regress Gaussian surfels with precise geometry. Extensive experiments on DTU reconstruction benchmarks demonstrate that our model achieves comparable results with state-of-the-art methods, and predict Gaussian surfels within 1 second, offering a 100x speedup without costly per-scene training.

  </details>



- **From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians**  
  Diego Gomez, Antoine Guédon, Nissim Maruani, Bingchen Gong, Maks Ovsjanikov  
  _2026-04-08_ · https://arxiv.org/abs/2604.07337v1  
  <details><summary>Abstract</summary>

  3D Gaussian Splatting (3DGS) has revolutionized fast novel view synthesis, yet its opacity-based formulation makes surface extraction fundamentally difficult. Unlike implicit methods built on Signed Distance Fields or occupancy, 3DGS lacks a global geometric field, forcing existing approaches to resort to heuristics such as TSDF fusion of blended depth maps. Inspired by the Objects as Volumes framework, we derive a principled occupancy field for Gaussian Splatting and show how it can be used to extract highly accurate watertight meshes of complex scenes. Our key contribution is to introduce a learnable oriented normal at each Gaussian element and to define an adapted attenuation formulation, which leads to closed-form expressions for both the normal and occupancy fields at arbitrary locations in space. We further introduce a novel consistency loss and a dedicated densification strategy to enforce Gaussians to wrap the entire surface by closing geometric holes, ensuring a complete shell of oriented primitives. We modify the differentiable rasterizer to output depth as an isosurface of our continuous model, and introduce Primal Adaptive Meshing for Region-of-Interest meshing at arbitrary resolution. We additionally expose fundamental biases in standard surface evaluation protocols and propose two more rigorous alternatives. Overall, our method Gaussian Wrapping sets a new state-of-the-art on DTU and Tanks and Temples, producing complete, watertight meshes at a fraction of the size of concurrent work-recovering thin structures such as the notoriously elusive bicycle spokes.

  </details>



- **Geo-EVS: Geometry-Conditioned Extrapolative View Synthesis for Autonomous Driving**  
  Yatong Lan, Rongkui Tang, Lei He  
  _2026-04-08_ · https://arxiv.org/abs/2604.07250v1  
  <details><summary>Abstract</summary>

  Extrapolative novel view synthesis can reduce camera-rig dependency in autonomous driving by generating standardized virtual views from heterogeneous sensors. Existing methods degrade outside recorded trajectories because extrapolated poses provide weak geometric support and no dense target-view supervision. The key is to explicitly expose the model to out-of-trajectory condition defects during training. We propose Geo-EVS, a geometry-conditioned framework under sparse supervision. Geo-EVS has two components. Geometry-Aware Reprojection (GAR) uses fine-tuned VGGT to reconstruct colored point clouds and reproject them to observed and virtual target poses, producing geometric condition maps. This design unifies the reprojection path between training and inference. Artifact-Guided Latent Diffusion (AGLD) injects reprojection-derived artifact masks during training so the model learns to recover structure under missing support. For evaluation, we use a LiDAR-Projected Sparse-Reference (LPSR) protocol when dense extrapolated-view ground truth is unavailable. On Waymo, Geo-EVS improves sparse-view synthesis quality and geometric accuracy, especially in high-angle and low-coverage settings. It also improves downstream 3D detection.

  </details>



- **Location Is All You Need: Continuous Spatiotemporal Neural Representations of Earth Observation Data**  
  Mojgan Madadikhaljan, Jonathan Prexl, Isabelle Wittmann, Conrad M Albrecht, Michael Schmitt  
  _2026-04-08_ · https://arxiv.org/abs/2604.07092v2  
  <details><summary>Abstract</summary>

  In this work, we present LIANet (Location Is All You Need Network), a coordinate-based neural representation that models multi-temporal spaceborne Earth observation (EO) data for a given region of interest as a continuous spatiotemporal neural field. Given only spatial and temporal coordinates, LIANet reconstructs the corresponding satellite imagery. Once pretrained, this neural representation can be adapted to various EO downstream tasks, such as semantic segmentation or pixel-wise regression, importantly, without requiring access to the original satellite data. LIANet intends to serve as a user-friendly alternative to Geospatial Foundation Models (GFMs) by eliminating the overhead of data access and preprocessing for end-users and enabling fine-tuning solely based on labels. We demonstrate the pretraining of LIANet across target areas of varying sizes and show that fine-tuning it for downstream tasks achieves competitive performance compared to training from scratch or using established GFMs. The source code and datasets are publicly available at https://github.com/mojganmadadi/LIANet/tree/v1.0.1.

  </details>


