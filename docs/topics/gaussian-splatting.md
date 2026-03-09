# Gaussian Splatting & 3DGS

_Updated: 2026-03-09 07:16 UTC_

Total papers shown: **3**


---

- **EntON: Eigenentropy-Optimized Neighborhood Densification in 3D Gaussian Splatting**  
  Miriam Jäger, Boris Jutzi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06216v1  
  <details><summary>Abstract</summary>

  We present a novel Eigenentropy-optimized neighboorhood densification strategy EntON in 3D Gaussian Splatting (3DGS) for geometrically accurate and high-quality rendered 3D reconstruction. While standard 3DGS produces Gaussians whose centers and surfaces are poorly aligned with the underlying object geometry, surface-focused reconstruction methods frequently sacrifice photometric accuracy. In contrast to the conventional densification strategy, which relies on the magnitude of the view-space position gradient, our approach introduces a geometry-aware strategy to guide adaptive splitting and pruning. Specifically, we compute the 3D shape feature Eigenentropy from the eigenvalues of the covariance matrix in the k-nearest neighborhood of each Gaussian center, which quantifies the local structural order. These Eigenentropy values are integrated into an alternating optimization framework: During the optimization process, the algorithm alternates between (i) standard gradient-based densification, which refines regions via view-space gradients, and (ii) Eigenentropy-aware densification, which preferentially densifies Gaussians in low-Eigenentropy (ordered, flat) neighborhoods to better capture fine geometric details on the object surface, and prunes those in high-Eigenentropy (disordered, spherical) regions. We provide quantitative and qualitative evaluations on two benchmark datasets: small-scale DTU dataset and large-scale TUM2TWIN dataset, covering man-made objects and urban scenes. Experiments demonstrate that our Eigenentropy-aware alternating densification strategy improves geometric accuracy by up to 33% and rendering quality by up to 7%, while reducing the number of Gaussians by up to 50% and training time by up to 23%. Overall, EnTON achieves a favorable balance between geometric accuracy, rendering quality and efficiency by avoiding unnecessary scene expansion.

  </details>



- **VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction**  
  Xiaoyang Yan, Muleilan Pei, Shaojie Shen  
  _2026-03-06_ · https://arxiv.org/abs/2603.06210v1  
  <details><summary>Abstract</summary>

  3D semantic occupancy prediction has become a crucial perception task for comprehensive scene understanding in autonomous driving. While recent advances have explored 3D Gaussian splatting for occupancy modeling to substantially reduce computational overhead, the generation of high-quality 3D Gaussians relies heavily on accurate geometric cues, which are often insufficient in purely vision-centric paradigms. To bridge this gap, we advocate for injecting the strong geometric grounding capability from Vision Foundation Models (VFMs) into occupancy prediction. In this regard, we introduce Visual Geometry Grounded Gaussian Splatting (VG3S), a novel framework that empowers Gaussian-based occupancy prediction with cross-view 3D geometric grounding. Specifically, to fully exploit the rich 3D geometric priors from a frozen VFM, we propose a plug-and-play hierarchical geometric feature adapter, which can effectively transform generic VFM tokens via feature aggregation, task-specific alignment, and multi-scale restructuring. Extensive experiments on the nuScenes occupancy benchmark demonstrate that VG3S achieves remarkable improvements of 12.6% in IoU and 7.5% in mIoU over the baseline. Furthermore, we show that VG3S generalizes seamlessly across diverse VFMs, consistently enhancing occupancy prediction accuracy and firmly underscoring the immense value of integrating priors derived from powerful, pre-trained geometry-grounded VFMs.

  </details>



- **Transforming Omnidirectional RGB-LiDAR data into 3D Gaussian Splatting**  
  Semin Bae, Hansol Lim, Jongseong Brad Choi  
  _2026-03-06_ · https://arxiv.org/abs/2603.06061v1  
  <details><summary>Abstract</summary>

  The demand for large-scale digital twins is rapidly growing in robotics and autonomous driving. However, constructing these environments with 3D Gaussian Splatting (3DGS) usually requires expensive, purpose-built data collection. Meanwhile, deployed platforms routinely collect extensive omnidirectional RGB and LiDAR logs, but a significant portion of these sensor data is directly discarded or strictly underutilized due to transmission constraints and the lack of scalable reuse pipeline. In this paper, we present an omnidirectional RGB-LiDAR reuse pipeline that transforms these archived logs into robust initialization assets for 3DGS. Direct conversion of such raw logs introduces practical bottlenecks: inherent non-linear distortion leads to unreliable Structure-from-Motion (SfM) tracking, and dense, unorganized LiDAR clouds cause computational overhead during 3DGS optimization. To overcome these challenges, our pipeline strategically integrates an ERP-to-cubemap conversion module for deterministic spatial anchoring, alongside PRISM-a color stratified downsampling strategy. By bridging these multi-modal inputs via Fast Point Feature Histograms (FPFH) based global registration and Iterative Closest Point (ICP), our pipeline successfully repurposes a considerable fraction of discarded data into usable SfM geometry. Furthermore, our LiDAR-reinforced initialization consistently enhances the final 3DGS rendering fidelity in structurally complex scenes compared to vision-only baselines. Ultimately, this work provides a deterministic workflow for creating simulation-grade digital twins from standard archived sensor logs.

  </details>


