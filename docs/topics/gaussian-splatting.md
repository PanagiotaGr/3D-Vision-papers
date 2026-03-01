# Gaussian Splatting & 3DGS

_Updated: 2026-03-01 07:02 UTC_

Total papers shown: **4**


---

- **Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking**  
  Maximilian Luz, Rohit Mohan, Thomas Nürnberg, Yakov Miron, Daniele Cattaneo, Abhinav Valada  
  _2026-02-26_ · https://arxiv.org/abs/2602.23172v1  
  <details><summary>Abstract</summary>

  Capturing 4D spatiotemporal surroundings is crucial for the safe and reliable operation of robots in dynamic environments. However, most existing methods address only one side of the problem: they either provide coarse geometric tracking via bounding boxes, or detailed 3D structures like voxel-based occupancy that lack explicit temporal association. In this work, we present Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking (LaGS) that advances spatiotemporal scene understanding in a holistic direction. Our approach incorporates camera-based end-to-end tracking with mask-based multi-view panoptic occupancy prediction, and addresses the key challenge of efficiently aggregating multi-view information into 3D voxel grids via a novel latent Gaussian splatting approach. Specifically, we first fuse observations into 3D Gaussians that serve as a sparse point-centric latent representation of the 3D scene, and then splat the aggregated features onto a 3D voxel grid that is decoded by a mask-based segmentation head. We evaluate LaGS on the Occ3D nuScenes and Waymo datasets, achieving state-of-the-art performance for 4D panoptic occupancy tracking. We make our code available at https://lags.cs.uni-freiburg.de/.

  </details>



- **PackUV: Packed Gaussian UV Maps for 4D Volumetric Video**  
  Aashish Rai, Angela Xing, Anushka Agarwal, Xiaoyan Cong, Zekun Li, Tao Lu, Aayush Prakash, Srinath Sridhar  
  _2026-02-26_ · https://arxiv.org/abs/2602.23040v1  
  <details><summary>Abstract</summary>

  Volumetric videos offer immersive 4D experiences, but remain difficult to reconstruct, store, and stream at scale. Existing Gaussian Splatting based methods achieve high-quality reconstruction but break down on long sequences, temporal inconsistency, and fail under large motions and disocclusions. Moreover, their outputs are typically incompatible with conventional video coding pipelines, preventing practical applications. We introduce PackUV, a novel 4D Gaussian representation that maps all Gaussian attributes into a sequence of structured, multi-scale UV atlas, enabling compact, image-native storage. To fit this representation from multi-view videos, we propose PackUV-GS, a temporally consistent fitting method that directly optimizes Gaussian parameters in the UV domain. A flow-guided Gaussian labeling and video keyframing module identifies dynamic Gaussians, stabilizes static regions, and preserves temporal coherence even under large motions and disocclusions. The resulting UV atlas format is the first unified volumetric video representation compatible with standard video codecs (e.g., FFV1) without losing quality, enabling efficient streaming within existing multimedia infrastructure. To evaluate long-duration volumetric capture, we present PackUV-2B, the largest multi-view video dataset to date, featuring more than 50 synchronized cameras, substantial motion, and frequent disocclusions across 100 sequences and 2B (billion) frames. Extensive experiments demonstrate that our method surpasses existing baselines in rendering fidelity while scaling to sequences up to 30 minutes with consistent quality.

  </details>



- **GSTurb: Gaussian Splatting for Atmospheric Turbulence Mitigation**  
  Hanliang Du, Zhangji Lu, Zewei Cai, Qijian Tang, Qifeng Yu, Xiaoli Liu  
  _2026-02-26_ · https://arxiv.org/abs/2602.22800v1  
  <details><summary>Abstract</summary>

  Atmospheric turbulence causes significant image degradation due to pixel displacement (tilt) and blur, particularly in long-range imaging applications. In this paper, we propose a novel framework for atmospheric turbulence mitigation, GSTurb, which integrates optical flow-guided tilt correction and Gaussian splatting for modeling non-isoplanatic blur. The framework employs Gaussian parameters to represent tilt and blur, and optimizes them across multiple frames to enhance restoration. Experimental results on the ATSyn-static dataset demonstrate the effectiveness of our method, achieving a peak PSNR of 27.67 dB and SSIM of 0.8735. Compared to the state-of-the-art method, GSTurb improves PSNR by 1.3 dB (a 4.5% increase) and SSIM by 0.048 (a 5.8% increase). Additionally, on real datasets, including the TSRWGAN Real-World and CLEAR datasets, GSTurb outperforms existing methods, showing significant improvements in both qualitative and quantitative performance. These results highlight that combining optical flow-guided tilt correction with Gaussian splatting effectively enhances image restoration under both synthetic and real-world turbulence conditions. The code for this method will be available at https://github.com/DuhlLiamz/3DGS_turbulence/tree/main.

  </details>



- **Sapling-NeRF: Geo-Localised Sapling Reconstruction in Forests for Ecological Monitoring**  
  Miguel Ángel Muñoz-Bañón, Nived Chebrolu, Sruthi M. Krishna Moorthy, Yifu Tao, Fernando Torres, Roberto Salguero-Gómez, Maurice Fallon  
  _2026-02-26_ · https://arxiv.org/abs/2602.22731v1  
  <details><summary>Abstract</summary>

  Saplings are key indicators of forest regeneration and overall forest health. However, their fine-scale architectural traits are difficult to capture with existing 3D sensing methods, which make quantitative evaluation difficult. Terrestrial Laser Scanners (TLS), Mobile Laser Scanners (MLS), or traditional photogrammetry approaches poorly reconstruct thin branches, dense foliage, and lack the scale consistency needed for long-term monitoring. Implicit 3D reconstruction methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) are promising alternatives, but cannot recover the true scale of a scene and lack any means to be accurately geo-localised. In this paper, we present a pipeline which fuses NeRF, LiDAR SLAM, and GNSS to enable repeatable, geo-localised ecological monitoring of saplings. Our system proposes a three-level representation: (i) coarse Earth-frame localisation using GNSS, (ii) LiDAR-based SLAM for centimetre-accurate localisation and reconstruction, and (iii) NeRF-derived object-centric dense reconstruction of individual saplings. This approach enables repeatable quantitative evaluation and long-term monitoring of sapling traits. Our experiments in forest plots in Wytham Woods (Oxford, UK) and Evo (Finland) show that stem height, branching patterns, and leaf-to-wood ratios can be captured with increased accuracy as compared to TLS. We demonstrate that accurate stem skeletons and leaf distributions can be measured for saplings with heights between 0.5m and 2m in situ, giving ecologists access to richer structural and quantitative data for analysing forest dynamics.

  </details>


