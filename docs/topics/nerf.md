# NeRF & Neural Radiance Fields

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **5**


---

- **Neural Implicit Representations for 3D Synthetic Aperture Radar Imaging**  
  Nithin Sugavanam, Emre Ertin  
  _2026-02-19_ · https://arxiv.org/abs/2602.17556v1  
  <details><summary>Abstract</summary>

  Synthetic aperture radar (SAR) is a tomographic sensor that measures 2D slices of the 3D spatial Fourier transform of the scene. In many operational scenarios, the measured set of 2D slices does not fill the 3D space in the Fourier domain, resulting in significant artifacts in the reconstructed imagery. Traditionally, simple priors, such as sparsity in the image domain, are used to regularize the inverse problem. In this paper, we review our recent work that achieves state-of-the-art results in 3D SAR imaging employing neural structures to model the surface scattering that dominates SAR returns. These neural structures encode the surface of the objects in the form of a signed distance function learned from the sparse scattering data. Since estimating a smooth surface from a sparse and noisy point cloud is an ill-posed problem, we regularize the surface estimation by sampling points from the implicit surface representation during the training step. We demonstrate the model's ability to represent target scattering using measured and simulated data from single vehicles and a larger scene with a large number of vehicles. We conclude with future research directions calling for methods to learn complex-valued neural representations to enable synthesizing new collections from the volumetric neural implicit representation.

  </details>



- **HS-3D-NeRF: 3D Surface and Hyperspectral Reconstruction From Stationary Hyperspectral Images Using Multi-Channel NeRFs**  
  Kibon Ku, Talukder Z. Jubery, Adarsh Krishnamurthy, Baskar Ganapathysubramanian  
  _2026-02-18_ · https://arxiv.org/abs/2602.16950v1  
  <details><summary>Abstract</summary>

  Advances in hyperspectral imaging (HSI) and 3D reconstruction have enabled accurate, high-throughput characterization of agricultural produce quality and plant phenotypes, both essential for advancing agricultural sustainability and breeding programs. HSI captures detailed biochemical features of produce, while 3D geometric data substantially improves morphological analysis. However, integrating these two modalities at scale remains challenging, as conventional approaches involve complex hardware setups incompatible with automated phenotyping systems. Recent advances in neural radiance fields (NeRF) offer computationally efficient 3D reconstruction but typically require moving-camera setups, limiting throughput and reproducibility in standard indoor agricultural environments. To address these challenges, we introduce HSI-SC-NeRF, a stationary-camera multi-channel NeRF framework for high-throughput hyperspectral 3D reconstruction targeting postharvest inspection of agricultural produce. Multi-view hyperspectral data is captured using a stationary camera while the object rotates within a custom-built Teflon imaging chamber providing diffuse, uniform illumination. Object poses are estimated via ArUco calibration markers and transformed to the camera frame of reference through simulated pose transformations, enabling standard NeRF training on stationary-camera data. A multi-channel NeRF formulation optimizes reconstruction across all hyperspectral bands jointly using a composite spectral loss, supported by a two-stage training protocol that decouples geometric initialization from radiometric refinement. Experiments on three agricultural produce samples demonstrate high spatial reconstruction accuracy and strong spectral fidelity across the visible and near-infrared spectrum, confirming the suitability of HSI-SC-NeRF for integration into automated agricultural workflows.

  </details>



- **StereoAdapter-2: Globally Structure-Consistent Underwater Stereo Depth Estimation**  
  Zeyu Ren, Xiang Li, Yiran Wang, Zeyu Zhang, Hao Tang  
  _2026-02-18_ · https://arxiv.org/abs/2602.16915v1  
  <details><summary>Abstract</summary>

  Stereo depth estimation is fundamental to underwater robotic perception, yet suffers from severe domain shifts caused by wavelength-dependent light attenuation, scattering, and refraction. Recent approaches leverage monocular foundation models with GRU-based iterative refinement for underwater adaptation; however, the sequential gating and local convolutional kernels in GRUs necessitate multiple iterations for long-range disparity propagation, limiting performance in large-disparity and textureless underwater regions. In this paper, we propose StereoAdapter-2, which replaces the conventional ConvGRU updater with a novel ConvSS2D operator based on selective state space models. The proposed operator employs a four-directional scanning strategy that naturally aligns with epipolar geometry while capturing vertical structural consistency, enabling efficient long-range spatial propagation within a single update step at linear computational complexity. Furthermore, we construct UW-StereoDepth-80K, a large-scale synthetic underwater stereo dataset featuring diverse baselines, attenuation coefficients, and scattering parameters through a two-stage generative pipeline combining semantic-aware style transfer and geometry-consistent novel view synthesis. Combined with dynamic LoRA adaptation inherited from StereoAdapter, our framework achieves state-of-the-art zero-shot performance on underwater benchmarks with 17% improvement on TartanAir-UW and 7.2% improvment on SQUID, with real-world validation on the BlueROV2 platform demonstrates the robustness of our approach. Code: https://github.com/AIGeeksGroup/StereoAdapter-2. Website: https://aigeeksgroup.github.io/StereoAdapter-2.

  </details>



- **Subtractive Modulative Network with Learnable Periodic Activations**  
  Tiou Wang, Zhuoqian Yang, Markus Flierl, Mathieu Salzmann, Sabine Süsstrunk  
  _2026-02-18_ · https://arxiv.org/abs/2602.16337v1  
  <details><summary>Abstract</summary>

  We propose the Subtractive Modulative Network (SMN), a novel, parameter-efficient Implicit Neural Representation (INR) architecture inspired by classical subtractive synthesis. The SMN is designed as a principled signal processing pipeline, featuring a learnable periodic activation layer (Oscillator) that generates a multi-frequency basis, and a series of modulative mask modules (Filters) that actively generate high-order harmonics. We provide both theoretical analysis and empirical validation for our design. Our SMN achieves a PSNR of $40+$ dB on two image datasets, comparing favorably against state-of-the-art methods in terms of both reconstruction accuracy and parameter efficiency. Furthermore, consistent advantage is observed on the challenging 3D NeRF novel view synthesis task. Supplementary materials are available at https://inrainbws.github.io/smn/.

  </details>



- **NeRFscopy: Neural Radiance Fields for in-vivo Time-Varying Tissues from Endoscopy**  
  Laura Salort-Benejam, Antonio Agudo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15775v1  
  <details><summary>Abstract</summary>

  Endoscopy is essential in medical imaging, used for diagnosis, prognosis and treatment. Developing a robust dynamic 3D reconstruction pipeline for endoscopic videos could enhance visualization, improve diagnostic accuracy, aid in treatment planning, and guide surgery procedures. However, challenges arise due to the deformable nature of the tissues, the use of monocular cameras, illumination changes, occlusions and unknown camera trajectories. Inspired by neural rendering, we introduce NeRFscopy, a self-supervised pipeline for novel view synthesis and 3D reconstruction of deformable endoscopic tissues from a monocular video. NeRFscopy includes a deformable model with a canonical radiance field and a time-dependent deformation field parameterized by SE(3) transformations. In addition, the color images are efficiently exploited by introducing sophisticated terms to learn a 3D implicit model without assuming any template or pre-trained model, solely from data. NeRFscopy achieves accurate results in terms of novel view synthesis, outperforming competing methods across various challenging endoscopy scenes.

  </details>


