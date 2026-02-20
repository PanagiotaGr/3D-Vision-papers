# Neural Rendering & View Synthesis

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **4**


---

- **3D Scene Rendering with Multimodal Gaussian Splatting**  
  Chi-Shiang Gau, Konstantinos D. Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Tara Javidi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17124v1  
  <details><summary>Abstract</summary>

  3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.

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


