# 3D Reconstruction

_Updated: 2026-02-20 07:12 UTC_

Total papers shown: **9**


---

- **Neural Implicit Representations for 3D Synthetic Aperture Radar Imaging**  
  Nithin Sugavanam, Emre Ertin  
  _2026-02-19_ · https://arxiv.org/abs/2602.17556v1  
  <details><summary>Abstract</summary>

  Synthetic aperture radar (SAR) is a tomographic sensor that measures 2D slices of the 3D spatial Fourier transform of the scene. In many operational scenarios, the measured set of 2D slices does not fill the 3D space in the Fourier domain, resulting in significant artifacts in the reconstructed imagery. Traditionally, simple priors, such as sparsity in the image domain, are used to regularize the inverse problem. In this paper, we review our recent work that achieves state-of-the-art results in 3D SAR imaging employing neural structures to model the surface scattering that dominates SAR returns. These neural structures encode the surface of the objects in the form of a signed distance function learned from the sparse scattering data. Since estimating a smooth surface from a sparse and noisy point cloud is an ill-posed problem, we regularize the surface estimation by sampling points from the implicit surface representation during the training step. We demonstrate the model's ability to represent target scattering using measured and simulated data from single vehicles and a larger scene with a large number of vehicles. We conclude with future research directions calling for methods to learn complex-valued neural representations to enable synthesizing new collections from the volumetric neural implicit representation.

  </details>



- **3D Scene Rendering with Multimodal Gaussian Splatting**  
  Chi-Shiang Gau, Konstantinos D. Polyzos, Athanasios Bacharis, Saketh Madhuvarasu, Tara Javidi  
  _2026-02-19_ · https://arxiv.org/abs/2602.17124v1  
  <details><summary>Abstract</summary>

  3D scene reconstruction and rendering are core tasks in computer vision, with applications spanning industrial monitoring, robotics, and autonomous driving. Recent advances in 3D Gaussian Splatting (GS) and its variants have achieved impressive rendering fidelity while maintaining high computational and memory efficiency. However, conventional vision-based GS pipelines typically rely on a sufficient number of camera views to initialize the Gaussian primitives and train their parameters, typically incurring additional processing cost during initialization while falling short in conditions where visual cues are unreliable, such as adverse weather, low illumination, or partial occlusions. To cope with these challenges, and motivated by the robustness of radio-frequency (RF) signals to weather, lighting, and occlusions, we introduce a multimodal framework that integrates RF sensing, such as automotive radar, with GS-based rendering as a more efficient and robust alternative to vision-only GS rendering. The proposed approach enables efficient depth prediction from only sparse RF-based depth measurements, yielding a high-quality 3D point cloud for initializing Gaussian functions across diverse GS architectures. Numerical tests demonstrate the merits of judiciously incorporating RF sensing into GS pipelines, achieving high-fidelity 3D scene rendering driven by RF-informed structural accuracy.

  </details>



- **Benchmarking the Effects of Object Pose Estimation and Reconstruction on Robotic Grasping Success**  
  Varun Burde, Pavel Burget, Torsten Sattler  
  _2026-02-19_ · https://arxiv.org/abs/2602.17101v1  
  <details><summary>Abstract</summary>

  3D reconstruction serves as the foundational layer for numerous robotic perception tasks, including 6D object pose estimation and grasp pose generation. Modern 3D reconstruction methods for objects can produce visually and geometrically impressive meshes from multi-view images, yet standard geometric evaluations do not reflect how reconstruction quality influences downstream tasks such as robotic manipulation performance. This paper addresses this gap by introducing a large-scale, physics-based benchmark that evaluates 6D pose estimators and 3D mesh models based on their functional efficacy in grasping. We analyze the impact of model fidelity by generating grasps on various reconstructed 3D meshes and executing them on the ground-truth model, simulating how grasp poses generated with an imperfect model affect interaction with the real object. This assesses the combined impact of pose error, grasp robustness, and geometric inaccuracies from 3D reconstruction. Our results show that reconstruction artifacts significantly decrease the number of grasp pose candidates but have a negligible effect on grasping performance given an accurately estimated pose. Our results also reveal that the relationship between grasp success and pose error is dominated by spatial error, and even a simple translation error provides insight into the success of the grasping pose of symmetric objects. This work provides insight into how perception systems relate to object manipulation using robots.

  </details>



- **Cholec80-port: A Geometrically Consistent Trocar Port Segmentation Dataset for Robust Surgical Scene Understanding**  
  Shunsuke Kikuchi, Atsushi Kouno, Hiroki Matsuzaki  
  _2026-02-19_ · https://arxiv.org/abs/2602.17060v1  
  <details><summary>Abstract</summary>

  Trocar ports are camera-fixed, pseudo-static structures that can persistently occlude laparoscopic views and attract disproportionate feature points due to specular, textured surfaces. This makes ports particularly detrimental to geometry-based downstream pipelines such as image stitching, 3D reconstruction, and visual SLAM, where dynamic or non-anatomical outliers degrade alignment and tracking stability. Despite this practical importance, explicit port labels are rare in public surgical datasets, and existing annotations often violate geometric consistency by masking the central lumen (opening), even when anatomical regions are visible through it. We present Cholec80-port, a high-fidelity trocar port segmentation dataset derived from Cholec80, together with a rigorous standard operating procedure (SOP) that defines a port-sleeve mask excluding the central opening. We additionally cleanse and unify existing public datasets under the same SOP. Experiments demonstrate that geometrically consistent annotations substantially improve cross-dataset robustness beyond what dataset size alone provides.

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



- **Breaking the Sub-Millimeter Barrier: Eyeframe Acquisition from Color Images**  
  Manel Guzmán, Antonio Agudo  
  _2026-02-18_ · https://arxiv.org/abs/2602.16281v1  
  <details><summary>Abstract</summary>

  Eyeframe lens tracing is an important process in the optical industry that requires sub-millimeter precision to ensure proper lens fitting and optimal vision correction. Traditional frame tracers rely on mechanical tools that need precise positioning and calibration, which are time-consuming and require additional equipment, creating an inefficient workflow for opticians. This work presents a novel approach based on artificial vision that utilizes multi-view information. The proposed algorithm operates on images captured from an InVision system. The full pipeline includes image acquisition, frame segmentation to isolate the eyeframe from background, depth estimation to obtain 3D spatial information, and multi-view processing that integrates segmented RGB images with depth data for precise frame contour measurement. To this end, different configurations and variants are proposed and analyzed on real data, providing competitive measurements from still color images with respect to other solutions, while eliminating the need for specialized tracing equipment and reducing workflow complexity for optical technicians.

  </details>



- **NeRFscopy: Neural Radiance Fields for in-vivo Time-Varying Tissues from Endoscopy**  
  Laura Salort-Benejam, Antonio Agudo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15775v1  
  <details><summary>Abstract</summary>

  Endoscopy is essential in medical imaging, used for diagnosis, prognosis and treatment. Developing a robust dynamic 3D reconstruction pipeline for endoscopic videos could enhance visualization, improve diagnostic accuracy, aid in treatment planning, and guide surgery procedures. However, challenges arise due to the deformable nature of the tissues, the use of monocular cameras, illumination changes, occlusions and unknown camera trajectories. Inspired by neural rendering, we introduce NeRFscopy, a self-supervised pipeline for novel view synthesis and 3D reconstruction of deformable endoscopic tissues from a monocular video. NeRFscopy includes a deformable model with a canonical radiance field and a time-dependent deformation field parameterized by SE(3) transformations. In addition, the color images are efficiently exploited by introducing sophisticated terms to learn a 3D implicit model without assuming any template or pre-trained model, solely from data. NeRFscopy achieves accurate results in terms of novel view synthesis, outperforming competing methods across various challenging endoscopy scenes.

  </details>



- **An Industrial Dataset for Scene Acquisitions and Functional Schematics Alignment**  
  Flavien Armangeon, Thibaud Ehret, Enric Meinhardt-Llopis, Rafael Grompone von Gioi, Guillaume Thibault, Marc Petit, Gabriele Facciolo  
  _2026-02-17_ · https://arxiv.org/abs/2602.15584v1  
  <details><summary>Abstract</summary>

  Aligning functional schematics with 2D and 3D scene acquisitions is crucial for building digital twins, especially for old industrial facilities that lack native digital models. Current manual alignment using images and LiDAR data does not scale due to tediousness and complexity of industrial sites. Inconsistencies between schematics and reality, and the scarcity of public industrial datasets, make the problem both challenging and underexplored. This paper introduces IRIS-v2, a comprehensive dataset to support further research. It includes images, point clouds, 2D annotated boxes and segmentation masks, a CAD model, 3D pipe routing information, and the P&ID (Piping and Instrumentation Diagram). The alignment is experimented on a practical case study, aiming at reducing the time required for this task by combining segmentation and graph matching.

  </details>


