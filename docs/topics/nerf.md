# NeRF & Neural Radiance Fields

_Updated: 2026-02-13 07:13 UTC_

Total papers shown: **6**


---

- **Projected Representation Conditioning for High-fidelity Novel View Synthesis**  
  Min-Seop Kwak, Minkyung Kwon, Jinhyeok Choi, Jiho Park, Seungryong Kim  
  _2026-02-12_ · https://arxiv.org/abs/2602.12003v1  
  <details><summary>Abstract</summary>

  We propose a novel framework for diffusion-based novel view synthesis in which we leverage external representations as conditions, harnessing their geometric and semantic correspondence properties for enhanced geometric consistency in generated novel viewpoints. First, we provide a detailed analysis exploring the correspondence capabilities emergent in the spatial attention of external visual representations. Building from these insights, we propose a representation-guided novel view synthesis through dedicated representation projection modules that inject external representations into the diffusion process, a methodology named ReNoV, short for representation-guided novel view synthesis. Our experiments show that this design yields marked improvements in both reconstruction fidelity and inpainting quality, outperforming prior diffusion-based novel-view methods on standard benchmarks and enabling robust synthesis from sparse, unposed image collections.

  </details>



- **Fighting MRI Anisotropy: Learning Multiple Cardiac Shapes From a Single Implicit Neural Representation**  
  Carolina Brás, Soufiane Ben Haddou, Thijs P. Kuipers, Laura Alvarez-Florez, R. Nils Planken, Fleur V. Y. Tjong, Connie Bezzina, Ivana Išgum  
  _2026-02-11_ · https://arxiv.org/abs/2602.11436v1  
  <details><summary>Abstract</summary>

  The anisotropic nature of short-axis (SAX) cardiovascular magnetic resonance imaging (CMRI) limits cardiac shape analysis. To address this, we propose to leverage near-isotropic, higher resolution computed tomography angiography (CTA) data of the heart. We use this data to train a single neural implicit function to jointly represent cardiac shapes from CMRI at any resolution. We evaluate the method for the reconstruction of right ventricle (RV) and myocardium (MYO), where MYO simultaneously models endocardial and epicardial left-ventricle surfaces. Since high-resolution SAX reference segmentations are unavailable, we evaluate performance by extracting a 4-chamber (4CH) slice of RV and MYO from their reconstructed shapes. When compared with the reference 4CH segmentation masks from CMRI, our method achieved a Dice similarity coefficient of 0.91 $\pm$ 0.07 and 0.75 $\pm$ 0.13, and a Hausdorff distance of 6.21 $\pm$ 3.97 mm and 7.53 $\pm$ 5.13 mm for RV and MYO, respectively. Quantitative and qualitative assessment demonstrate the model's ability to reconstruct accurate, smooth and anatomically plausible shapes, supporting improvements in cardiac shape analysis.

  </details>



- **SurfPhase: 3D Interfacial Dynamics in Two-Phase Flows from Sparse Videos**  
  Yue Gao, Hong-Xing Yu, Sanghyeon Chang, Qianxi Fu, Bo Zhu, Yoonjin Won, Juan Carlos Niebles, Jiajun Wu  
  _2026-02-11_ · https://arxiv.org/abs/2602.11154v1  
  <details><summary>Abstract</summary>

  Interfacial dynamics in two-phase flows govern momentum, heat, and mass transfer, yet remain difficult to measure experimentally. Classical techniques face intrinsic limitations near moving interfaces, while existing neural rendering methods target single-phase flows with diffuse boundaries and cannot handle sharp, deformable liquid-vapor interfaces. We propose SurfPhase, a novel model for reconstructing 3D interfacial dynamics from sparse camera views. Our approach integrates dynamic Gaussian surfels with a signed distance function formulation for geometric consistency, and leverages a video diffusion model to synthesize novel-view videos to refine reconstruction from sparse observations. We evaluate on a new dataset of high-speed pool boiling videos, demonstrating high-quality view synthesis and velocity estimation from only two camera views. Project website: https://yuegao.me/SurfPhase.

  </details>



- **Towards Learning a Generalizable 3D Scene Representation from 2D Observations**  
  Martin Gromniak, Jan-Gerrit Habekost, Sebastian Kamp, Sven Magg, Stefan Wermter  
  _2026-02-11_ · https://arxiv.org/abs/2602.10943v1  
  <details><summary>Abstract</summary>

  We introduce a Generalizable Neural Radiance Field approach for predicting 3D workspace occupancy from egocentric robot observations. Unlike prior methods operating in camera-centric coordinates, our model constructs occupancy representations in a global workspace frame, making it directly applicable to robotic manipulation. The model integrates flexible source views and generalizes to unseen object arrangements without scene-specific finetuning. We demonstrate the approach on a humanoid robot and evaluate predicted geometry against 3D sensor ground truth. Trained on 40 real scenes, our model achieves 26mm reconstruction error, including occluded regions, validating its ability to infer complete 3D occupancy beyond traditional stereo vision methods.

  </details>



- **Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings**  
  Tianxiang Dai, Jonathan Fan  
  _2026-02-11_ · https://arxiv.org/abs/2602.10495v1  
  <details><summary>Abstract</summary>

  Multi-Resolution Hash Encoding (MHE), the foundational technique behind Instant Neural Graphics Primitives, provides a powerful parameterization for neural fields. However, its spatial behavior lacks rigorous understanding from a physical systems perspective, leading to reliance on heuristics for hyperparameter selection. This work introduces a novel analytical approach that characterizes MHE by examining its Point Spread Function (PSF), which is analogous to the Green's function of the system. This methodology enables a quantification of the encoding's spatial resolution and fidelity. We derive a closed-form approximation for the collision-free PSF, uncovering inherent grid-induced anisotropy and a logarithmic spatial profile. We establish that the idealized spatial bandwidth, specifically the Full Width at Half Maximum (FWHM), is determined by the average resolution, $N_{\text{avg}}$. This leads to a counterintuitive finding: the effective resolution of the model is governed by the broadened empirical FWHM (and therefore $N_{\text{avg}}$), rather than the finest resolution $N_{\max}$, a broadening effect we demonstrate arises from optimization dynamics. Furthermore, we analyze the impact of finite hash capacity, demonstrating how collisions introduce speckle noise and degrade the Signal-to-Noise Ratio (SNR). Leveraging these theoretical insights, we propose Rotated MHE (R-MHE), an architecture that applies distinct rotations to the input coordinates at each resolution level. R-MHE mitigates anisotropy while maintaining the efficiency and parameter count of the original MHE. This study establishes a methodology based on physical principles that moves beyond heuristics to characterize and optimize MHE.

  </details>



- **CompSplat: Compression-aware 3D Gaussian Splatting for Real-world Video**  
  Hojun Song, Heejung Choi, Aro Kim, Chae-yeong Song, Gahyeon Kim, Soo Ye Kim, Jaehyup Lee, Sang-hyo Park  
  _2026-02-10_ · https://arxiv.org/abs/2602.09816v1  
  <details><summary>Abstract</summary>

  High-quality novel view synthesis (NVS) from real-world videos is crucial for applications such as cultural heritage preservation, digital twins, and immersive media. However, real-world videos typically contain long sequences with irregular camera trajectories and unknown poses, leading to pose drift, feature misalignment, and geometric distortion during reconstruction. Moreover, lossy compression amplifies these issues by introducing inconsistencies that gradually degrade geometry and rendering quality. While recent studies have addressed either long-sequence NVS or unposed reconstruction, compression-aware approaches still focus on specific artifacts or limited scenarios, leaving diverse compression patterns in long videos insufficiently explored. In this paper, we propose CompSplat, a compression-aware training framework that explicitly models frame-wise compression characteristics to mitigate inter-frame inconsistency and accumulated geometric errors. CompSplat incorporates compression-aware frame weighting and an adaptive pruning strategy to enhance robustness and geometric consistency, particularly under heavy compression. Extensive experiments on challenging benchmarks, including Tanks and Temples, Free, and Hike, demonstrate that CompSplat achieves state-of-the-art rendering quality and pose accuracy, significantly surpassing most recent state-of-the-art NVS approaches under severe compression conditions.

  </details>


