# NeRF & Neural Radiance Fields

_Updated: 2026-02-12 07:16 UTC_

Total papers shown: **6**


---

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



- **Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit**  
  Zhendong Wang, Cihan Ruan, Jingchuan Xiao, Chuqing Shi, Wei Jiang, Wei Wang, Wenjie Liu, Nam Ling  
  _2026-02-09_ · https://arxiv.org/abs/2602.08909v1  
  <details><summary>Abstract</summary>

  We investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multi-view optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patterns: mixture-structured scales and bimodal radiance across diverse scenes. To understand what determines these parameters, we apply learnability probes by training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental density-stratification. Dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals the dual character of RORs: geometric primitives where point clouds suffice, and view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.

  </details>



- **Rotated Lights for Consistent and Efficient 2D Gaussians Inverse Rendering**  
  Geng Lin, Matthias Zwicker  
  _2026-02-09_ · https://arxiv.org/abs/2602.08724v1  
  <details><summary>Abstract</summary>

  Inverse rendering aims to decompose a scene into its geometry, material properties and light conditions under a certain rendering model. It has wide applications like view synthesis, relighting, and scene editing. In recent years, inverse rendering methods have been inspired by view synthesis approaches like neural radiance fields and Gaussian splatting, which are capable of efficiently decomposing a scene into its geometry and radiance. They then further estimate the material and lighting that lead to the observed scene radiance. However, the latter step is highly ambiguous and prior works suffer from inaccurate color and baked shadows in their albedo estimation albeit their regularization. To this end, we propose RotLight, a simple capturing setup, to address the ambiguity. Compared to a usual capture, RotLight only requires the object to be rotated several times during the process. We show that as few as two rotations is effective in reducing artifacts. To further improve 2DGS-based inverse rendering, we additionally introduce a proxy mesh that not only allows accurate incident light tracing, but also enables a residual constraint and improves global illumination handling. We demonstrate with both synthetic and real world datasets that our method achieves superior albedo estimation while keeping efficient computation.

  </details>


