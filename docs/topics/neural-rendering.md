# Neural Rendering & View Synthesis

_Updated: 2026-02-10 07:20 UTC_

Total papers shown: **6**


---

- **Grow with the Flow: 4D Reconstruction of Growing Plants with Gaussian Flow Fields**  
  Weihan Luo, Lily Goli, Sherwin Bahmani, Felix Taubner, Andrea Tagliasacchi, David B. Lindell  
  _2026-02-09_ · https://arxiv.org/abs/2602.08958v1  
  <details><summary>Abstract</summary>

  Modeling the time-varying 3D appearance of plants during their growth poses unique challenges: unlike many dynamic scenes, plants generate new geometry over time as they expand, branch, and differentiate. Recent motion modeling techniques are ill-suited to this problem setting. For example, deformation fields cannot introduce new geometry, and 4D Gaussian splatting constrains motion to a linear trajectory in space and time and cannot track the same set of Gaussians over time. Here, we introduce a 3D Gaussian flow field representation that models plant growth as a time-varying derivative over Gaussian parameters -- position, scale, orientation, color, and opacity -- enabling nonlinear and continuous-time growth dynamics. To initialize a sufficient set of Gaussian primitives, we reconstruct the mature plant and learn a process of reverse growth, effectively simulating the plant's developmental history in reverse. Our approach achieves superior image quality and geometric accuracy compared to prior methods on multi-view timelapse datasets of plant growth, providing a new approach for appearance modeling of growing 3D structures.

  </details>



- **Analysis of Converged 3D Gaussian Splatting Solutions: Density Effects and Prediction Limit**  
  Zhendong Wang, Cihan Ruan, Jingchuan Xiao, Chuqing Shi, Wei Jiang, Wei Wang, Wenjie Liu, Nam Ling  
  _2026-02-09_ · https://arxiv.org/abs/2602.08909v1  
  <details><summary>Abstract</summary>

  We investigate what structure emerges in 3D Gaussian Splatting (3DGS) solutions from standard multi-view optimization. We term these Rendering-Optimal References (RORs) and analyze their statistical properties, revealing stable patterns: mixture-structured scales and bimodal radiance across diverse scenes. To understand what determines these parameters, we apply learnability probes by training predictors to reconstruct RORs from point clouds without rendering supervision. Our analysis uncovers fundamental density-stratification. Dense regions exhibit geometry-correlated parameters amenable to render-free prediction, while sparse regions show systematic failure across architectures. We formalize this through variance decomposition, demonstrating that visibility heterogeneity creates covariance-dominated coupling between geometric and appearance parameters in sparse regions. This reveals the dual character of RORs: geometric primitives where point clouds suffice, and view synthesis primitives where multi-view constraints are essential. We provide density-aware strategies that improve training robustness and discuss architectural implications for systems that adaptively balance feed-forward prediction and rendering-based refinement.

  </details>



- **VedicTHG: Symbolic Vedic Computation for Low-Resource Talking-Head Generation in Educational Avatars**  
  Vineet Kumar Rakesh, Ahana Bhattacharjee, Soumya Mazumdar, Tapas Samanta, Hemendra Kumar Pandey, Amitabha Das, Sarbajit Pal  
  _2026-02-09_ · https://arxiv.org/abs/2602.08775v1  
  <details><summary>Abstract</summary>

  Talking-head avatars are increasingly adopted in educational technology to deliver content with social presence and improved engagement. However, many recent talking-head generation (THG) methods rely on GPU-centric neural rendering, large training sets, or high-capacity diffusion models, which limits deployment in offline or resource-constrained learning environments. A deterministic and CPU-oriented THG framework is described, termed Symbolic Vedic Computation, that converts speech to a time-aligned phoneme stream, maps phonemes to a compact viseme inventory, and produces smooth viseme trajectories through symbolic coarticulation inspired by Vedic sutra Urdhva Tiryakbhyam. A lightweight 2D renderer performs region-of-interest (ROI) warping and mouth compositing with stabilization to support real-time synthesis on commodity CPUs. Experiments report synchronization accuracy, temporal stability, and identity consistency under CPU-only execution, alongside benchmarking against representative CPU-feasible baselines. Results indicate that acceptable lip-sync quality can be achieved while substantially reducing computational load and latency, supporting practical educational avatars on low-end hardware. GitHub: https://vineetkumarrakesh.github.io/vedicthg

  </details>



- **Rotated Lights for Consistent and Efficient 2D Gaussians Inverse Rendering**  
  Geng Lin, Matthias Zwicker  
  _2026-02-09_ · https://arxiv.org/abs/2602.08724v1  
  <details><summary>Abstract</summary>

  Inverse rendering aims to decompose a scene into its geometry, material properties and light conditions under a certain rendering model. It has wide applications like view synthesis, relighting, and scene editing. In recent years, inverse rendering methods have been inspired by view synthesis approaches like neural radiance fields and Gaussian splatting, which are capable of efficiently decomposing a scene into its geometry and radiance. They then further estimate the material and lighting that lead to the observed scene radiance. However, the latter step is highly ambiguous and prior works suffer from inaccurate color and baked shadows in their albedo estimation albeit their regularization. To this end, we propose RotLight, a simple capturing setup, to address the ambiguity. Compared to a usual capture, RotLight only requires the object to be rotated several times during the process. We show that as few as two rotations is effective in reducing artifacts. To further improve 2DGS-based inverse rendering, we additionally introduce a proxy mesh that not only allows accurate incident light tracing, but also enables a residual constraint and improves global illumination handling. We demonstrate with both synthetic and real world datasets that our method achieves superior albedo estimation while keeping efficient computation.

  </details>



- **FLAG-4D: Flow-Guided Local-Global Dual-Deformation Model for 4D Reconstruction**  
  Guan Yuan Tan, Ngoc Tuan Vu, Arghya Pal, Sailaja Rajanala, Raphael Phan C. -W., Mettu Srinivas, Chee-Ming Ting  
  _2026-02-09_ · https://arxiv.org/abs/2602.08558v1  
  <details><summary>Abstract</summary>

  We introduce FLAG-4D, a novel framework for generating novel views of dynamic scenes by reconstructing how 3D Gaussian primitives evolve through space and time. Existing methods typically rely on a single Multilayer Perceptron (MLP) to model temporal deformations, and they often struggle to capture complex point motions and fine-grained dynamic details consistently over time, especially from sparse input views. Our approach, FLAG-4D, overcomes this by employing a dual-deformation network that dynamically warps a canonical set of 3D Gaussians over time into new positions and anisotropic shapes. This dual-deformation network consists of an Instantaneous Deformation Network (IDN) for modeling fine-grained, local deformations and a Global Motion Network (GMN) for capturing long-range dynamics, refined through mutual learning. To ensure these deformations are both accurate and temporally smooth, FLAG-4D incorporates dense motion features from a pretrained optical flow backbone. We fuse these motion cues from adjacent timeframes and use a deformation-guided attention mechanism to align this flow information with the current state of each evolving 3D Gaussian. Extensive experiments demonstrate that FLAG-4D achieves higher-fidelity and more temporally coherent reconstructions with finer detail preservation than state-of-the-art methods.

  </details>



- **Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields**  
  Berthy T. Feng, Andrew A. Chael, David Bromley, Aviad Levis, William T. Freeman, Katherine L. Bouman  
  _2026-02-08_ · https://arxiv.org/abs/2602.08029v1  
  <details><summary>Abstract</summary>

  With the success of static black-hole imaging, the next frontier is the dynamic and 3D imaging of black holes. Recovering the dynamic 3D gas near a black hole would reveal previously-unseen parts of the universe and inform new physics models. However, only sparse radio measurements from a single viewpoint are possible, making the dynamic 3D reconstruction problem significantly ill-posed. Previously, BH-NeRF addressed the ill-posed problem by assuming Keplerian dynamics of the gas, but this assumption breaks down near the black hole, where the strong gravitational pull of the black hole and increased electromagnetic activity complicate fluid dynamics. To overcome the restrictive assumptions of BH-NeRF, we propose PI-DEF, a physics-informed approach that uses differentiable neural rendering to fit a 4D (time + 3D) emissivity field given EHT measurements. Our approach jointly reconstructs the 3D velocity field with the 4D emissivity field and enforces the velocity as a soft constraint on the dynamics of the emissivity. In experiments on simulated data, we find significantly improved reconstruction accuracy over both BH-NeRF and a physics-agnostic approach. We demonstrate how our method may be used to estimate other physics parameters of the black hole, such as its spin.

  </details>


