# Neural Rendering & View Synthesis

_Updated: 2026-03-31 07:42 UTC_

Total papers shown: **4**


---

- **Physically Inspired Gaussian Splatting for HDR Novel View Synthesis**  
  Huimin Zeng, Yue Bai, Hailing Wang, Yun Fu  
  _2026-03-30_ · https://arxiv.org/abs/2603.28020v1  
  <details><summary>Abstract</summary>

  High dynamic range novel view synthesis (HDR-NVS) reconstructs scenes with dynamic details by fusing multi-exposure low dynamic range (LDR) views, yet it struggles to capture ambient illumination-dependent appearance. Implicitly supervising HDR content by constraining tone-mapped results fails in correcting abnormal HDR values, and results in limited gradients for Gaussians in under/over-exposed regions. To this end, we introduce PhysHDR-GS, a physically inspired HDR-NVS framework that models scene appearance via intrinsic reflectance and adjustable ambient illumination. PhysHDR-GS employs a complementary image-exposure (IE) branch and Gaussian-illumination (GI) branch to faithfully reproduce standard camera observations and capture illumination-dependent appearance changes, respectively. During training, the proposed cross-branch HDR consistency loss provides explicit supervision for HDR content, while an illumination-guided gradient scaling strategy mitigates exposure-biased gradient starvation and reduces under-densified representations. Experimental results across realistic and synthetic datasets demonstrate our superiority in reconstructing HDR details (e.g., a PSNR gain of 2.04 dB over HDR-GS), while maintaining real-time rendering speed (up to 76 FPS). Code and models are available at https://huimin-zeng.github.io/PhysHDR-GS/.

  </details>



- **RehearsalNeRF: Decoupling Intrinsic Neural Fields of Dynamic Illuminations for Scene Editing**  
  Changyeon Won, Hyunjun Jung, Jungu Cho, Seonmi Park, Chi-Hoon Lee, Hae-Gon Jeon  
  _2026-03-30_ · https://arxiv.org/abs/2603.27948v1  
  <details><summary>Abstract</summary>

  Although there has been significant progress in neural radiance fields, an issue on dynamic illumination changes still remains unsolved. Different from relevant works that parameterize time-variant/-invariant components in scenes, subjects' radiance is highly entangled with their own emitted radiance and lighting colors in spatio-temporal domain. In this paper, we present a new effective method to learn disentangled neural fields under the severe illumination changes, named RehearsalNeRF. Our key idea is to leverage scenes captured under stable lighting like rehearsal stages, easily taken before dynamic illumination occurs, to enforce geometric consistency between the different lighting conditions. In particular, RehearsalNeRF employs a learnable vector for lighting effects which represents illumination colors in a temporal dimension and is used to disentangle projected light colors from scene radiance. Furthermore, our RehearsalNeRF is also able to reconstruct the neural fields of dynamic objects by simply adopting off-the-shelf interactive masks. To decouple the dynamic objects, we propose a new regularization leveraging optical flow, which provides coarse supervision for the color disentanglement. We demonstrate the effectiveness of RehearsalNeRF by showing robust performances on novel view synthesis and scene editing under dynamic illumination conditions. Our source code and video datasets will be publicly available.

  </details>



- **Poppy: Polarization-based Plug-and-Play Guidance for Enhancing Monocular Normal Estimation**  
  Irene Kim, Sai Tanmay Reddy Chakkera, Alexandros Graikos, Dimitris Samaras, Akshat Dave  
  _2026-03-29_ · https://arxiv.org/abs/2603.27891v1  
  <details><summary>Abstract</summary>

  Monocular surface normal estimators trained on large-scale RGB-normal data often perform poorly in the edge cases of reflective, textureless, and dark surfaces. Polarization encodes surface orientation independently of texture and albedo, offering a physics-based complement for these cases. Existing polarization methods, however, require multi-view capture or specialized training data, limiting generalization. We introduce Poppy, a training-free framework that refines normals from any frozen RGB backbone using single-shot polarization measurements at test time. Keeping backbone weights frozen, Poppy optimizes per-pixel offsets to the input RGB and output normal along with a learned reflectance decomposition. A differentiable rendering layer converts the refined normals into polarization predictions and penalizes mismatches with the observed signal. Across seven benchmarks and three backbone architectures (diffusion, flow, and feed-forward), Poppy reduces mean angular error by 23-26% on synthetic data and 6-16% on real data. These results show that guiding learned RGB-based normal estimators with polarization cues at test time refines normals on challenging surfaces without retraining.

  </details>



- **From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis**  
  Ranran Huang, Weixun Luo, Ye Mao, Krystian Mikolajczyk  
  _2026-03-29_ · https://arxiv.org/abs/2603.27455v1  
  <details><summary>Abstract</summary>

  In this paper, we introduce NAS3R, a self-supervised feed-forward framework that jointly learns explicit 3D geometry and camera parameters with no ground-truth annotations and no pretrained priors. During training, NAS3R reconstructs 3D Gaussians from uncalibrated and unposed context views and renders target views using its self-predicted camera parameters, enabling self-supervised training from 2D photometric supervision. To ensure stable convergence, NAS3R integrates reconstruction and camera prediction within a shared transformer backbone regulated by masked attention, and adopts a depth-based Gaussian formulation that facilitates well-conditioned optimization. The framework is compatible with state-of-the-art supervised 3D reconstruction architectures and can incorporate pretrained priors or intrinsic information when available. Extensive experiments show that NAS3R achieves superior results to other self-supervised methods, establishing a scalable and geometry-aware paradigm for 3D reconstruction from unconstrained data. Code and models are publicly available at https://ranrhuang.github.io/nas3r/.

  </details>


