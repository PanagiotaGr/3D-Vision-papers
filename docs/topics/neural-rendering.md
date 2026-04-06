# Neural Rendering & View Synthesis

_Updated: 2026-04-06 08:00 UTC_

Total papers shown: **2**


---

- **GenSmoke-GS: A Multi-Stage Method for Novel View Synthesis from Smoke-Degraded Images Using a Generative Model**  
  Qida Cao, Xinyuan Hu, Changyue Shi, Jiajun Ding, Zhou Yu, Jun Yu  
  _2026-04-03_ · https://arxiv.org/abs/2604.03039v1  
  <details><summary>Abstract</summary>

  This paper describes our method for Track 2 of the NTIRE 2026 3D Restoration and Reconstruction (3DRR) Challenge on smoke-degraded images. In this task, smoke reduces image visibility and weakens the cross-view consistency required by scene optimization and rendering. We address this problem with a multi-stage pipeline consisting of image restoration, dehazing, MLLM-based enhancement, 3DGS-MCMC optimization, and averaging over repeated runs. The main purpose of the pipeline is to improve visibility before rendering while limiting scene-content changes across input views. Experimental results on the challenge benchmark show improved quantitative performance and better visual quality than the provided baselines. The code is available at https://github.com/plbbl/GenSmoke-GS. Our method achieved a ranking of 1 out of 14 participants in Track 2 of the NTIRE 3DRR Challenge, as reported on the official competition website: https://www.codabench.org/competitions/13993/#/results-tab.

  </details>



- **GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes**  
  Mijeong Kim, Jungtaek Kim, Bohyung Han  
  _2026-04-03_ · https://arxiv.org/abs/2604.02915v1  
  <details><summary>Abstract</summary>

  We present GP-4DGS, a novel framework that integrates Gaussian Processes (GPs) into 4D Gaussian Splatting (4DGS) for principled probabilistic modeling of dynamic scenes. While existing 4DGS methods focus on deterministic reconstruction, they are inherently limited in capturing motion ambiguity and lack mechanisms to assess prediction reliability. By leveraging the kernel-based probabilistic nature of GPs, our approach introduces three key capabilities: (i) uncertainty quantification for motion predictions, (ii) motion estimation for unobserved or sparsely sampled regions, and (iii) temporal extrapolation beyond observed training frames. To scale GPs to the large number of Gaussian primitives in 4DGS, we design spatio-temporal kernels that capture the correlation structure of deformation fields and adopt variational Gaussian Processes with inducing points for tractable inference. Our experiments show that GP-4DGS enhances reconstruction quality while providing reliable uncertainty estimates that effectively identify regions of high motion ambiguity. By addressing these challenges, our work takes a meaningful step toward bridging probabilistic modeling and neural graphics.

  </details>


