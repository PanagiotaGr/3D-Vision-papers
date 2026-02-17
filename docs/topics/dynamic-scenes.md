# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-02-17 07:13 UTC_

Total papers shown: **3**


---

- **Event-based Visual Deformation Measurement**  
  Yuliang Wu, Wei Zhai, Yuxin Cui, Tiesong Zhao, Yang Cao, Zheng-Jun Zha  
  _2026-02-16_ · https://arxiv.org/abs/2602.14376v1  
  <details><summary>Abstract</summary>

  Visual Deformation Measurement (VDM) aims to recover dense deformation fields by tracking surface motion from camera observations. Traditional image-based methods rely on minimal inter-frame motion to constrain the correspondence search space, which limits their applicability to highly dynamic scenes or necessitates high-speed cameras at the cost of prohibitive storage and computational overhead. We propose an event-frame fusion framework that exploits events for temporally dense motion cues and frames for spatially dense precise estimation. Revisiting the solid elastic modeling prior, we propose an Affine Invariant Simplicial (AIS) framework. It partitions the deformation field into linearized sub-regions with low-parametric representation, effectively mitigating motion ambiguities arising from sparse and noisy events. To speed up parameter searching and reduce error accumulation, a neighborhood-greedy optimization strategy is introduced, enabling well-converged sub-regions to guide their poorly-converged neighbors, effectively suppress local error accumulation in long-term dense tracking. To evaluate the proposed method, a benchmark dataset with temporally aligned event streams and frames is established, encompassing over 120 sequences spanning diverse deformation scenarios. Experimental results show that our method outperforms the state-of-the-art baseline by 1.6% in survival rate. Remarkably, it achieves this using only 18.9% of the data storage and processing resources of high-speed video methods.

  </details>



- **Flow4R: Unifying 4D Reconstruction and Tracking with Scene Flow**  
  Shenhan Qian, Ganlin Zhang, Shangzhe Wu, Daniel Cremers  
  _2026-02-15_ · https://arxiv.org/abs/2602.14021v1  
  <details><summary>Abstract</summary>

  Reconstructing and tracking dynamic 3D scenes remains a fundamental challenge in computer vision. Existing approaches often decouple geometry from motion: multi-view reconstruction methods assume static scenes, while dynamic tracking frameworks rely on explicit camera pose estimation or separate motion models. We propose Flow4R, a unified framework that treats camera-space scene flow as the central representation linking 3D structure, object motion, and camera motion. Flow4R predicts a minimal per-pixel property set-3D point position, scene flow, pose weight, and confidence-from two-view inputs using a Vision Transformer. This flow-centric formulation allows local geometry and bidirectional motion to be inferred symmetrically with a shared decoder in a single forward pass, without requiring explicit pose regressors or bundle adjustment. Trained jointly on static and dynamic datasets, Flow4R achieves state-of-the-art performance on 4D reconstruction and tracking tasks, demonstrating the effectiveness of the flow-central representation for spatiotemporal scene understanding.

  </details>



- **Gaussian Sequences with Multi-Scale Dynamics for 4D Reconstruction from Monocular Casual Videos**  
  Can Li, Jie Gu, Jingmin Chen, Fangzhou Qiu, Lei Sun  
  _2026-02-14_ · https://arxiv.org/abs/2602.13806v1  
  <details><summary>Abstract</summary>

  Understanding dynamic scenes from casual videos is critical for scalable robot learning, yet four-dimensional (4D) reconstruction under strictly monocular settings remains highly ill-posed. To address this challenge, our key insight is that real-world dynamics exhibits a multi-scale regularity from object to particle level. To this end, we design the multi-scale dynamics mechanism that factorizes complex motion fields. Within this formulation, we propose Gaussian sequences with multi-scale dynamics, a novel representation for dynamic 3D Gaussians derived through compositions of multi-level motion. This layered structure substantially alleviates ambiguity of reconstruction and promotes physically plausible dynamics. We further incorporate multi-modal priors from vision foundation models to establish complementary supervision, constraining the solution space and improving the reconstruction fidelity. Our approach enables accurate and globally consistent 4D reconstruction from monocular casual videos. Experiments of dynamic novel-view synthesis (NVS) on benchmark and real-world manipulation datasets demonstrate considerable improvements over existing methods.

  </details>


