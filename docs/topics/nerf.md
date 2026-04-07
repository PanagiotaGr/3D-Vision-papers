# NeRF & Neural Radiance Fields

_Updated: 2026-04-07 07:47 UTC_

Total papers shown: **2**


---

- **PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis**  
  Inseong Choi, Siwoo Lee, Seung-Hun Nam, Soohwan Song  
  _2026-04-06_ · https://arxiv.org/abs/2604.04576v1  
  <details><summary>Abstract</summary>

  Diffusion models are promising for sparse-view novel view synthesis (NVS), as they can generate pseudo-ground-truth views to aid 3D reconstruction pipelines like 3D Gaussian Splatting (3DGS). However, these synthesized images often contain photometric and geometric inconsistencies, and their direct use for supervision can impair reconstruction. To address this, we propose Partial-Reference Image Quality Assessment (PR-IQA), a framework that evaluates diffusion-generated views using reference images from different poses, eliminating the need for ground truth. PR-IQA first computes a geometrically consistent partial quality map in overlapping regions. It then performs quality completion to inpaint this partial map into a dense, full-image map. This completion is achieved via a cross-attention mechanism that incorporates reference-view context, ensuring cross-view consistency and enabling thorough quality assessment. When integrated into a diffusion-augmented 3DGS pipeline, PR-IQA restricts supervision to high-confidence regions identified by its quality maps. Experiments demonstrate that PR-IQA outperforms existing IQA methods, achieving full-reference-level accuracy without ground-truth supervision. Thus, our quality-aware 3DGS approach more effectively filters inconsistencies, producing superior 3D reconstructions and NVS results.The project page is available at https://kakaomacao.github.io/pr-iqa-project-page/.

  </details>



- **Real-time Neural Six-way Lightmaps**  
  Wei Li, Hanxiao Sun, Tao Huang, Haoxiang Wang, Tongtong Wang, Zherong Pan, Kui Wu  
  _2026-04-04_ · https://arxiv.org/abs/2604.03748v1  
  <details><summary>Abstract</summary>

  Participating media are a pervasive and intriguing visual effect in virtual environments. Unfortunately, rendering such phenomena in real-time is notoriously difficult due to the computational expense of estimating the volume rendering equation. While the six-way lightmaps technique has been widely used in video games to render smoke with a camera-oriented billboard and approximate lighting effects using six precomputed lightmaps, achieving a balance between realism and efficiency, it is limited to pre-simulated animation sequences and is ignorant of camera movement. In this work, we propose a neural six-way lightmaps method to strike a long-sought balance between dynamics and visual realism. Our approach first generates a guiding map from the camera view using ray marching with a large sampling distance to approximate smoke scattering and silhouette. Then, given a guiding map, we train a neural network to predict the corresponding six-way lightmaps. The resulting lightmaps can be seamlessly used in existing game engine pipelines. This approach supports visually appealing rendering effects while enabling real-time user interactivity, including smoke-obstacle interaction, camera movement, and light change. By conducting a series of comprehensive benchmarks, we demonstrate that our method is well-suited for real-time applications, such as games and VR/AR.

  </details>


