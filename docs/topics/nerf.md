# NeRF & Neural Radiance Fields

_Updated: 2026-02-02 07:16 UTC_

Total papers shown: **3**


---

- **EAG-PT: Emission-Aware Gaussians and Path Tracing for Indoor Scene Reconstruction and Editing**  
  Xijie Yang, Mulin Yu, Changjian Jiang, Kerui Ren, Tao Lu, Jiangmiao Pang, Dahua Lin, Bo Dai, Linning Xu  
  _2026-01-30_ · https://arxiv.org/abs/2601.23065v1  
  <details><summary>Abstract</summary>

  Recent reconstruction methods based on radiance field such as NeRF and 3DGS reproduce indoor scenes with high visual fidelity, but break down under scene editing due to baked illumination and the lack of explicit light transport. In contrast, physically based inverse rendering relies on mesh representations and path tracing, which enforce correct light transport but place strong requirements on geometric fidelity, becoming a practical bottleneck for real indoor scenes. In this work, we propose Emission-Aware Gaussians and Path Tracing (EAG-PT), aiming for physically based light transport with a unified 2D Gaussian representation. Our design is based on three cores: (1) using 2D Gaussians as a unified scene representation and transport-friendly geometry proxy that avoids reconstructed mesh, (2) explicitly separating emissive and non-emissive components during reconstruction for further scene editing, and (3) decoupling reconstruction from final rendering by using efficient single-bounce optimization and high-quality multi-bounce path tracing after scene editing. Experiments on synthetic and real indoor scenes show that EAG-PT produces more natural and physically consistent renders after editing than radiant scene reconstructions, while preserving finer geometric detail and avoiding mesh-induced artifacts compared to mesh-based inverse path tracing. These results suggest promising directions for future use in interior design, XR content creation, and embodied AI.

  </details>



- **Under-Canopy Terrain Reconstruction in Dense Forests Using RGB Imaging and Neural 3D Reconstruction**  
  Refael Sheffer, Chen Pinchover, Haim Zisman, Dror Ozeri, Roee Litman  
  _2026-01-30_ · https://arxiv.org/abs/2601.22861v1  
  <details><summary>Abstract</summary>

  Mapping the terrain and understory hidden beneath dense forest canopies is of great interest for numerous applications such as search and rescue, trail mapping, forest inventory tasks, and more. Existing solutions rely on specialized sensors: either heavy, costly airborne LiDAR, or Airborne Optical Sectioning (AOS), which uses thermal synthetic aperture photography and is tailored for person detection. We introduce a novel approach for the reconstruction of canopy-free, photorealistic ground views using only conventional RGB images. Our solution is based on the celebrated Neural Radiance Fields (NeRF), a recent 3D reconstruction method. Additionally, we include specific image capture considerations, which dictate the needed illumination to successfully expose the scene beneath the canopy. To better cope with the poorly lit understory, we employ a low light loss. Finally, we propose two complementary approaches to remove occluding canopy elements by controlling per-ray integration procedure. To validate the value of our approach, we present two possible downstream tasks. For the task of search and rescue (SAR), we demonstrate that our method enables person detection which achieves promising results compared to thermal AOS (using only RGB images). Additionally, we show the potential of our approach for forest inventory tasks like tree counting. These results position our approach as a cost-effective, high-resolution alternative to specialized sensors for SAR, trail mapping, and forest-inventory tasks.

  </details>



- **Diachronic Stereo Matching for Multi-Date Satellite Imagery**  
  Elías Masquil, Luca Savant Aira, Roger Marí, Thibaud Ehret, Pablo Musé, Gabriele Facciolo  
  _2026-01-30_ · https://arxiv.org/abs/2601.22808v1  
  <details><summary>Abstract</summary>

  Recent advances in image-based satellite 3D reconstruction have progressed along two complementary directions. On one hand, multi-date approaches using NeRF or Gaussian-splatting jointly model appearance and geometry across many acquisitions, achieving accurate reconstructions on opportunistic imagery with numerous observations. On the other hand, classical stereoscopic reconstruction pipelines deliver robust and scalable results for simultaneous or quasi-simultaneous image pairs. However, when the two images are captured months apart, strong seasonal, illumination, and shadow changes violate standard stereoscopic assumptions, causing existing pipelines to fail. This work presents the first Diachronic Stereo Matching method for satellite imagery, enabling reliable 3D reconstruction from temporally distant pairs. Two advances make this possible: (1) fine-tuning a state-of-the-art deep stereo network that leverages monocular depth priors, and (2) exposing it to a dataset specifically curated to include a diverse set of diachronic image pairs. In particular, we start from a pretrained MonSter model, trained initially on a mix of synthetic and real datasets such as SceneFlow and KITTI, and fine-tune it on a set of stereo pairs derived from the DFC2019 remote sensing challenge. This dataset contains both synchronic and diachronic pairs under diverse seasonal and illumination conditions. Experiments on multi-date WorldView-3 imagery demonstrate that our approach consistently surpasses classical pipelines and unadapted deep stereo models on both synchronic and diachronic settings. Fine-tuning on temporally diverse images, together with monocular priors, proves essential for enabling 3D reconstruction from previously incompatible acquisition dates. Left image (winter) Right image (autumn) DSM geometry Ours (1.23 m) Zero-shot (3.99 m) LiDAR GT Figure 1. Output geometry for a winter-autumn image pair from Omaha (OMA 331 test scene). Our method recovers accurate geometry despite the diachronic nature of the pair, exhibiting strong appearance changes, which cause existing zero-shot methods to fail. Missing values due to perspective shown in black. Mean altitude error in parentheses; lower is better.

  </details>


