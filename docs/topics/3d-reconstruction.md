# 3D Reconstruction

_Updated: 2026-02-23 07:19 UTC_

Total papers shown: **2**


---

- **Diff2DGS: Reliable Reconstruction of Occluded Surgical Scenes via 2D Gaussian Splatting**  
  Tianyi Song, Danail Stoyanov, Evangelos Mazomenos, Francisco Vasconcelos  
  _2026-02-20_ · https://arxiv.org/abs/2602.18314v1  
  <details><summary>Abstract</summary>

  Real-time reconstruction of deformable surgical scenes is vital for advancing robotic surgery, improving surgeon guidance, and enabling automation. Recent methods achieve dense reconstructions from da Vinci robotic surgery videos, with Gaussian Splatting (GS) offering real-time performance via graphics acceleration. However, reconstruction quality in occluded regions remains limited, and depth accuracy has not been fully assessed, as benchmarks like EndoNeRF and StereoMIS lack 3D ground truth. We propose Diff2DGS, a novel two-stage framework for reliable 3D reconstruction of occluded surgical scenes. In the first stage, a diffusion-based video module with temporal priors inpaints tissue occluded by instruments with high spatial-temporal consistency. In the second stage, we adapt 2D Gaussian Splatting (2DGS) with a Learnable Deformation Model (LDM) to capture dynamic tissue deformation and anatomical geometry. We also extend evaluation beyond prior image-quality metrics by performing quantitative depth accuracy analysis on the SCARED dataset. Diff2DGS outperforms state-of-the-art approaches in both appearance and geometry, reaching 38.02 dB PSNR on EndoNeRF and 34.40 dB on StereoMIS. Furthermore, our experiments demonstrate that optimizing for image quality alone does not necessarily translate into optimal 3D reconstruction accuracy. To address this, we further optimize the depth quality of the reconstructed 3D results, ensuring more faithful geometry in addition to high-fidelity appearance.

  </details>



- **RoEL: Robust Event-based 3D Line Reconstruction**  
  Gwangtak Bae, Jaeho Shin, Seunggu Kang, Junho Kim, Ayoung Kim, Young Min Kim  
  _2026-02-20_ · https://arxiv.org/abs/2602.18258v1  
  <details><summary>Abstract</summary>

  Event cameras in motion tend to detect object boundaries or texture edges, which produce lines of brightness changes, especially in man-made environments. While lines can constitute a robust intermediate representation that is consistently observed, the sparse nature of lines may lead to drastic deterioration with minor estimation errors. Only a few previous works, often accompanied by additional sensors, utilize lines to compensate for the severe domain discrepancies of event sensors along with unpredictable noise characteristics. We propose a method that can stably extract tracks of varying appearances of lines using a clever algorithmic process that observes multiple representations from various time slices of events, compensating for potential adversaries within the event data. We then propose geometric cost functions that can refine the 3D line maps and camera poses, eliminating projective distortions and depth ambiguities. The 3D line maps are highly compact and can be equipped with our proposed cost function, which can be adapted for any observations that can detect and extract line structures or projections of them, including 3D point cloud maps or image observations. We demonstrate that our formulation is powerful enough to exhibit a significant performance boost in event-based mapping and pose refinement across diverse datasets, and can be flexibly applied to multimodal scenarios. Our results confirm that the proposed line-based formulation is a robust and effective approach for the practical deployment of event-based perceptual modules. Project page: https://gwangtak.github.io/roel/

  </details>


