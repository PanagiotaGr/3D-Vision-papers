# Dynamic Scenes & 4D Reconstruction

_Updated: 2026-03-16 07:46 UTC_

Total papers shown: **3**


---

- **Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos**  
  Rohith Peddi, Saurabh, Shravan Shanmugam, Likhitha Pallapothula, Yu Xiang, Parag Singla, Vibhav Gogate  
  _2026-03-13_ · https://arxiv.org/abs/2603.13185v1  
  <details><summary>Abstract</summary>

  Spatio-temporal scene graphs provide a principled representation for modeling evolving object interactions, yet existing methods remain fundamentally frame-centric: they reason only about currently visible objects, discard entities upon occlusion, and operate in 2D. To address this, we first introduce ActionGenome4D, a dataset that upgrades Action Genome videos into 4D scenes via feed-forward 3D reconstruction, world-frame oriented bounding boxes for every object involved in actions, and dense relationship annotations including for objects that are temporarily unobserved due to occlusion or camera motion. Building on this data, we formalize World Scene Graph Generation (WSGG), the task of constructing a world scene graph at each timestamp that encompasses all interacting objects in the scene, both observed and unobserved. We then propose three complementary methods, each exploring a different inductive bias for reasoning about unobserved objects: PWG (Persistent World Graph), which implements object permanence via a zero-order feature buffer; MWAE (Masked World Auto-Encoder), which reframes unobserved-object reasoning as masked completion with cross-view associative retrieval; and 4DST (4D Scene Transformer), which replaces the static buffer with differentiable per-object temporal attention enriched by 3D motion and camera-pose features. We further design and evaluate the performance of strong open-source Vision-Language Models on the WSGG task via a suite of Graph RAG-based approaches, establishing baselines for unlocalized relationship prediction. WSGG thus advances video scene understanding toward world-centric, temporally persistent, and interpretable scene reasoning.

  </details>



- **SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization**  
  Tianwei Ye, Xiaoguang Mei, Yifan Xia, Fan Fan, Jun Huang, Jiayi Ma  
  _2026-03-13_ · https://arxiv.org/abs/2603.12937v1  
  <details><summary>Abstract</summary>

  Establishing accurate point-to-point correspondences between non-rigid 3D shapes remains a critical challenge, particularly under non-isometric deformations and topological noise. Existing functional map pipelines suffer from ambiguities that geometric descriptors alone cannot resolve, and spatial inconsistencies inherent in the projection of truncated spectral bases to dense pointwise correspondences. In this paper, we introduce SGMatch, a learning-based framework for semantic-guided non-rigid shape matching. Specifically, we design a Semantic-Guided Local Cross-Attention module that integrates semantic features from vision foundation models into geometric descriptors while preserving local structural continuity. Furthermore, we introduce a regularization objective based on conditional flow matching, which supervises a time-varying velocity field to encourage spatial smoothness of the recovered correspondences. Experimental results on multiple benchmarks demonstrate that SGMatch achieves competitive performance across near-isometric settings and consistent improvements under non-isometric deformations and topological noise.

  </details>



- **Catalyst4D: High-Fidelity 3D-to-4D Scene Editing via Dynamic Propagation**  
  Shifeng Chen, Yihui Li, Jun Liao, Hongyu Yang, Di Huang  
  _2026-03-13_ · https://arxiv.org/abs/2603.12766v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D scene editing using NeRF and 3DGS enable high-quality static scene editing. In contrast, dynamic scene editing remains challenging, as methods that directly extend 2D diffusion models to 4D often produce motion artifacts, temporal flickering, and inconsistent style propagation. We introduce Catalyst4D, a framework that transfers high-quality 3D edits to dynamic 4D Gaussian scenes while maintaining spatial and temporal coherence. At its core, Anchor-based Motion Guidance (AMG) builds a set of structurally stable and spatially representative anchors from both original and edited Gaussians. These anchors serve as robust region-level references, and their correspondences are established via optimal transport to enable consistent deformation propagation without cross-region interference or motion drift. Complementarily, Color Uncertainty-guided Appearance Refinement (CUAR) preserves temporal appearance consistency by estimating per-Gaussian color uncertainty and selectively refining regions prone to occlusion-induced artifacts. Extensive experiments demonstrate that Catalyst4D achieves temporally stable, high-fidelity dynamic scene editing and outperforms existing methods in both visual quality and motion coherence.

  </details>


