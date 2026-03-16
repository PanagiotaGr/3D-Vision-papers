# 3D Reconstruction

_Updated: 2026-03-16 07:46 UTC_

Total papers shown: **2**


---

- **Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos**  
  Rohith Peddi, Saurabh, Shravan Shanmugam, Likhitha Pallapothula, Yu Xiang, Parag Singla, Vibhav Gogate  
  _2026-03-13_ · https://arxiv.org/abs/2603.13185v1  
  <details><summary>Abstract</summary>

  Spatio-temporal scene graphs provide a principled representation for modeling evolving object interactions, yet existing methods remain fundamentally frame-centric: they reason only about currently visible objects, discard entities upon occlusion, and operate in 2D. To address this, we first introduce ActionGenome4D, a dataset that upgrades Action Genome videos into 4D scenes via feed-forward 3D reconstruction, world-frame oriented bounding boxes for every object involved in actions, and dense relationship annotations including for objects that are temporarily unobserved due to occlusion or camera motion. Building on this data, we formalize World Scene Graph Generation (WSGG), the task of constructing a world scene graph at each timestamp that encompasses all interacting objects in the scene, both observed and unobserved. We then propose three complementary methods, each exploring a different inductive bias for reasoning about unobserved objects: PWG (Persistent World Graph), which implements object permanence via a zero-order feature buffer; MWAE (Masked World Auto-Encoder), which reframes unobserved-object reasoning as masked completion with cross-view associative retrieval; and 4DST (4D Scene Transformer), which replaces the static buffer with differentiable per-object temporal attention enriched by 3D motion and camera-pose features. We further design and evaluate the performance of strong open-source Vision-Language Models on the WSGG task via a suite of Graph RAG-based approaches, establishing baselines for unlocalized relationship prediction. WSGG thus advances video scene understanding toward world-centric, temporally persistent, and interpretable scene reasoning.

  </details>



- **Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass**  
  Sangmin Kim, Minhyuk Hwang, Geonho Cha, Dongyoon Wee, Jaesik Park  
  _2026-03-13_ · https://arxiv.org/abs/2603.12789v1  
  <details><summary>Abstract</summary>

  Recent advances in 3D foundation models have led to growing interest in reconstructing humans and their surrounding environments. However, most existing approaches focus on monocular inputs, and extending them to multi-view settings requires additional overhead modules or preprocessed data. To this end, we present CHROMM, a unified framework that jointly estimates cameras, scene point clouds, and human meshes from multi-person multi-view videos without relying on external modules or preprocessing. We integrate strong geometric and human priors from Pi3X and Multi-HMR into a single trainable neural network architecture, and introduce a scale adjustment module to solve the scale discrepancy between humans and the scene. We also introduce a multi-view fusion strategy to aggregate per-view estimates into a single representation at test-time. Finally, we propose a geometry-based multi-person association method, which is more robust than appearance-based approaches. Experiments on EMDB, RICH, EgoHumans, and EgoExo4D show that CHROMM achieves competitive performance in global human motion and multi-view pose estimation while running over 8x faster than prior optimization-based multi-view approaches. Project page: https://nstar1125.github.io/chromm.

  </details>


