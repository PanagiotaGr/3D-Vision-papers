# 3D Reconstruction

_Updated: 2026-02-15 07:05 UTC_

Total papers shown: **2**


---

- **UPDA: Unsupervised Progressive Domain Adaptation for No-Reference Point Cloud Quality Assessment**  
  Bingxu Xie, Fang Zhou, Jincan Wu, Yonghui Liu, Weiqing Li, Zhiyong Su  
  _2026-02-12_ · https://arxiv.org/abs/2602.11969v1  
  <details><summary>Abstract</summary>

  While no-reference point cloud quality assessment (NR-PCQA) approaches have achieved significant progress over the past decade, their performance often degrades substantially when a distribution gap exists between the training (source domain) and testing (target domain) data. However, to date, limited attention has been paid to transferring NR-PCQA models across domains. To address this challenge, we propose the first unsupervised progressive domain adaptation (UPDA) framework for NR-PCQA, which introduces a two-stage coarse-to-fine alignment paradigm to address domain shifts. At the coarse-grained stage, a discrepancy-aware coarse-grained alignment method is designed to capture relative quality relationships between cross-domain samples through a novel quality-discrepancy-aware hybrid loss, circumventing the challenges of direct absolute feature alignment. At the fine-grained stage, a perception fusion fine-grained alignment approach with symmetric feature fusion is developed to identify domain-invariant features, while a conditional discriminator selectively enhances the transfer of quality-relevant features. Extensive experiments demonstrate that the proposed UPDA effectively enhances the performance of NR-PCQA methods in cross-domain scenarios, validating its practical applicability. The code is available at https://github.com/yokeno1/UPDA-main.

  </details>



- **RI-Mamba: Rotation-Invariant Mamba for Robust Text-to-Shape Retrieval**  
  Khanh Nguyen, Dasith de Silva Edirimuni, Ghulam Mubashar Hassan, Ajmal Mian  
  _2026-02-12_ · https://arxiv.org/abs/2602.11673v1  
  <details><summary>Abstract</summary>

  3D assets have rapidly expanded in quantity and diversity due to the growing popularity of virtual reality and gaming. As a result, text-to-shape retrieval has become essential in facilitating intuitive search within large repositories. However, existing methods require canonical poses and support few object categories, limiting their real-world applicability where objects can belong to diverse classes and appear in random orientations. To address this challenge, we propose RI-Mamba, the first rotation-invariant state-space model for point clouds. RI-Mamba defines global and local reference frames to disentangle pose from geometry and uses Hilbert sorting to construct token sequences with meaningful geometric structure while maintaining rotation invariance. We further introduce a novel strategy to compute orientational embeddings and reintegrate them via feature-wise linear modulation, effectively recovering spatial context and enhancing model expressiveness. Our strategy is inherently compatible with state-space models and operates in linear time. To scale up retrieval, we adopt cross-modal contrastive learning with automated triplet generation, allowing training on diverse datasets without manual annotation. Extensive experiments demonstrate RI-Mamba's superior representational capacity and robustness, achieving state-of-the-art performance on the OmniObject3D benchmark across more than 200 object categories under arbitrary orientations. Our code will be made available at https://github.com/ndkhanh360/RI-Mamba.git.

  </details>


