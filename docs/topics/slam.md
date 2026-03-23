# SLAM & Localization

_Updated: 2026-03-23 07:38 UTC_

Total papers shown: **1**


---

- **Investigating a Policy-Based Formulation for Endoscopic Camera Pose Recovery**  
  Jan Emily Mangulabnan, Akshat Chauhan, Laura Fleig, Lalithkumar Seenivasan, Roger D. Soberanis-Mukul, S. Swaroop Vedula, Russell H. Taylor, Masaru Ishii, Gregory D. Hager, Mathias Unberath  
  _2026-03-20_ · https://arxiv.org/abs/2603.20045v1  
  <details><summary>Abstract</summary>

  In endoscopic surgery, surgeons continuously locate the endoscopic view relative to the anatomy by interpreting the evolving visual appearance of the intraoperative scene in the context of their prior knowledge. Vision-based navigation systems seek to replicate this capability by recovering camera pose directly from endoscopic video, but most approaches do not embody the same principles of reasoning about new frames that makes surgeons successful. Instead, they remain grounded in feature matching and geometric optimization over keyframes, an approach that has been shown to degrade under the challenging conditions of endoscopic imaging like low texture and rapid illumination changes. Here, we pursue an alternative approach and investigate a policy-based formulation of endoscopic camera pose recovery that seeks to imitate experts in estimating trajectories conditioned on the previous camera state. Our approach directly predicts short-horizon relative motions without maintaining an explicit geometric representation at inference time. It thus addresses, by design, some of the notorious challenges of geometry-based approaches, such as brittle correspondence matching, instability in texture-sparse regions, and limited pose coverage due to reconstruction failure. We evaluate the proposed formulation on cadaveric sinus endoscopy. Under oracle state conditioning, we compare short-horizon motion prediction quality to geometric baselines achieving lowest mean translation error and competitive rotational accuracy. We analyze robustness by grouping prediction windows according to texture richness and illumination change indicating reduced sensitivity to low-texture conditions. These findings suggest that a learned motion policy offers a viable alternative formulation for endoscopic camera pose recovery.

  </details>


