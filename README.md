# 3D-Vision-papers

An automated system for collecting, filtering, and organizing **recent research papers in 3D Computer Vision** from arXiv.

The repository uses a Python script combined with **GitHub Actions** to periodically retrieve new papers based on predefined keywords and categories.

---

##  Scope

This project focuses on research areas including (but not limited to):

- 3D Reconstruction
- Multi-view Geometry
- NeRFs
- SLAM
- Depth Estimation
- Point Clouds
- 3D Scene Understanding

---

##  How it works

1. `daily_arxiv.py` queries arXiv for recent papers
2. Filters papers based on keywords and categories defined in `config.yaml`
3. Stores results in markdown format
4. A GitHub Action (`daily.yml`) runs the process automatically on a daily basis

---

## 🗂 Repository Structure

