# Multi-Head Neural Operator (MHNO) for Modelling Interfacial Dynamics

This repository contains the code and data for the paper:

**Multi-Head Neural Operator for Modelling Interfacial Dynamics**
*Mohammad Sadegh Eshaghi, Navid Valizadeh, Cosmin Anitescu, Yizheng Wang, Xiaoying Zhuang, Timon Rabczuk*
📄 [IJMS Publication](https://www.sciencedirect.com/science/article/pii/S0020740326002195) | [arXiv:2507.17763](https://doi.org/10.48550/arXiv.2507.17763)

---

## 📢 Availability

📦 **Data and Models**: [Download here](https://seafile.cloud.uni-hannover.de/d/9f6f5fafc17548c1837d/)
📄 **Published in**: [International Journal of Mechanical Sciences](https://www.sciencedirect.com/science/article/pii/S0020740326002195)

---

## 🧠 Overview

Interfacial dynamics play a central role in many physical systems, such as phase transitions, microstructure evolution, and thin-film growth. These problems are modeled by stiff, nonlinear, time-dependent PDEs, which are challenging to solve, especially over long time horizons.

We propose the **Multi-Head Neural Operator (MHNO)**, a neural operator architecture designed to capture **PDEs with long temporal dynamics**. MHNO combines time-step-specific projection operators and explicit temporal message passing, enabling the accurate and efficient prediction of full-time evolutions in a single forward pass.

We validate MHNO on a wide range of phase field models, including:
- Antiphase boundary motion
- Spinodal decomposition
- Pattern formation
- Atomic-scale modeling
- Molecular Beam Epitaxy (MBE)

---

## 📁 Repository Structure
- **configs/** – Configuration files for different problems (AC2D, AC3D, CH2D, CH2DNL, CH3D, SH2D, PFC2D, MBE2D) and neural operator architectures (FNO2d, FNO3d, TNO2d, TNO3d)
- **MatlabCode/** – Scripts for data generation
- **comparision.py** – Benchmarking MHNO against other methods
- **hyperparameter_sweep.py** – Hyperparameter sweeping utilities
- **hyperparameter_tuning.py** – Hyperparameter tuning experiments
- **hyperparameter_result.py** – Analysis and reporting of hyperparameter results
- **main.py** – Entry point for training and evaluation
- **networks.py** – Neural operator architecture definitions (MHNO, FNO, etc.)
- **training.py** – Training loop and checkpointing
- **post_processing.py** – Metrics calculation and visualization
- **utilities.py** – Helper functions and utilities 


---

## 📫 Contact

For questions or collaboration requests, feel free to reach out to:  
- Mohammad Sadegh Eshaghi: eshaghi.khanghah@iop.uni-hannover.de
- Navid Valizadeh: valizadeh@iop.uni-hannover.de
- Xiaoying Zhuang: zhuang@iop.uni-hannover.de

---

## 📘 Citation

If you use this work in your research, please cite:

```bibtex
@article{eshaghi2025mhno,
  title={Multi-Head Neural Operator for Modelling Interfacial Dynamics},
  author={Eshaghi, Mohammad Sadegh and Valizadeh, Navid and Anitescu, Cosmin and Wang, Yizheng and Zhuang, Xiaoying and Rabczuk, Timon},
  journal={International Journal of Mechanical Sciences},
  year={2025},
  doi={10.1016/j.ijmecsci.2025.109972}
}
