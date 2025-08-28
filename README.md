# PyCLM
A Modular Closed-Loop Microscopy Software in Python

## Overview

This approach to smart microscopy is oriented around three major goals:

1. **Efficiency**

- Ideally, the number of possible simultaneous experiments should be limited only by the physical constraints of the system.
- Microscope control, data export, image segmentation, and pattern generation are split into separate processes that run in parallel.

2. **(Re)usability**