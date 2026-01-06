# Installation

_Documentation is currently incomplete. Interested users are recommended to check back in over the next week as the documentation is continually updated._

Installation can be technically demanding, and ease of installation will depend on your microscope setup. PyCLM is tested on a Nikon Ti2-Eclipse confocal microscope with a Mightex Polygon 1000 DMD, running on a Windows OS.

## 1. Install and configure MicroManager
PyCLM does not communicate with your microscope hardware directly. Instead, it operates through pymmcore-plus, which is built on top of [MicroManager](https://micro-manager.org/), a well-documented microscope control platform with support for a wide range of devices. MicroManager software will be used to determine imaging configurations, and define config groups. It is worth familiarizing yourself with MicroManager before using PyCLM.

## 2. Create a Python virtual environment
Clone this GitHub repository to a directory where you can run code:

```bash
git clone https://github.com/Harrison-Oatman/PyCLM.git
```

## 3. Install and configure pymmcore-plus
(Instructions on installing pymmcore-plus will go here. For now, please refer to pymmcore-plus documentation).

## 4. Install CellposeSAM
(Instructions for CellposeSAM if applicable).

## 5. Calibrate your DMD
For PyCLM to deliver spatially modulated stimuli, the relationship between DMD pixel coordinates and camera pixel coordinates needs to be determined. MicroManager provides a tool for calibrating the DMD using the "[Projector](https://micro-manager.org/Projector_Plugin)" plugin. This plugin will generate a set of DMD patterns to approximate the affine transform to/from the DMD.

## 6. Update pyclm_config.toml
``pyclm_config.toml`` is a configuration file used during every run of PyCLM to get important information about the DMD, including the DMD shape, and the affine transform.
