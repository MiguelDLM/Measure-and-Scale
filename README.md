# Measure and Scale (Blender Add-on)

This is a fork of the original "Measure and Scale" add-on by **Belaid ZIANE**.

## Improvements in this Fork

### Scene Unit Scale Support
The core improvement in this version is the native support for Blender's **Unit Scale** settings (`Scene Properties > Units`).

In the original version, the add-on assumed a 1:1 ratio between Blender internal units and meters. This caused significant discrepancies when working in projects with custom scales (e.g., set to `0.001` for millimeters).

**Fixed behavior:**
- **Accurate Measurement:** The measurement tool now correctly factors in the `Unit Scale`, showing the real-world distance based on your scene settings.
- **Reliable Scaling:** When entering a target dimension, the add-on accurately calculates the required internal scaling factor relative to the scene units.

## Installation
1. Download the repository as a ZIP file.
2. In Blender, go to `Edit > Preferences > Add-ons > Install...`.
3. Select the downloaded ZIP file and enable "Measure and Scale".

## Usage
Find the tool in the **3D Viewport > Sidebar (N-panel) > Item > Measure and Scale**.
