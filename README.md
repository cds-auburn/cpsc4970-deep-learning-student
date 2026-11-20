# CPSC4970: Deep Learning Sample code

This repository contains code displayed during lecture demos.

Each module folder holds that module's notebooks. From Module 2 onward a module
may also contain:

| Folder | Contents |
|---|---|
| `demos/` | Demonstration notebooks, standalone scripts and small helper modules |
| `models/` | Trained weights, metadata and a README for each model |
| `data/` | Small data files used by the demonstrations, with their sources |
| `tools/` | Scripts that train the course's small models or download larger ones |

Paths in the lecture slides, such as `data/iris/test.csv`, are relative to the
module folder. Run the scripts from there; each inference notebook's Setup
cell locates that folder itself.
