# Land Use Change: RGB Encoding of Multidimensional Geospatial Drivers, Image Synthesis and CNN Prediction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the dataset, code, and trained models supporting the research article:

Qiao, Z., Wang, M., Xu, W., Zhang, X., Liu, W. (2025). Land Use Change: RGB Encoding of Multidimensional Geospatial Drivers, Image Synthesis and CNN Prediction.

## Overview

This research proposes a novel deep learning prediction framework based on "RGB encoding-CNN recognition" that transforms heterogeneous geospatial drivers into visualized images for land use change (LUC) prediction. The methodology achieves a paradigm shift from traditional "data-driven" to "image-driven" approaches by:

- Converting multidimensional geospatial drivers into RGB-encoded feature images
- Preserving spatial topology, neighborhood relationships, and geometric characteristics
- Implicitly encoding driver interactions through color mixing principles
- Applying modified CNNs with spatial attention, channel attention, and multi-scale fusion modules


## Dataset Description

### Study Area
The dataset covers ten suburban districts of Beijing Municipality, China, including Mentougou, Fangshan, Tongzhou, Shunyi, Changping, Daxing, Huairou, Pinggu, Miyun, and Yanqing Districts (total area: ~12,800 km²).

### Spatial Units
- **Grid resolution**: 500m × 500m
- **Total grid cells**: 61,513
- **Effective analysis units**: 12,125 (where residential land area proportion >30%)

### Temporal Coverage
- **Historical period**: 2010-2020
- **Prediction period**: 2020-2035

### Driver Variables (9 candidates)
1. **Endogenous foundation**: Village size (V) - small/medium/large
2. **Locational conditions**: 
   - Distance to city center (C)
   - Distance to industrial park/new town (I)
   - Distance to town government (T)
   - Distance to reservoir (Re)
3. **Transportation accessibility**:
   - Distance to expressway (E)
   - Distance to national/provincial highway (N)
   - Distance to county road (Cr)
   - Distance to railway (Ra)

### Driver Configuration Schemes
- **Total combinations**: 78 three-variable configurations (C(9,3))
- **RGB-encoded images per scheme**: 12,125 (one per grid unit)
- **Total image samples**: 945,750

### Land Use Change Categories
- **Increase**: Rural residential land expansion
- **Unchanged**: Stable rural residential land
- **Decrease**: Rural residential land reduction

### Data Format
- **Spatial data**: Shapefiles (EPSG:4326)
- **Raster data**: GeoTIFF (30m resolution)
- **RGB images**: PNG (variable size, georeferenced)
- **Labels**: CSV with grid ID and change category

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU acceleration)


## License

This dataset and code are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- **Share**: Copy and redistribute the material in any medium or format
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made

## Contact

**Corresponding Author**: Maojun Wang  
📧 Email: zehaoqiao@188.com  
🏛️ Institution: College of Resources Environment and Tourism, Capital Normal University  
📍 Address: Haidian District, Beijing 100048, China

## Acknowledgments

This work was financially supported by the National Natural Science Foundation of China (Grant No. 42571244).

Full image data for the examples is available at the following link: https://pan.baidu.com/s/1wLrrB50FEhgSL0DITV1qcQ, code: yrks. For complete data, please contact us.
