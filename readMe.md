


## Introduction
This repository contains the code and data for the research paper "_Distilling actionable insights through road network features to alleviate traffic congestion_". The study extracts 14 different features from OSM urban tiles of size varying from 0.25 sq. km. to 2 sq.km to predict congestion patterns in various cities. Seven cities were considered: Auckland, New York City, Cape Town, Bogota, Mexico City, Mumbai, and Istanbul. 


### Usage
To run the analysis scripts and generate the congestion features, check out to the [commit id](https://github.com/mie-lab/Urbanscales/tree/e09f7908619f9a6bab882e30002c4268ffe27f67c) at paper submit (09f7908619f9a6bab882e30002c4268ffe27f67c). A tarball of the processed data comprising the jam factor at the segment level for seven cities using [this](https://polybox.ethz.ch/index.php/s/05TB4iMrMR673Xz) link to reproduce the results from the paper. To avoid spam, the password to access the link is provided at the top of the config file `config.py`. 

### Installation
The Python version used is Python 3.8.17. The environment can be installed using the conda yml file in the home directory. However, the straightforward pip installation is more straightforward and reliable in our experience to avoid conflict between geopandas and shapely packages. 
```bash
pip install -r requirements.txt 
```
A google [colab notebook](https://github.com/mie-lab/Urbanscales/blob/main/Google_colab_quickstart_example.ipynb) is provided to demonstrate the usage of the repository. The colab version was run for one city (Auckland at 1 sq.km (50x50 tiles) and completed in approximately 90 minutes. When running on a local machine, multiple cities can be run in parallel and the number of threads per city is configurable inside `urbanscales/Pipeline.py`.

## Data
The network data is derived from OpenStreetMaps and processed using [osmnx](https://github.com/gboeing/osmnx) and the jam factor data is obtained from the [HERE api](https://www.here.com/docs/bundle/traffic-api-developer-guide-v7/page/topics/use-cases/flow-filter-jam-factor.html). 

## Citation
If you find this repository useful for your research or if you use any of the methodologies in your work, please consider citing our paper:

```bibtex
@article{kumarUrbanscales2024,
  author = {Nishant Kumar and Yatao Zhang and Nina Wiedemann and Jimi Oke and Martin Raubal},
  title = {Distilling actionable insights through road network features to alleviate traffic congestion},
  journal = {Journal/Conference Name},
  year = {2024},
  doi = {10.0000/researchsquareToBeFilled/000000},
  address = {Singapore-ETH Centre, Singapore; ETH Zurich, Switzerland; University of Massachusetts Amherst, USA}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact
For any additional questions or feedback, please contact Nishant.

