

# Urban Scales:  Road network features in square urban tiles predict congestion levels 

## Introduction
This repository contains the code and data for the research paper "_Investigating the Link Between Road Network and Congestion for Highly Congested Cities_". The study extracts 14 different features from OSM urban tiles of size varying from 0.25 sq. km. to 2 sq.km to predict congestion patterns in various cities. Seven cities were considered: Auckland, New York City, Cape Town, Bogota, Mexico City, Mumbai, and Istanbul. 


### Usage
To run the analysis scripts and generate the congestion features, check out to the [commit id](https://github.com/mie-lab/Urbanscales/tree/e09f7908619f9a6bab882e30002c4268ffe27f67c) at paper submit (09f7908619f9a6bab882e30002c4268ffe27f67c). We lack permission to share the raw traffic data collected, but a tarball of the processed data comprising the jam factor at the segment level for seven cities using [this](https://polybox.ethz.ch/index.php/s/05TB4iMrMR673Xz) link to reproduce the results from the paper. To avoid spams, the password to access the link is provided at the top of the config file `config.py`.

### Installation
The [wiki](https://github.com/mie-lab/Urbanscales/blob/main/Installing_and_running.md) page comprises all the necessary information to install the environment and run the pipeline. 

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

