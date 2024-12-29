

# Urban Scales:  Road network features in square urban tiles predict congestion levels 

## Introduction
This repository contains the code and data for the research paper "_Investigating the Link Between Road Network and Congestion for Highly Congested Cities_". The study extracts 14 different features from OSM urban tiles of size varying from 0.25 sq. km. to 2 sq.km to predict congestion patterns in various cities. Seven cities were considered: Auckland, New York City, Cape Town, Bogota, Mexico City, Mumbai, and Istanbul. 

### Prerequisites
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/mie-lab/Urbanscales.git
```


### Usage
To run the analysis scripts and generate the congestion features, check out to the [commit id]([url](https://github.com/mie-lab/Urbanscales/tree/e09f7908619f9a6bab882e30002c4268ffe27f67c)) at paper submit (09f7908619f9a6bab882e30002c4268ffe27f67c). We do not have the necessary permissions to share the raw traffic data collected. Please feel free to contact the corresponding author for a tarball of the collected data to reproduce the results from this paper. 


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

