

# Urban Scales:  Road network features in square urban tiles predict congestion levels 

## Introduction
This repository contains the code and data for the research paper "Investigating the Link Between Road Network and Congestion for Highly Congested Cities". The study extracts 14 different features from OSM urban tiles of size varying from 0.25 sq. km. to 2 sq.km to predict congestion patterns in various cities. Seven cities were considered: Auckland, New York City, Cape Town, Bogota, Mexico City, Mumbai, and Istanbul. 

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
To run the analysis scripts and generate the congestion features, check out to the [commit id]([url](https://github.com/mie-lab/Urbanscales/tree/ed2b6b2e37d5aa3e39d74ac7e4ca859db4ec4ede)) at paper submit


## Data
The network data is derived from OpenStreetMaps and processed using `[osmnx](https://github.com/gboeing/osmnx)` and the jam factor data is obtained from the [HERE api](https://www.here.com/docs/bundle/traffic-api-developer-guide-v7/page/topics/use-cases/flow-filter-jam-factor.html). 

## Citation
If you find this repository useful for your research or if you use any of the methodologies in your work, please consider citing our paper:

```bibtex
@article{kumarUrbanscales2024,
  author = {Nishant Kumar and Yatao Zhang and Nina Wiedemann and Jimi Oke and Martin Raubal},
  title = {Investigating the Link Between Road Network and Congestion for Highly Congested Cities},
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

