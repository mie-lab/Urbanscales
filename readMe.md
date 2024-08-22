

```markdown
# Urban Scales: Analyzing Congestion Using OSM Data

## Introduction
This repository contains the code and data for the research paper "Analyzing Urban Congestion Through OpenStreetMap Data". The study extracts 14 different features from OSM urban tiles of size 1 sq km to predict congestion patterns in various cities including Auckland, New York City, London, Cape Town, Bogota, Mexico City, Mumbai, and Istanbul. These features include graph-based metrics such as number of nodes and edges, traffic dynamics like intersection counts and traffic lights, and other relevant urban characteristics.

## Features
The features analyzed in this study include:
- Number of nodes
- Number of edges
- Total crossings
- Traffic lights
- Free turns
- Average lanes
- Local centrality
- And more...

## Getting Started

### Prerequisites
- Python 3.x
- Geopandas
- Osmnx
- Other Python libraries as specified in `requirements.txt`

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/mie-lab/Urbanscales.git
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Usage
To run the analysis scripts and generate the congestion features, checkout to the commit used for the paper:
```bash
git checkout <commit-id>  # Replace <commit-id> with the actual commit ID
python analyze_congestion.py
```

## Data
The data used in this study is derived from OpenStreetMap and processed using the Osmnx library. Note that due to the size of the data, only sample datasets are included in this repository.

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and use a new branch for your contributions. Pull requests are warmly welcome.

## Citation
If you find this repository useful for your research or if you use any of the methodologies in your work, please consider citing our paper:

### Recommended Citation
Nishant Kumar, Yatao Zhang, Nina Wiedemann, Jimi Oke, and Martin Raubal. "Investigating the Link Between Road Network and Congestion for Highly Congested Cities." *Under Review*, 2024. DOI: 10.0000/researchsquare.XXX/000000

```bibtex
@article{kumarUrbanscales2024,
  author = {Nishant Kumar and Yatao Zhang and Nina Wiedemann and Jimi Oke and Martin Raubal},
  title = {Investigating the Link Between Road Network and Congestion for Highly Congested Cities},
  journal = {Journal/Conference Name},
  year = {2024},
  doi = {10.0000/researchsquare.fake/000000},
  address = {Singapore-ETH Centre, Singapore; ETH Zurich, Switzerland; University of Massachusetts Amherst, USA}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact
For any additional questions or feedback, please contact [your contact information or that of your lab].

```

commit ID relevant for reproducing the results described in the paper: ec9e079b53142c4ac3ce12d48e5509b8349fcac1 