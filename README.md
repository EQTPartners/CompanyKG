# CompanyKG: Dataset access and evaluation

## Pre-Requisites

* Python 3.8

There are also optional dependencies, if you want to be able to convert the KG
to one of the data structures used by these packages:

* [DGL](https://pypi.org/project/dgl/): `dgl`
* [iGraph](https://pypi.org/project/python-igraph/): `python-igraph`
* [PyTorch Geometric (PyG)](https://pypi.org/project/torch-geometric/): `torch-geometric`


## Setup

The `companykg` Python package provides a data structure to load CompanyKG into memory,
convert between different graph representations and run evaluation of trained embeddings
or other company-ranking predictors on three evaluation tasks.

To install the `comapnykg` package and its Python dependencies, activate a virtual 
environment (such as [Virtualenv](https://github.com/pyenv/pyenv-virtualenv) or Conda) and run:

```bash
pip install -e .
```

The first time you instantiate the CompanyKG class, if the dataset is not already available
(in the default subdirectory or another location you specify), the latest version will be automatically
downloaded from Zenodo:

```python
from companykg import CompanyKG

ckg = CompanyKG()
```


## Basic usage

By default, the CompanyKG dataset will be loaded from (and potentially downloaded to) 
a `data` subdirectory of the CWD. If you have already downloaded the dataset and want to
load it from its current location, specify the path:

```python
ckg = CompanyKG(data_root_folder="/path/to/stored/companykg/directory")
```

The graph can be loaded with different vector representations (embeddings) of
company description data associated with the nodes: `msbert` (mSBERT), `simcse`(SimCSE), 
`ada2` (ADA2) or `pause` (PAUSE).

```python
ckg = CompanyKG(nodes_feature_type="pause")
```

If you want to experiment with different embedding types, you can also load embeddings
of a different type into an already-loaded graph:

```python
ckg.change_feature_type("simcse")
```

By default, edge weights are not loaded into the graph. To change this use:
```python
ckg = CompanyKG(load_edges_weights=True)
```


A tutorial showing further ways to use CompanyKG is [here](./tutorials/tutorial.ipynb).


## Training benchmark models

Implementations of various benchmark graph-based learning models are provided in this repository.

To use them, install the `ckg_benchmarks` Python package, along with its dependencies, from the
`benchmarks` subdirectory. First install `companykg` as above and then:

```bash
cd benchmarks
pip install -e .
```

Further instructions for using the benchmarks package for model training and provided in
the [benchmarks README file](./benchmarks/README.md).


## Cite This Work
```bibtex
@article{companykg_2023_8010239,
    author = {Lele Cao and
                Vilhelm von Ehrenheim and
                Mark Granroth-Wilding and
                Richard Anselmo Stahl and
                Drew McCornack and
                Armin Catovic and
                Dhiana Deva Cavacanti Rocha},
    title = {{CompanyKG Dataset: A Large-Scale Heterogeneous Graph for Company Similarity Quantification}},
    month = June,
    year = 2023,
    journal = {Zenodo},
    version = {1.1},
    volume = {10.5281/zenodo.8010239},
    url = {https://doi.org/10.5281/zenodo.8010239}
}
```
