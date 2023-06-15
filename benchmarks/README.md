# CompanyKG benchmarks

The code in this directory, released as a PyPi package `companykg-benchmarks`, implements all model
training are required code for experimentation that constitutes the initial benchmarking of the 
CompanyKG knowledge graph.


## Setup

You will need Python>=3.8.

All other dependencies needed to run the full model training and evaluation are
covered by `setup.py`, so just follow instructions to install the `ckg_benchmarks`
package in your virtual environment.

Note that we also depend on the CompanyKG package `companykg`, the main package
provided at the top level in this same repository.
See [the main README file](../README.md) for instructions.


Activate a Python virtual environment (such as 
[Virtualenv](https://github.com/pyenv/pyenv-virtualenv) or Conda).
`cd` to the directory containing this `README` (`/benchmarks/` in the repository).
Install the `companykg-benchmarks` package and its dependencies using:
```bash
pip install -e .
```

Note that this package has a lot more dependencies than `companykg`, since it
needs all the machine learning libraries used to train models.


## Benchmark models

`companykg-benchmarks` provides training routines for the following graph learning-based
graph node encoding models:
* GRACE (GCL)
* MVGRL (GCL)
* GraphSAGE
* GraphMAE
* eGraphMAE

Each model, once trained on the CompanyKG graph, can be used to produce a new embedding
to represent each node of the graph. These can be used to measure the similarity of companies
and thus applied to the three evaluation tasks:
* Similarity Prediction (SP)
* Similarity Ranking (SR)
* Competitor Retrieval (CR)

Each model is trained on the CompanyKG graph, loaded using the `companykg` package. Once
this is downloaded, the training routines can be pointed to the local dataset using the
`--data-root-folder` option.

The models all have the same training interface and training can be run from the command line
or programmatically, e.g. in a Jupyter notebook. We provide examples of both below.


### Command-line training

You can run training from the command line in the following form (from within a 
virtual environment with the `companykg-benchmarks` package installed):
```bash
python -m ckg_benchmarks.<METHOD-NAME>.train ...
```
where `<MODEL-TYPE>` is `gcl` (GRACE and MVGRL), `graphsage`, `graphmae` or `egraphmae`.

The remaining options control data location, training options and model hyperparameters.
See the `--help` for each command for more details.

Examples of training commands for each model type are provided below and in the `tutorials`
directory.

### Python training

Each model's `train` module contains a `train_model` function that can be called to train
and return a trainer instance, which includes the trained model. 
The function's keyword arguments match the options of the command-line
interface.

```python
from ckg_benchmarks.<METHOD-NAME>.train import train_model

trainer = train_model(
    data_root_folder="path/to/data",
    n_layer=3,
    ...
)
# Final trained model is available as:
trainer.model
```

See below for examples of training with specific methods.

The notebook [gcl_train](../tutorials/gcl_train.ipynb) provides a full example of
training and evaluating with one method.



### Examples

For each of the training methods, we show below an example of 
how to run training from the command line and an equivalent 
example in Python code.

Note that all examples use extremely limited hyperparameters, so the
resulting model will not be good, but can be trained with small memory
in a short time. In practice, you would want to adjust the hyperparameters
to something more like the published model selection results.


#### GCL (GRACE and MVGRL)

Train with GRACE from the command line:
```bash
python -m ckg_benchmarks.gcl.train \
    --device -1 \
    --method grace \
    --n-layer 1 \
    --embedding-dim 8 \
    --epochs 1 \
    --sampler-edges 2 \
    --batch-size 128
```

`device=-1` forces use of CPU. Select a different device number
to use a GPU.

To train MVGRL, simply change the `method` parameter:
```bash
python -m ckg_benchmarks.gcl.train \
    --device -1 \
    --method mvgrl \
    --n-layer 1 \
    --embedding-dim 8 \
    --epochs 1 \
    --sampler-edges 2 \
    --batch-size 128
```

To do the same thing from Python code:
```python
from ckg_benchmarks.gcl.train import train_model
trainer = train_model(
    # Use CPU
    device=-1,
    # Train with GRACE; you can also use 'mvgrl' here
    method="grace",
    # Typically we use 2 or 3
    n_layer=1,
    # Minimum value we usually consider is 8
    embedding_dim=8,
    # For our experiments we trained for 100 epochs, here just 1 for testing
    epochs=1,
    # We usually sample 5 or 10 edges for training
    sampler_edges=2,
    # For GPU you'll want to set your batch size bigger if you can, as it makes it faster
    batch_size=128,
)
```


#### GraphSAGE

Train from command line:
```bash
python -m ckg_benchmarks.graphsage.train \
    --device -1 \
    --n-layer 2 \
    --embedding_dim 8 \
    --epochs 1 \
    --train-batch-size 256 \
    --inference-batch-size 256 \
    --n-sample-neighbor 2 \
```

The same thing from Python code:
```python
from ckg_benchmarks.graphsage.train import train_model
trainer = train_model(
    device=-1,
    n_layer=2,
    embedding_dim=8,
    epochs=1,
    train_batch_size=256,
    inference_batch_size=256,
    n_sample_neighbor=2,
)
```



#### GraphMAE

Train from command line:
```bash
python -m ckg_benchmarks.graphmae.train \
    --device -1 \
    --n-layer 2 \
    --embedding_dim 8 \
    --epochs 1 \
    --batch-size 16 \
    --disable-metis
```

The same thing from Python code:
```python
from ckg_benchmarks.graphmae.train import train_model
trainer = train_model(
    device=-1,
    n_layer=2,
    embedding_dim=8,
    epochs=1,
    batch_size=16,
    disable_metis=True,
)
```

`disable_metis` is useful for training with small memory, but you may want to
drop this option for more efficient training.



#### eGraphMAE

Train from command line:
```bash
python -m ckg_benchmarks.egraphmae.train \
    --device -1 \
    --n-layer 2 \
    --embedding-dim 8 \
    --epochs 1 \
    --batch-size 16 \
    --disable-metis
```

The same thing from Python code:
```python
from ckg_benchmarks.egraphmae.train import train_model
trainer = train_model(
    device=-1,
    n_layer=2,
    embedding_dim=8,
    epochs=1,
    batch_size=16,
    disable_metis=True,
)
```



### Evaluating trained models

Evaluation on all three tasks is implemented in the `companykg` package.
As part of the training routine, the full graph's node embeddings are projected
into the GNN's embedding space and the resulting embeddings are output to a file.
It is therefore easy to evaluate these embeddings using CompanyKG.

When training is run from the command line, the path to which the embeddings
are output is printed at the end of training. You can use the `companykg.eval`
tool to run all evaluation tasks on the embeddings:
```bash
python -m companykg.eval <PATH-TO-EMBEDDINGS>
```

When training is run from Python, the returned trainer instance provides a method
`evaluate()` that runs the CompanyKG evaluation method on the projected embeddings.
The evaluation results will be output to stdout (unless you specify `silent=True)
and the returned dictionary contains the results for the tasks.
```python
from ckg_benchmarks.gcl.train import train_model

trainer = train_model(...)
results_dict = trainer.evaluate(embed=trainer.embeddings)
```
