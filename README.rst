.. image:: https://upload.wikimedia.org/wikipedia/commons/f/f4/Logo_EPFL.svg
   :alt: EPFL logo
   :width: 75px
   :align: right

==============
SkipTrain
==============

This repository contains the code for running the experiments in the paper "Energy-Aware Decentralized Learning with Intermittent Model Training" authored by Martijn de Vos,  Akash Dhasade, Paolo Dini, Elia Guerra, Anne-Marie Kermarrec, Marco Miozzo, Rafael Pires, and Rishi Sharma.

-------------------------
Setting up
-------------------------

* Clone this repository
* Download its submodules ::

    git submodule update --init --recursive

* Create a new Conda environment ::

    conda create --name skiptrain python==3.8.16
    conda activate skiptrain

* Install requirements ::

    pip install -r requirements.txt

* Install SkipTrain ::
  
    pip3 install --editable .[dev]

* Download CIFAR-10 using ``download_dataset.py``. ::

    python download_dataset.py

* (Optional) Download the FEMNIST from LEAF <https://github.com/TalwalkarLab/leaf> and place them in ``eval/data/``.
 
----------------
Running the code
----------------

* The ``tutorial/`` folder contains a working example on a 3-regular graph.

    * SkipTrain: ``tutorial/run_skiptrain.sh``
    * SkipTrain Constrained: ``tutorial/run_skiptrain_constrained.sh``
    * DPSGD: ``tutorial/run_dpsgd.sh``
    * Greedy: ``tutorial/run_greedy.sh``

* The topologies adopted for the simulation are available in the ``tutorial/`` folder.

------
Citing
------

Cite us as ::
..
    @inproceedings{skiptrain,
   author = {},
   title = {Energy-Aware Decentralized Learning with Intermittent Model Training},
   year = {2023},
   isbn = {},
   publisher = {},
   address = {},
   url = {},
   doi = {},
   booktitle = {},
   pages = {},
   numpages = {},
   keywords = {decentralized learning, machine learning, energy effiency, peer-to-peer},
   location = {},
   series = {}
   }
..