# ee292fproject

https://github.com/wlmeng11/ee292fproject

# Setup
``
conda env create -f environment.yml
conda activate ee292f
pip install -r requirements.txt
``
To run the Jupyter notebooks in this environment:
``
conda install jupyter # IMPORTANT: pip install jupyter won't work correctly!
python -m ipykernel install --user --name ee292f --display-name "Python 3.7 (ee292f)"
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter notebook
``

Note that `pip install jupyter` will not work correctly because it won't install jupyter into the path of the conda env, ie. `~/.conda/envs/python38/bin/jupyter`.
