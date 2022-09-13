conda create --name MLE-Hands-on-III python=3.9 -y
conda activate tf29_3
conda install -c conda-forge jupyter pandas matplotlib scikit-learn seaborn plotly scipy opencv ipywidgets -y
conda install -c apple tensorflow-deps -y
python -m pip install tensorflow-macos 
python -m pip install tensorflow-metal 
python -m pip install tensorflow-datasets
