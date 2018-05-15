# Fighting Gerrymandering with PyMC3

This is the code and data needed to recreate the analysis from [Fighting Gerrymandering with PyMC3](https://us.pycon.org/2018/schedule/presentation/110/).  The end result is precinct level estimates of voting by race in the North Carolina 2016 congressional election, following the hierarchical model from [King, Rosen, and Tanner, 1999](http://journals.sagepub.com/doi/10.1177/0049124199028001004), [Rosen et al., 2001](https://gking.harvard.edu/files/abs/rosen-abs.shtml), and [R library](https://gking.harvard.edu/eir). See the slides for further citations.

## Video

A video of the talk is [available on youtube](https://youtu.be/G9I5ZnkWR0A).

## Slides

Slides are [available on speakerdeck](https://speakerdeck.com/karink520/fighting-gerrymandering-with-pymc3).

## Running the code

The python files have the following requirements:

- `geopandas`
- `pandas`
- `requests`
- `pymc3`
- `matplotlib`
- `numpy`
- `scipy`
- `geojson` (if using precomputed data)

You will also need to install `jupyter` or `jupyterlab` to run the notebooks.  All code was run with Python 3.6.4.
## Organization

### Notebooks
You should start here, and maybe also stop here, unless you want to really mess with things.
- `toy_data.ipynb` This notebook builds a toy election data set, and runs the inference on it.
- `north_carolina.ipynb` This notebook uses the 2016 North Carolina congressional election data, and runs ecological inference on it.  Note that running the notebook all the way through will take around 2 hours, even with the precomputed data sets. There is some light cacheing so running it the second time takes only around 10 minutes.

### Data Directories

- `precomputed_data/` contains the cleaned and organized data files.
- `data/` contains the manual mappings of districts from the census to districts from OpenElections. In case you want to build the data yourself, you should put all the required files in there.  See `load_data.py` for the URLs where I gathered the data, and for the file names that the script expects.

### Python Files
- `load_data.py` This is the Python code that was originally meant to collect the data and organize it. There was a little too much ad-hoc analysis to completely automate it, but the docstrings let you know what URL I found the data at.
- `inference.py` This file does all the statistical inference with PyMC3. It expects dataframes from the `load_data.py` file.  The outputs are traces.
- `plots.py` Supports plotting the analyses that were done.  The API is a little awkward, because it is optimized for making plots for the PyCon talk.  For example, they involve multiple `step`s to walk through what the plots mean.
