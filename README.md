# PhysioNet dataset exploratory analysis

The electrocardiogram (ECG) is a non-invasive representation of the electrical activity of the heart from electrodes placed on the surface of the torso. The standard 12-lead ECG has been widely used to diagnose a variety of cardiac abnormalities such as cardiac arrhythmias, and predicts cardiovascular morbidity and mortality. The early and correct diagnosis of cardiac abnormalities can increase the chances of successful treatments. However, manual interpretation of the electrocardiogram is time-consuming, and requires skilled personnel with a high degree of training.

Automatic detection and classification of cardiac abnormalities can assist physicians in the diagnosis of the growing number of ECGs recorded. Over the last decade, there have been increasing numbers of attempts to stimulate 12-lead ECG classification. Many of these algorithms seem to have the potential for accurate identification of cardiac abnormalities. However, most of these methods have only been tested or developed in single, small, or relatively homogeneous datasets. 


## How to set it up

- Unzip the file to physionet-analysis
- Build the Docker image

```
cd physionet-analysis && docker build -t physionet-analysis .
```

- Run the image:
```
docker run --network=host -v ${PWD}:/home/jovyan/work -it physionet-analysis jupyter-notebook
```

- When prompted in the console, use the provided http link to have access to the jupyter notebook environment
- Navigate through the browser to the work/physionet-analysis.ipynb notebook
- Run all the cells of the notebook using kernel/Restart & Run All