# UNetSeptumSegmentation
U-net based Deep Network that segments the heart septum in echo images.
### Executer file: Main.py
### ML model is in UNet.py
### Datasource interface: DataSetOOP.py
It is a wrapper class encapsulating funcationalities to load the dataset and augment it. The data, which includes echo images and corresponding masks as lables, should be provided in root directory as "Data" folder.
Due to the restrictions of Johns Hopkins Hospital, I cannot make my own dataset available.
