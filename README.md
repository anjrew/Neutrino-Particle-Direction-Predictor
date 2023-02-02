# Neutrino Particle Direction Predictor

A model based on data from the "IceCube" detector, which observes the cosmos from deep within the South Pole ice. This project is created for the Kaggle Competition [IceCube - Neutrinos in Deep Ice](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/overview)

![Example Event](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1132983%2F6891ec67d9d40315637b1b292c3a486b%2FExample_event.png?generation=1666631264548536&alt=media)

--- 

## Prerequisites

- Dependencies
    - Bash command line
    - Anaconda for python environment management

---

## Setup environment

Run the command: `source bin/activate.sh` at the project root directory.

---

## Data

The data source for the project can be found [here](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/data) on the Kaggle page.

It should be placed in the 'data/' directory directly so the file structure looks something like:
 
 - data/   
    - test/
    - train/
    - ...

---

## Observations of the dataset

- The events have a time element so the observations are dependent on each other.
- Each event has its own set of sub events that are picked up by a sensor
- There is a time series element to this
