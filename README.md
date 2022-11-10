# PREDICTION OF FAULTY WATER PUMPS



## Project Background 

This project was conceptualized by **`Taarifa`** as a competitive project among Data Scientists hosted on **`Drivendata platform`**. The project competition can be assessed [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/)

**`Taarifa`** is a platform that offers business management solutions to clients who find it easy in online data storage, information analysis, and tons of new features. They ensure high data security with ease of access to the administration on all devices. In other words, **`Taarifa`** is an open source platform for the crowd sourced reporting and triaging of infrastructure related issues.

For this project, the data is sourced from the **`Taarifa waterpoints dashboard`**, which aggregates data from the **`Tanzania Ministry of Water.`**

## Problem Statement

This project seeks to build a model that would predicts if a water-pump is functional, or needs some repairs, or totally non-functional. Prediction of one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

The project is a `multi-classification` problem 

The most performing model of all the models will be built will b deployed to production. 

## Project Dataset 

Dataset for this project have two files.

    - `water-train.csv`: contains all the features
    - `water-label.csv`: contains the label

### Dataset Feature Overview 

The goal is to predict the operating condition of a waterpoint for each record in the dataset. The following set of information will give a background overview about the waterpoint features:

- `amount_tsh` - Total static head (amount water available to waterpoint)
- `date_recorded` - The date the row was entered
- `funder` - Who funded the well
- `gps_height` - Altitude of the well
- `installer` - Organization that installed the well
- `longitude` - GPS coordinate
- `latitude` - GPS coordinate
- `wpt_name` - Name of the waterpoint if there is one
- `num_private` -
- `basin` - Geographic water basin
- `subvillage` - Geographic location
- `region` - Geographic location
- `region_code` - Geographic location (coded)
- `district_code` - Geographic location (coded)
- `lga` - Geographic location
- `ward` - Geographic location
- `population` - Population around the well
- `public_meeting` - True/False
- `recorded_by` - Group entering this row of data
- `scheme_management` - Who operates the waterpoint
- `scheme_name` - Who operates the waterpoint
- `permit` - If the waterpoint is permitted
- `construction_year` - Year the waterpoint was constructed
- `extraction_type` - The kind of extraction the waterpoint uses
- `extraction_type_group` - The kind of extraction the waterpoint uses
- `extraction_type_class` - The kind of extraction the waterpoint uses
- `management` - How the waterpoint is managed
- `management_group` - How the waterpoint is managed
- `payment` - What the water costs
- `payment_type` - What the water costs
- `water_quality` - The quality of the water
- `quality_group` - The quality of the water
- `quantity` - The quantity of water
- `quantity_group` - The quantity of water
- `source` - The source of the water
- `source_type` - The source of the water
- `source_class` - The source of the water
- `waterpoint_type` - The kind of waterpoint
- `waterpoint_type_group` - The kind of waterpoint


## Project Prediction Models 

Since the project problem is a classifier problem, I'll be building three prediction models:
- `Random Forest`
- `Gradient Boosting`: (xgboost)
- `Logistic regression`

The most performing model will be deployed to production. 

## Project Summary 

Of all the models built, while training the models with the provided dataset, the most performing of all was `Random Forest`. Focus was placed more on the label class where water-pump-status seems functioning but will need repair. This will guide a sudden collapse of any waterpoint since prompt action is taken on any waterpoint that seem to have some trait of malfunctioning. I decided to keen in to where the model would predict the water-pump points that are functioning but need repair. 

To make the model have more predictive power, I did some hypertuning work on some parameters to determine which value of a parameter works best. The final model was built on the most promising hyperparameters. 

After the final model was built, I decided to wrap the model in a `docker container`, which was later deployed on `AWS Cloud`. This repo contain all the files needed to access the model in the cloud. 

## Project Deployment

You can access the project app here ==> 'water-pump-project-serving-env.eba-gyvqqh2a.us-east-1.elasticbeanstalk.com'






