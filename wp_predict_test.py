#!/usr/bin/env python
# coding: utf-8

import requests


host = 'water-pump-project-serving-env.eba-gyvqqh2a.us-east-1.elasticbeanstalk.com'
url = f'http://{host}/predict'


water_pump_code = '45739'

water_pump_info = {
    "amount_tsh": 3.556423,
    "gps_height": 2.954243, 
    "num_private": 0.0, 
    "region_code": 1.230449,
    "district_code": 0.602060, 
    "population": 2.741152, 
    "recorded_by": 1,
    "funder": "government_of_tanzania",
    "installer": "missing",
    "wpt_name": "sokoni",
    "subvillage": "mtakuja",
    "lga": "njombe",
    "scheme_management": "water_board",
    "basin_lake": "lake_tanganyika",
    "public_meeting": "true",
    "permit": "true",
    "management_group": "user-group",
    "payment": "pay_annually",
    "water_quality": "soft",
    "quality_group": "good",
    "quantity_group": "enough",
    "source_type": "dam",
    "waterpoint_type_group": "communal_standpipe",
    "extraction_type_class": "gravity",
    "loc_cluster": 3,
    "water_pump_age": 24
}


# Sending post request to the web services with the customer info 
response = requests.post(url, json=water_pump_info)
response = response.json()
print(response)

if response['wp_repair'] == True:
    print('send a technician to repair the water-pump at: %s' % water_pump_code)
else:
    print('no need sending a technician to %s' % water_pump_code)
