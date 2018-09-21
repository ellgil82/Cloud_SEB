# Cloud_SEB

This folder contains scripts used for processing data collected from automatic weather stations (AWSs), from the British Antarctic Survey's atmospheric research aircraft facility, MASIN, and the Met Office Unified Model data using python.

This work has formed the basis of the second part of my PhD thesis, and of a paper currently in preparation.
 
 +++++++++++

Observational data: 

AWS data are taken from AWS14 and 15, operated and managed by IMAU. AWS15 was operational 2009-14, while AWS14 was still operational at the time of writing in Sep 2018. They both measure surface meteorological variables and radiation terms, while AWS14 measures the full surface energy balance. The data are described in detail in Kuipers Munneke et al. (2012) The Cryosphere 6, 353-363. doi: 10.5194/tc-6-353-2012  

* Link to general metadata and information: https://www.projects.science.uu.nl/iceclimate/aws/antarctica.php

* Link to specific information about AWS14: https://www.projects.science.uu.nl/iceclimate/aws/files_oper/oper_29157.php 

Aircraft data are collected by MASIN , the British Antarctic Survey's instrumented Twin Otter. The full instrumental specifications can be found here on the BAS website (link below), and the configuration and processing methods used during the OFCAP campaign (which these scripts are written to process) are described in detail in Lachlan-Cope et al. (2016) Atmospheric Chemistry and Physics 16, 15605-15617. doi: 10.5194/acp-16-15605-2016.

* Link to information about MASIN: https://www.bas.ac.uk/polar-operations/sites-and-facilities/facility/airborne-science-and-technology/twin-otter-meteorological-airborne-science-instrumentation-masin/

* Link to instrument specifications: https://www.bas.ac.uk/polar-operations/sites-and-facilities/facility/airborne-science-and-technology/twin-otter-meteorological-airborne-science-instrumentation-masin/instrumentation-specifications/

+++++++++++

Model data:

Code is written to process model data from the Met Office Unified Model (vn11.1), in their proprietary format, .pp, which results in a smaller file size. However, this could very easily be adapted to work with netCDF files, and model output from any version of the model from vn7 onwards (provided the right parameters are available). The scripts require the installation of the python package Iris (https://scitools.org.uk/iris/docs/latest/), which is a tool for processing CF-compliant climate and model data. 