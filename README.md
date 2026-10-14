<h3 align="center">SGLICollect</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](/COPYING.txt)

</div>

---

<p align="center"> SGLICollect is an open-source software for searching, downloading and reading SGLI/GCOM-C products from both GPORTAL and JASMES. 
    <br> 
    Acknowledgment: This project was developed as part of my master's degree at Kyoto University of Advanced Science, under the Remote Sensing Laboratory. 
    <br> 
    This project is built for research purposes to collect and extract data.
    <br> 
    The research contributes towards my master's degree, under the supervision of Professor Salem Ibrahim Salem.
    <br> 
</p>

## Installation

This section explains how to download and set up SGLICollect for various environments (The GUI version of SGLICollect is currently under construction)

### Stand-alone installation (Linux and MacOS only)

1. Download the bash installer:
`curl -L -o SGLICollect_installer.sh https://raw.githubusercontent.com/muhammads97/SGLICollect/main/installer.sh`

2. Run the installer:
`sh SGLICollect_installer.sh`

3. Add SGLICollect to the system path

Now you can use SGLICollect as follows:
`SGLICollect --version`

### Conda python installation

To use SGLICollect inside a conda environment please follow the instructions in this section

1. Clone this repo:
`git clone https://github.com/RS-ML-Hub/SGLICollect.git`
2. Install the conda environment:
`conda env create -f environment.yml`
3. Activate the conda environment:
`conda activate SGLICollect`
4. Validate installation:
`python <path to SGLICollect> --version`

## Usage

The SGLICollect operates using a JSON configuration file and a CSV file.

The services provided are as follows:
1. Search (using latitude, longitude, and date)
2. Download (using product URL)
3. Extract (using product directory, latitude, and longitude)

<img width="1088" alt="Screenshot 2024-03-15 at 13 55 35" src="https://github.com/muhammads97/SGLICollect/assets/33841931/cbf02c2b-70a9-48d3-b97b-320eeda2548f">

<img width="1089" alt="Screenshot 2024-03-15 at 13 53 37" src="https://github.com/muhammads97/SGLICollect/assets/33841931/78b60009-5902-4622-a032-6f162bfe2fc5">

All the parameters for these services are set using the JSON configuration file and the CSV file

To run the SGLICollect, set the parameters needed in `config.json` and run:
for the stand-alone version:
`SGLICollect -c <path to config.json>`
for the conda python installation:
`python <path to SGLICollect> -c <path to config.json>`

Configuration options:
1. `operations`: an array of the operations that will be performed, the available options are: `search`, `download`, and `extract`.
2. `args`: an object containing the arguments to be used for these operations:
    - `product`      : satellite product, please refer to the [list of products](#list-of-products).
    - `csv`          : a path to a CSV file for bulk processing mode. (This will override the parameters provided for single-entry mode), refer to [CSV file format](#csv-file-format).
    - `api`          : `GPORTAL` or `JASMES` (Default: `GPORTAL`).
    - `download_dir` : Download directory (for both single-entry and bulk-processing). If not provided, the files will be downloaded in the temp directory and they will be deleted once the processing ends.
    - `cred`         : path to JSON file containing account and password (refer to [Credentials file](#credentials-file)).
    - `product_dir`  : directory of all products used for the extraction step for bulk-processing mode.
    - `no_repeat`    : `true` or `false`. Used for search and download, if the entry already exists in the CSV or the file is already downloaded the operation will be skipped. (default: false)


### List of products

#### GPortal
1. `L1B`: Level 1 B SGLI product from GPortal.
2. `L2P`: Level 2 Water Quality Products from GPortal.
3. `L2R`: Level 2 Remote Sensing Reflectance from GPortal.

#### JASMES
1. `ALL`: All JASMES Products will be obtained:
    - NWLR_380
    - NWLR_412
    - NWLR_443
    - NWLR_490
    - NWLR_530
    - NWLR_565
    - NWLR_670
    - CDOM
    - CHLA
    - TSM
    - SST          

### Credentials file

a Json file containing `account` and `password` for GPortal or JASMES
Example:
```
{
    "account": "username",
    "password": "password"
}
```

### CSV file format

#### For Search:

1. date
2. lat
3. lon

##### Search output:

1. Output columns for GPORTAL:
    - identifier
    - file_status
    - download_url
    - preview_url
    - cloud_coverage (%)
2. Output columns for JASMES:
    - ftp_path_<product> (example: ftp_path_CDOM)

#### For Download:

1. download_url (GPortal)/ftp_path_<product> (JASMES)

##### Download output:

No updates to the CSV, the files will be downloaded either in the specified download_dir or in the temp folder.

#### For Extract:

1. identifier (GPortal)/ ftp_path_<product> (JASMES)
2. lat
3. lon

##### Extract output (GPORTAL):

1. L1B:
    - Rt11_GPORTAL: Reflectance at band 11  
    - Rt10_GPORTAL: Reflectance at band 10  
    - Rt09_GPORTAL: Reflectance at band 09  
    - Rt08_GPORTAL: Reflectance at band 08  
    - Rt07_GPORTAL: Reflectance at band 07  
    - Rt06_GPORTAL: Reflectance at band 06  
    - Rt05_GPORTAL: Reflectance at band 05  
    - Rt04_GPORTAL: Reflectance at band 04  
    - Rt03_GPORTAL: Reflectance at band 03  
    - Rt02_GPORTAL: Reflectance at band 02   
    - Rt01_GPORTAL: Reflectance at band 01
    - Lt11_GPORTAL: Radiance at band 11
    - Lt10_GPORTAL: Radiance at band 10
    - Lt09_GPORTAL: Radiance at band 09
    - Lt08_GPORTAL: Radiance at band 08
    - Lt07_GPORTAL: Radiance at band 07
    - Lt06_GPORTAL: Radiance at band 06
    - Lt05_GPORTAL: Radiance at band 05
    - Lt04_GPORTAL: Radiance at band 04
    - Lt03_GPORTAL: Radiance at band 03
    - Lt02_GPORTAL: Radiance at band 02
    - Lt01_GPORTAL: Radiance at band 01
    - land_GPORTAL: percentage of land in the pixel
2. L2R:
    - Rrs_670_GPORTAL: Remote sensing reflectance at wavelength 670 nm
    - Rrs_565_GPORTAL: Remote sensing reflectance at wavelength 565 nm
    - Rrs_530_GPORTAL: Remote sensing reflectance at wavelength 530 nm
    - Rrs_490_GPORTAL: Remote sensing reflectance at wavelength 490 nm
    - Rrs_443_GPORTAL: Remote sensing reflectance at wavelength 443 nm
    - Rrs_412_GPORTAL: Remote sensing reflectance at wavelength 412 nm
    - Rrs_380_GPORTAL: Remote sensing reflectance at wavelength 380 nm
3. L2P:
    - Chla_GPORTAL      : Chlorophyll-a concentration using JAXA's Standard Chla Algorithm for GPortal (mg/m3)
    - aCDOM_412_GPORTAL : absorption of Colored Dissolved Organic Matter at wavelength 412 nm (1/m)
    - TSM_GPORTAL       : Total Suspended Matter (g/m3)


##### Extract output (JASMES):
1. NWLR_380_JASMES
2. NWLR_412_JASMES
3. NWLR_443_JASMES
4. NWLR_490_JASMES
5. NWLR_530_JASMES
6. NWLR_565_JASMES
7. NWLR_670_JASMES
8. Rrs_380_JASMES
9. Rrs_412_JASMES
10. Rrs_443_JASMES
11. Rrs_490_JASMES
12. Rrs_530_JASMES
13. Rrs_565_JASMES
14. Rrs_670_JASMES
15. CDOM_JASMES
16. CHLA_JASMES
17. TSM_JASMES
18. SST_JASMES

### Example usage (single-entry)

`config.json` :
```
{
  "operations": ["search", "download", "extract"],
  "args": {
    "api": "GPORTAL",
    "product": "L2R",
    "latitude": 44.853750035714505,
    "longitude": 138.32875040812206,
    "date": "2023/11/21",
    "download_dir": "~/Downloads/",
    "cred": "./cred.json"
  }
}
```
Note: in this example, product_path doesn't need to be set because after the download finishes the product_path will be calculated automatically.

### Example usage (bulk-processing)

`config.json` :
```
{
  "operations": ["search", "download", "extract"],
  "args": {
    "api": "GPORTAL",
    "product": "L2R",
    "csv": "./test.csv"
    "no_repeat": false,
    "download_dir": "~/Downloads/",
    "product_dir": "~/Downloads/",
    "cred": "./cred.json"
  }
}
```
### How to Cite SGLICollect
If you use SGLICollect in your research or project, please cite it using the following reference:

Salah, Muhammad & Salem, Salem Ibrahim (2024). muhammads97/SGLICollect: v2.1.0 (v2.1.0). Zenodo. https://doi.org/10.5281/zenodo.10819786
