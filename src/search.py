#
# Copyright (c) 2023 Muhammad Salah msalah.29.10@gmail.com
# Licensed under AGPL-3.0-or-later.
# Refer to COPYING.txt for the AGPL license.
# All rights reserved.
# This project is developed as part of my research in the Remote Sensing Laboratory
# in Kyoto University of Advanced Science towards my Master's Degree course.
# The research was mainly supervised by Professor Salem Ibrahim Salem.
#


from argparse import Namespace
import pandas as pd
from tqdm import tqdm
import time
from src import download
from src.api_types import SGLIAPIs
from src.gportal import GportalApi, GPortalLvlProd, GPortalResolution
from src.jasmes import JasmesCollector

"""
Utility function to get the value from a dict and validate it
"""
def get_value(d: dict, key: str):
    """
    Utility function to get the value from a dict and validate it
    """
    if key in d and d[key]:
        return d[key]
    
def fill_group(df, g, api):
    
    for j in range(len(g)):
        if api ==SGLIAPIs.GPORTAL:
            df.loc[df.index[g.index[j]], "identifier"] = g.iloc[0]["identifier"]
            df.loc[df.index[g.index[j]], "file_status"] = g.iloc[0]["file_status"]
            #df.loc[df.index[g.index[j]], "resolution"] = g.iloc[0]["resolution"]
            df.loc[df.index[g.index[j]], "download_url"] = g.iloc[0]["download_url"]
            df.loc[df.index[g.index[j]], "preview_url"] = g.iloc[0]["preview_url"]
            df.loc[df.index[g.index[j]], "cloud_coverage"] = g.iloc[0]["cloud_coverage"]
        elif api == SGLIAPIs.JASMES:
            df.loc[df.index[g.index[j]], "file_name"] = g.iloc[0]["file_name"]
            df.loc[df.index[g.index[j]], "file_size"] = g.iloc[0]["file_size"]
            df.loc[df.index[g.index[j]], "ftp_path"] = g.iloc[0]["ftp_path"]
            df.loc[df.index[g.index[j]], "box_id"] = g.iloc[0]["box_id"]

def search(args: Namespace):
    """
    Bulk search operation using csv file
    arguments provided through json file or cmdline arguments:
        - api: GPORTAL or JASMES, default: GPORTAL
        - product: GPortal Products or Jasmes Products
        - csv: path to csv file
        - no_repeat: boolean, if identifier exists will not search the corresponding row
    CSV file columns:
        - date
        - lat
        - lon
        - identifier/file_name: optional (if no_repeat set and identifier/file_name provided the corresponding row will be skipped)

    Output columns for GPORTAL:
        - identifier
        - file_status
        - resolution
        - download_url
        - preview_url
        - cloud_coverage (%)
    Output columns for JASMES:
        - ftp_path
        - file_name
        - file_size
        - box_id
    """
    if args.api == SGLIAPIs.GPORTAL:
        pl = GPortalLvlProd(args.product)
        api = GportalApi(pl)
        id_key = "identifier"
    elif args.api == SGLIAPIs.JASMES:
        api = JasmesCollector()
        api.set_auth_details(args.cred)
        id_key = "file_name"

    print("=============================")
    print("Searching CSV ...")
    print("=============================")

    df = pd.read_csv(args.csv, low_memory=True) # read csv
    # Support CSV files containing either 'date' or 'time'
    if "date" in df.columns:
        date_source = "date"
    elif "time" in df.columns:
        date_source = "time"
    else:
        raise ValueError(
            "Input CSV must contain either a 'date' or 'time' column."
        )

    # Parse timestamps and normalise dates to YYYY/MM/DD.
    # UTC is used because OC-CCI timestamps end with 'Z'.
    parsed_dates = pd.to_datetime(
        df[date_source],
        errors="coerce",
        utc=True
    )

    # Fail clearly if any timestamps cannot be parsed
    invalid = parsed_dates.isna()

    if invalid.any():
        examples = df.loc[invalid, date_source].head(5).tolist()
        raise ValueError(
            f"Found {invalid.sum()} invalid dates in "
            f"column '{date_source}'. Examples: {examples}"
        )

    # Create or update the date column for SGLICollect
    df["date"] = parsed_dates.dt.strftime("%Y/%m/%d")

    # Group observations by location and observation date
    grouped = df.groupby(["lat", "lon", "date"])
    pbar = tqdm(total=len(df), position=0, leave=True) # prepare progress bar

    for i, ((lat, lon, date), g) in enumerate(grouped):

        # Skip in situ observations before SGLI availability
        if pd.to_datetime(date) < pd.Timestamp("2018-01-01"):
            continue
    
        id = get_value(g.iloc[0].fillna(0).to_dict(), id_key)
        resolution = GPortalResolution.H

        # progress if no repeat and id exists
        if id and args.no_repeat: 
            fill_group(df, g, args.api)
            pbar.update(len(g))
            continue

        search_error = False
        while True:
            if search_error:
                time.sleep(10)
            try:
                # send search request 
                result = api.search(date, lat, lon, resolution, verbose=False)
                search_error = False
            except Exception as e:
                print(e)
                search_error = True
            if not search_error: break

        # if results returned add to the data
        if result != None:
            for j in range(len(g)):
                result.to_dataframe(df, g.index[j])

        pbar.update(len(g)) # update progress bar

        # save to csv every 100 row
        if (i % 100 == 0): df.to_csv(args.csv, index=False)

    df.to_csv(args.csv, index=False) # save to csv
    pbar.close()

    # move to download option if download is set
    if args.download:
        download(args)
        

