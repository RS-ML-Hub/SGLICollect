#
# Copyright (c) 2023 Muhammad Salah msalah.29.10@gmail.com
# Licensed under AGPL-3.0-or-later.
# Refer to COPYING.txt for the AGPL license.
# All rights reserved.
# This project is developed as part of my research in the Remote Sensing Laboratory
# in Kyoto University of Advanced Science towards my Master's Degree course.
# The research was mainly supervised by Professor Salem Ibrahim Salem.
#

from pathlib import Path
from src.jasmes import JASMESInternalProd
from src.extractors.extractor_interface import Extractor
import numpy as np
from netCDF4 import Dataset

class JASMESExtractor(Extractor):
    def __init__(self, path: Path, prod:JASMESInternalProd):
        self.__prod = prod
        self.__nc = Dataset(path, "r")

    def __handle_digital_number(self, rrs: bool = False) -> np.ndarray:
        """
        Convert digital number to the desired product with DN masking
        :prod JASMESProd 
        """
        # Get data
        prod:str = str(self.__prod.value)

        data = self.__nc.variables[prod]

        # Extract raw DN data as numpy array
        raw_dn = data[:].data.astype(np.float32)

        # Mask DN values that are outside valid range or represent error/no observation
        mask = np.full(raw_dn.shape, False, dtype=bool)

        # Mask DN < Minimum_valid_DN
        if hasattr(data, "Minimum_valid_DN"):
            mask |= (raw_dn < data.Minimum_valid_DN)

        # Mask DN > Maximum_valid_DN
        if hasattr(data, "Maximum_valid_DN"):
            mask |= (raw_dn > data.Maximum_valid_DN)

        # Mask DN == Error_DN
        if hasattr(data, "Error_DN"):
            mask |= (raw_dn == data.Error_DN)

        # Mask DN == No_observation_DN
        if hasattr(data, "No_observation_DN"):
            mask |= (raw_dn == data.No_observation_DN)

        # Set masked DN to NaN before any conversion
        raw_dn = np.where(mask, np.nan, raw_dn)

        # Convertion from DN to physical value is done automatically by netCDF4 if scale and offset attributes exist
        if rrs:
            scale  = data.Rrs_scale_factor
            offset = data.Rrs_add_offset
            # VIP Note: as NetCDF4 automatically applies scale_factor and add_offset to Rrs, we need to reverse that first
            #  then apply the Rrs scaling and offset
            digital_data = (raw_dn - data.add_offset)/data.scale_factor
            physical_data = digital_data * scale + offset
        else:
            physical_data = raw_dn

        return physical_data
    
    def get_lat_lon(self) -> tuple[list[float], list[float]]:
        return self.__nc.variables["Latitude"][:].data, self.__nc.variables["Longitude"][:].data
    
    @classmethod
    def get_file_ext(self)->str:
        return "nc"
    @classmethod
    def make_file_name(self, file_name)->str:
        return file_name

    def __find_entry(self, lat_arr, lon_arr, lat, lon) ->tuple[int, int]:
        dist = 1000
        lat_index = -1
        for i,l in enumerate(lat_arr):
            if np.abs(l - lat) < dist:
                dist = np.abs(l-lat)
                lat_index = i
        lon_index = -1
        dist = 1000
        for i,l in enumerate(lon_arr):
            if np.abs(l - lon) < dist:
                dist = np.abs(l-lon)
                lon_index = i
        return lat_index, lon_index

    def get_pixel(self, lat:float, lon:float) -> dict:
        lat_mat, lon_mat = self.get_lat_lon()
        row, col = self.__find_entry(lat_mat, lon_mat, lat, lon)
        if str(self.__prod.value).startswith("NWL"):
            pixel = {
                f"{str(self.__prod.value)}_JASMES": self.__handle_digital_number()[row, col],
                f"{str(self.__prod.value).replace('NWLR', 'Rrs')}_JASMES": self.__handle_digital_number(rrs=True)[row, col]
            }
        else:
            pixel = {
                f"{str(self.__prod.value)}_JASMES": self.__handle_digital_number()[row, col],
            }
        return pixel