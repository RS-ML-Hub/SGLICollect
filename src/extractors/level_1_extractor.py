#
# Copyright (c) 2023 Muhammad Salah msalah.29.10@gmail.com
# Licensed under AGPL-3.0-or-later.
# Refer to COPYING.txt for the AGPL license.
# All rights reserved.
# This project is developed as part of my research in the Remote Sensing Laboratory
# in Kyoto University of Advanced Science towards my Master's Degree course.
# The research was mainly supervised by Professor Salem Ibrahim Salem.
#

from .extractor_interface import Extractor
from .utils import find_entry
import numpy as np



class GPortalL1BExtractor(Extractor):
        
    def __handle_digital_number(self, prod_name:str) -> np.ndarray[float, float]:
        """
        Convert digital number to the desired product

        :prod_name str the desired product (Lt01 to Lt11 for radiance) or (Rt01 to Rt11 for reflectance)
        """
        # Get data
        radiance = prod_name.startswith("L")
        if radiance: real_prod_name = prod_name.replace('Lt', 'Lt_VN')
        else: real_prod_name = prod_name.replace('Rt', 'Lt_VN')

        data = self._h5['Image_data/' + real_prod_name]
        rad_or_ref = data[:]

        errors = np.zeros(shape=rad_or_ref.shape, dtype=np.bool_)

        # Validate
        errors[rad_or_ref == data.attrs['Error_DN'][0]] = True
        errors[rad_or_ref > data.attrs['Maximum_valid_DN'][0]] = True
        errors[rad_or_ref < data.attrs['Minimum_valid_DN'][0]] = True

        # Convert DN to physical value
        if radiance:
            slope = data.attrs['Slope'][0]
            offset = data.attrs['Offset'][0]
        else:
            slope = data.attrs['Slope_reflectance'][0]
            offset = data.attrs['Offset_reflectance'][0]
        mask = data.attrs['Mask']

        rad_or_ref = (rad_or_ref&mask) * slope.astype(np.float32) + offset
        rad_or_ref[errors] = np.nan
        return rad_or_ref

    def get_pixel(self, lat:float, lon:float) -> dict:
        lat_mat, lon_mat = self.get_lat_lon()
        row, col, _ = find_entry(lat_mat, lon_mat, lat, lon)
        pixel = {
            "Lt01_GPORTAL": self.__handle_digital_number("Lt01")[row, col],
            "Lt02_GPORTAL": self.__handle_digital_number("Lt02")[row, col],
            "Lt03_GPORTAL": self.__handle_digital_number("Lt03")[row, col],
            "Lt04_GPORTAL": self.__handle_digital_number("Lt04")[row, col],
            "Lt05_GPORTAL": self.__handle_digital_number("Lt05")[row, col],
            "Lt06_GPORTAL": self.__handle_digital_number("Lt06")[row, col],
            "Lt07_GPORTAL": self.__handle_digital_number("Lt07")[row, col],
            "Lt08_GPORTAL": self.__handle_digital_number("Lt08")[row, col],
            "Lt09_GPORTAL": self.__handle_digital_number("Lt09")[row, col],
            "Lt10_GPORTAL": self.__handle_digital_number("Lt10")[row, col],
            "Lt11_GPORTAL": self.__handle_digital_number("Lt11")[row, col],
            "Rt01_GPORTAL": self.__handle_digital_number("Rt01")[row, col],
            "Rt02_GPORTAL": self.__handle_digital_number("Rt02")[row, col],
            "Rt03_GPORTAL": self.__handle_digital_number("Rt03")[row, col],
            "Rt04_GPORTAL": self.__handle_digital_number("Rt04")[row, col],
            "Rt05_GPORTAL": self.__handle_digital_number("Rt05")[row, col],
            "Rt06_GPORTAL": self.__handle_digital_number("Rt06")[row, col],
            "Rt07_GPORTAL": self.__handle_digital_number("Rt07")[row, col],
            "Rt08_GPORTAL": self.__handle_digital_number("Rt08")[row, col],
            "Rt09_GPORTAL": self.__handle_digital_number("Rt09")[row, col],
            "Rt10_GPORTAL": self.__handle_digital_number("Rt10")[row, col],
            "Rt11_GPORTAL": self.__handle_digital_number("Rt11")[row, col],
            "land_GPORTAL": self._h5["Image_data"]["Land_water_flag"][:][row, col]
        }            
        return pixel


