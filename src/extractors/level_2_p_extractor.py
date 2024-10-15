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

class GPortalL2PExtractor(Extractor):
    __FLAGS: list[str] = ['DATAMISS', 'LAND', 'ATMFAIL', 'CLDICE', 'CLDAFFCTD', 
                        'STRAYLIGHT', 'HIGLINT', 'MODGLINT', 'HISOLZ', 'HISENZ', 
                        'TURBIDW', 'SHALLOW', 'ITERFAILCDOM', 'CHLWARN', 'LOWNLW'
                        ]
        
    def __digital_number_to_prod(self, prod_name:str) -> np.ndarray[float, float]:
        """
        Convert digital number to the desired product

        :prod_name str the desired product (Lt01 to Lt11 for radiance) or (Rt01 to Rt11 for reflectance)
        """
        # Get product data
        real_prod_name = prod_name.upper()
        data = self._h5['Image_data/' + real_prod_name]

        # Validate
        prod = data[:].astype(np.float32)
        if 'Error_DN' in data.attrs:
            prod[prod == data.attrs['Error_DN'][0]] = np.nan
        if 'Maximum_valid_DN' in data.attrs:
            prod[prod > data.attrs['Maximum_valid_DN'][0]] = np.nan
        if 'Minimum_valid_DN' in data.attrs:
            prod[prod < data.attrs['Minimum_valid_DN'][0]] = np.nan

        # Convert DN to physical value
        slope = data.attrs['Slope'][0]
        offset = data.attrs['Offset'][0]
        prod = prod * slope + offset
        return prod
        
    def __calculate_flags(self, flags_value: int) -> dict:
        """
        Calculates a dictionary mapping of flag->boolean that indicates each flag and if it's set or not

        :flags_value int the integer value of the flag
        """
        flags = {}
        order = 0
        while order < 15:
            flags[self.__FLAGS[order]+"_GPORTAL"] = flags_value % 2
            flags_value //= 2
            order += 1
        return flags

    def get_pixel(self, lat:float, lon:float) -> dict:
        lat_mat, lon_mat = self.get_lat_lon()
        row, col, distance = find_entry(lat_mat, lon_mat, lat, lon)
        # print("=> matching point with distance^2 = %f" % (distance))
        flags = self.__calculate_flags(self._h5["Image_data/QA_flag"][row, col])
        pixel = {
            "Chla_GPORTAL": self.__digital_number_to_prod("Chla")[row, col],
            "aCDOM_412_GPORTAL": self.__digital_number_to_prod("CDOM")[row, col],
            "TSM_GPORTAL": self.__digital_number_to_prod("TSM")[row, col],
        }
        pixel.update(flags)
        return pixel


