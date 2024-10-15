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

class GPortalL2RExtractor(Extractor):
    __FLAGS: list[str] = ['DATAMISS', 'LAND', 'ATMFAIL', 'CLDICE', 'CLDAFFCTD', 
                        'STRAYLIGHT', 'HIGLINT', 'MODGLINT', 'HISOLZ', 'HITAUA', 
                        'EPSOUT', 'OVERITER', 'NEGNLW', 'HIGHWS', 'TURBIDW'
                        ]
        
    def __digital_number_to_rrs(self, prod_name:str) -> np.ndarray[float, float]:
        """
        Convert digital number to the desired product

        :prod_name str the desired product (Lt01 to Lt11 for radiance) or (Rt01 to Rt11 for reflectance)
        """
        # Get Rrs data
        real_prod_name = prod_name.replace('Rrs', 'NWLR')
        data = self._h5['Image_data/' + real_prod_name]

        # Validate
        rrs = data[:].astype(np.float32)
        if 'Error_DN' in data.attrs:
            rrs[rrs == data.attrs['Error_DN'][0]] = np.nan
        if 'Maximum_valid_DN' in data.attrs:
            rrs[rrs > data.attrs['Maximum_valid_DN'][0]] = np.nan
        if 'Minimum_valid_DN' in data.attrs:
            rrs[rrs < data.attrs['Minimum_valid_DN'][0]] = np.nan

        # Convert DN to physical value
        slope = data.attrs['Rrs_slope'][0]
        offset = data.attrs['Rrs_offset'][0]
        rrs = rrs * slope + offset
        return rrs
        
    def __calculate_flags(self, flags_value: int) -> dict:
        """
        Calculates a dictionary mapping of flag->boolean that indicates each flag and if it's set or not

        :flags_value int the integer value of the flag
        """
        flags = {}
        order = 0
        while order < 15:
            flags[self.__FLAGS[order]+"_GPROTAL"] = flags_value % 2
            flags_value //= 2
            order += 1
        return flags

    def get_pixel(self, lat:float, lon:float) -> dict:
        lat_mat, lon_mat = self.get_lat_lon()
        row, col, _ = find_entry(lat_mat, lon_mat, lat, lon)
        # print("=> matching point with distance^2 = %f at %d %d" % (distance, row, col))
        # print(lat_mat[row, col], lon_mat[row, col])

        flags = self.__calculate_flags(self._h5["Image_data/QA_flag"][row, col])
        pixel = {
            "Rrs_380_GPORTAL": self.__digital_number_to_rrs("Rrs_380")[row, col],
            "Rrs_412_GPORTAL": self.__digital_number_to_rrs("Rrs_412")[row, col],
            "Rrs_443_GPORTAL": self.__digital_number_to_rrs("Rrs_443")[row, col],
            "Rrs_490_GPORTAL": self.__digital_number_to_rrs("Rrs_490")[row, col],
            "Rrs_530_GPORTAL": self.__digital_number_to_rrs("Rrs_530")[row, col],
            "Rrs_565_GPORTAL": self.__digital_number_to_rrs("Rrs_565")[row, col],
            "Rrs_670_GPORTAL": self.__digital_number_to_rrs("Rrs_670")[row, col],
        }
        pixel.update(flags)
        return pixel


