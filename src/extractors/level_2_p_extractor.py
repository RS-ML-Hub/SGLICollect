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
    # GCOM-C/SGLI Level-2 IWPR version 3 QA flags
    __FLAGS: list[str] = ['DATAMISS', 'LAND', 'ATMFAIL', 'CLDICE', 'CLDAFFCTD', 
                        'STRAYLIGHT', 'HIGLINT', 'MODGLINT', 'HISOLZ', 'HITAUA', 
                        'NEGNLW', "SPARE_11" , 'SHALLOW', 'ITERFAILCDOM', 'CHLWARN', 'SPARE_15'
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
        - Decode the 16-bit QA flag of a GCOM-C/SGLI Level-2 in-water properties (IWPR) pixel.
        - Each bit represents a quality condition associated with products such as CHLA, CDOM, and TSM.
        - Multiple flags can be active simultaneously.
        - The function preserves the original integer, its binary representation, and all decoded flag names.
        - The fill value 65535 is treated as invalid data.
        0: "DATAMISS",       # No observation data in one or more bands
        1: "LAND",           # Land pixel
        2: "ATMFAIL",        # Atmospheric correction failure
        3: "CLDICE",         # Apparent cloud/ice (high reflectance)
        4: "CLDAFFCTD",      # Cloud-affected (near-cloud or thin/sub-pixel cloud)
        5: "STRAYLIGHT",     # Stray light anticipated
        6: "HIGLINT",        # High sun glint predicted (atmospheric correction abandoned)
        7: "MODGLINT",       # Moderate sun glint predicted (correction applied)
        8: "HISOLZ",         # Solar zenith larger than threshold
        9: "HITAUA",         # Aerosol optical thickness larger than threshold
        10: "NEGNLW",        # Negative normalized water-leaving radiance in one or more bands
        11: "SPARE_11",      # Reserved / unused
        12: "SHALLOW",       # Water shallower than threshold
        13: "ITERFAILCDOM",  # Iteration failure for CDOM algorithm
        14: "CHLWARN",       # Chlorophyll-a estimate out of range
        15: "SPARE_15"       # Reserved / unused

        Parameters
        ----------
        flags_value : int
            Raw unsigned 16-bit value from Image_data/QA_flag.

        Returns
        -------
        dict
            Original QA value, binary representation,
            QA validity, combined flag names, and individual
            flags (0, 1, or NaN).

        """
        value = int(flags_value)

        if not 0 <= value <= 65535:
            raise ValueError(f"Invalid QA value: {value}")

        valid = value != 65535

        result = {
            "QA_flag_GPORTAL": value,
            "QA_flag_binary_GPORTAL": f"0b{value:016b}",
            "QA_flag_valid_GPORTAL": int(valid),
        }

        active_flags = []

        for bit, name in enumerate(self.__FLAGS):
            if valid:
                is_set = (value >> bit) & 1
            else:
                is_set = np.nan

            result[f"{name}_GPORTAL"] = is_set

            if valid and is_set:
                active_flags.append(name)

        if not valid:
            result["QA_flag_names_GPORTAL"] = "INVALID_QA"
        elif active_flags:
            result["QA_flag_names_GPORTAL"] = "|".join(
                active_flags
            )
        else:
            result["QA_flag_names_GPORTAL"] = "NONE"

        return result
    
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


