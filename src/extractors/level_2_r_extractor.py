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
                        'GAMMA_OUT', 'OVERITER', 'NEGNLW', 'HIGHWS', 'SPARE_14',
                        "SPARE_15"
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
        - Decode the 16-bit quality assurance (QA) flag of a GCOM-C/SGLI Level-2 reflectance pixel.
        - Each bit represents a specific quality condition, such as land, cloud contamination, sun glint, or atmospheric correction failure.
        - Multiple conditions can be active simultaneously.
        - The function returns the original QA value, its binary representation, the names of all active flags, and individual
           binary indicators (0 or 1) for each quality condition.
        - The fill value (65535) is treated as invalid QA data, with individual decoded flags set to NaN.        
        0: "DATAMISS",       # No observation data in one or more band[s]
        1: "LAND",           # Land pixel
        2: "ATMFAIL",        # Atmospheric correction failure
        3: "CLDICE",         # Apparent cloud/ice (high reflectance)
        4: "CLDAFFCTD",      # Cloud-affected (near-cloud or thin/sub-pixel cloud)
        5: "STRAYLIGHT",     # Stray light anticipated
        6: "HIGLINT",        # High sun glint predicted
        7: "MODGLINT",       # Moderate sun glint predicted (correction applied)
        8: "HISOLZ",         # Solar zenith larger than threshold
        9: "HITAUA",         # Aerosol optical thickness larger than threshold
        10: "GAMMA-OUT",     # Atmospheric correction warning: Gamma out-of-bounds
        11: "OVERITER",      # Maximum iterations reached for NIR correction
        12: "NEGNLW",        # Negative nLw in one or more bands
        13: "HIGHWS"         # Surface wind speed higher than threshold
        # Bits 14 and 15 are SPARE    
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
            raise ValueError(f"Invalid 16-bit QA value: {value}")

        result = {
            "QA_flag_GPORTAL": value,
            "QA_flag_binary_GPORTAL": f"0b{value:016b}",
            "QA_flag_valid_GPORTAL": int(value != 65535),
        }

        # Missing QA data
        if value == 65535:
            result["QA_flag_names_GPORTAL"] = "INVALID_QA"

            for name in self.__FLAGS:
                result[f"{name}_GPORTAL"] = np.nan

            return result

        active_flags = []

        # Decode every bit independently
        for bit, name in enumerate(self.__FLAGS):
            is_set = (value >> bit) & 1
            result[f"{name}_GPORTAL"] = is_set

            if is_set:
                active_flags.append(name)

        # Human-readable combined flags
        result["QA_flag_names_GPORTAL"] = (
            "|".join(active_flags)
            if active_flags else "NONE"
        )

        return result

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


