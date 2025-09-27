#
# Copyright (c) 2023 Muhammad Salah msalah.29.10@gmail.com
# Licensed under AGPL-3.0-or-later.
# Refer to COPYING.txt for the AGPL license.
# All rights reserved.
# This project is developed as part of my research in the Remote Sensing Laboratory
# in Kyoto University of Advanced Science towards my Master's Degree course.
# The research was mainly supervised by Professor Salem Ibrahim Salem.
#
from src.args import args
from src.download import download
from src.search import search
from src.extract import extract
from src.utils import empty_temp

def main() -> None:
    if args.search:
        search(args)
    elif args.download:
        download(args)
    elif args.extract:
        extract(args)
    empty_temp()

if __name__ == "__main__":
    main()
