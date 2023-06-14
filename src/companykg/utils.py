"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging

import requests

from companykg.settings import CHUNK_SIZE

logger = logging.getLogger(__name__)


def download_zenodo_file(uri: str, dest_path: str) -> None:
    """Zenodo file downloader that maintains O(1) memory consumption"""
    logger.info(f"Downloading {uri} to {dest_path}")
    with requests.get(uri, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
    logger.info("...[DONE]")
