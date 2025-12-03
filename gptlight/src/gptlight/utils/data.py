from __future__ import annotations

from typing import Optional

import urllib.request
import zipfile
import io
import logging
import pandas as pd

def fetch_verdict_text() -> str:
    """
        Fetch Verdict Text
    """
    
    url = ("https://github.com/Yatoute/"
        "gpt-light/tree/main/gptlight/assets/datasets"
        "the-verdict.txt")

    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")

    return text


def fetch_sms_spam_collection() -> Optional[pd.DataFrame] :
    """ 
    Fetch sms and spam collection
    """
    
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    
    try:
        response = urllib.request.urlopen(url)
        
        zip_data = response.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            with z.open("SMSSpamCollection") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=["Label", "Text"])
    except Exception as e:
        logging.error(f"Error when fetching the sms and spam collection : {e}")
        return None

    return df