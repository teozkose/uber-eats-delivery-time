"""Download the cleaned Uber Eats delivery dataset (MIT-licensed copy published on GitHub)."""
import os
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/liamarguedas/uber-eats-delivery-time/main/data/cleaned_data.csv"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_data.csv")

if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
    print(f"already present: {DEST}")
    sys.exit(0)
print(f"downloading {URL}")
urllib.request.urlretrieve(URL, DEST)
print(f"saved {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")
