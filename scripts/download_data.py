from pathlib import Path
from urllib.request import urlretrieve

base_url = (
    "https://raw.githubusercontent.com/"
    "declare-lab/MELD/master/data/MELD/"
)

data_dir = Path("../data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "dev", "test"]:
    filename = f"{split}_sent_emo.csv"
    urlretrieve(base_url + filename, data_dir / filename)
    print(f"Downloaded {filename}")