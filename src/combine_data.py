import os
import pandas as pd
from tqdm.auto import tqdm


def combine_data(source_dir, target_dir=None, *, outfname):
    """Extracts and combine data files from separate data file

    Parameters:
        source_dir (str): Input directory containing data files
        target_dir (str): Output directory to save combined data. if None,
        the combined data is saved in the same location as the source_dir.
        outfname (str): Output filename
    """
    if target_dir is None:
        target_dir = os.path.normpath(source_dir)
    else:
        os.makedirs(target_dir, exist_ok=True)

    filepaths = (os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith("txt"))
    filepaths = sorted(filepaths, key=lambda x: int(x.split("_")[-1].split(".")[-2]))

    data = {"SPD": [], "ANG": [], "TRQ": []}
    for filepath in tqdm(filepaths, total=len(filepaths), desc="Reading and extracting data from files"):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                speed, angle, torque = line.split(",")
                spd = speed.split(":")
                ang = angle.split(":")
                torq = torque.split(":")
                data["SPD"].append(float(spd[-1]))
                data["ANG"].append(float(ang[-1]))
                data["TRQ"].append(float(torq[-1]))

    df = pd.DataFrame.from_dict(data)
    df.to_csv(f"{target_dir}/{outfname}.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="python combine_data.py", description="Sensor Data",
                                     usage="%(prog)s [options] [")
    parser.add_argument("--combine", "-c", action='store_true', help="combines all data files in source directory")
    parser.add_argument("--source_dir", "-i", type=str, help="Source directory")
    parser.add_argument("--target_dir", "-o", type=str, help="Target directory to saved combined data")
    parser.add_argument("--fname", "-f", type=str, help="Output filename")

    opt = parser.parse_args()

    SOURCE_DIR = opt.source_dir
    TARGET_DIR = opt.target_dir
    FNAME = opt.fname

    if opt.combine:
        combine_data(source_dir=SOURCE_DIR, target_dir=TARGET_DIR, outfname=FNAME)
    else:
        parser.parse_args(["-h"])