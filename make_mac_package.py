"""把 Mac mini 需要的程式與資料打包成 mac_v2.zip (解壓縮後合併進 Mac 上的 mac_package)。

用法: python make_mac_package.py            程式 + CIC-IDS2017 壓縮檔 (約 580 MB)
      python make_mac_package.py --code     只有程式 (幾十 KB，之後更新程式用)
      python make_mac_package.py --cic18    程式 + CSE-CIC-IDS2018 抽樣後的資料 (約 1.1 GB)
"""
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "mac_package"
CODE = ["nids_repro.py", "nids_core.py", "nids_data.py", "tune.py", "cicids.py", "multiseed.py",
        "unsw.py", "run_v2.py", "shortcut_v2.py", "adversarial_v2.py", "pipeline_v2.py",
        "baselines_v2.py", "yin_repro.py", "cic2018.py", "bench_devices.py"]
SCRIPTS = ["tune_mac.sh", "pipeline_mac.sh", "collect_v2.sh", "after_pipeline_mac.sh"]
DATA = ["data/cicids2017/original/MachineLearningCSV.zip",
        "data/cicids2017/improved/CICIDS2017_improved.zip"]
SMALL_DATA = ["data/KDDTest-21.txt", "data/cicids2017/improved.json"]   # 小檔案，連 --code 也一起打包
DATA_CIC18 = ["data/cicids2018/improved.npz", "data/cicids2018/improved.json"]


def cloud():
    """雲端 GPU 主機用的套件: nids_cloud/ (程式 + 雲端腳本 + 套件版本 + CSE-CIC-IDS2018 資料)。"""
    out = ROOT / "cloud_cic18.zip"
    with zipfile.ZipFile(out, "w") as z:
        files = [(ROOT / n, n) for n in CODE + ["requirements_laptop.txt"]]
        files += [(p, p.name) for p in (ROOT / "cloud").glob("*.sh")]
        for f, name in files:
            info = zipfile.ZipInfo.from_file(f, f"nids_cloud/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if f.suffix == ".sh" else 0o644) << 16
            z.writestr(info, f.read_bytes().replace(b"\r\n", b"\n"))
        for rel in SMALL_DATA + DATA_CIC18:
            z.write(ROOT / rel, f"nids_cloud/{rel}", compress_type=zipfile.ZIP_STORED)
        names = z.namelist()
    print(f"{out.name}: {len(names)} 個檔案, {out.stat().st_size / 1e6:.1f} MB")


def main():
    if "--cloud" in sys.argv:
        return cloud()
    code_only = "--code" in sys.argv
    cic18 = "--cic18" in sys.argv
    for name in CODE:
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, PKG / name)
    out = ROOT / ("mac_v2_code.zip" if code_only else "mac_cic18.zip" if cic18 else "mac_v2.zip")
    data = [] if code_only else DATA_CIC18 if cic18 else DATA
    with zipfile.ZipFile(out, "w") as z:
        for name in CODE + SCRIPTS + [p.name for p in PKG.glob("*.sh") if p.name not in SCRIPTS]:
            f = PKG / name
            if not f.exists():
                continue
            info = zipfile.ZipInfo.from_file(f, f"mac_package/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if f.suffix == ".sh" else 0o644) << 16
            z.writestr(info, f.read_bytes().replace(b"\r\n", b"\n"))
        for rel in SMALL_DATA:
            z.write(ROOT / rel, f"mac_package/{rel}", compress_type=zipfile.ZIP_DEFLATED)
        for rel in data:
            z.write(ROOT / rel, f"mac_package/{rel}", compress_type=zipfile.ZIP_STORED)
        names = z.namelist()
    print(f"{out.name}: {len(names)} 個檔案, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
