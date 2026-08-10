"""
install_font.py
===============
Jednorázový skript – stáhne xkcd font a nainstaluje ho do matplotlib.
Spusť jednou:
    python install_font.py

Po instalaci funguje sketch_report.py bez fontových varování.
"""

import os
import shutil
import urllib.request
import matplotlib
import matplotlib.font_manager as fm

FONT_URL   = "https://github.com/ipython/xkcd-font/raw/master/xkcd-script/font/xkcd-script.ttf"
FONT_NAME  = "xkcd-script.ttf"

def install():
    # Cílová složka uvnitř matplotlib (funguje pro venv i system Python)
    mpl_fonts = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    dest = os.path.join(mpl_fonts, FONT_NAME)

    if os.path.exists(dest):
        print(f"Font už je nainstalován: {dest}")
    else:
        print(f"Stahuji font z {FONT_URL} ...")
        try:
            urllib.request.urlretrieve(FONT_URL, dest)
            print(f"Font uložen: {dest}")
        except Exception as e:
            print(f"Stahování selhalo: {e}")
            print("\nManuální instalace:")
            print(f"  1. Stáhni: {FONT_URL}")
            print(f"  2. Zkopíruj do: {mpl_fonts}")
            return False

    # Smaž font cache aby matplotlib font zaregistroval
    cache = matplotlib.get_cachedir()
    for f in os.listdir(cache):
        if f.endswith(".json") or "font" in f.lower():
            try:
                os.remove(os.path.join(cache, f))
            except Exception:
                pass
    print(f"Font cache vyčištěna: {cache}")

    # Obnov seznam fontů
    fm.fontManager.addfont(dest)
    prop = fm.FontProperties(fname=dest)
    print(f"Font registrován jako: '{prop.get_name()}'")
    print("\n✓ Hotovo – spusť sketch_report.py")
    return True

if __name__ == "__main__":
    install()
