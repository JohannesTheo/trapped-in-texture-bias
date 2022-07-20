"""
This script downloads model weights defined in https://github.com/HobbitLong/PyContrast/tree/master/pycontrast/detection
"""

import sys
sys.path.append("../../datasets/")
from utils import download
from pathlib import Path

MODELS = {
    #"R_50_FPN_Random_1x":     "https://www.dropbox.com/s/szuzbuxmbglnnrj/model_final.pth?dl=1",
    #"R_50_FPN_Random_2x":     "https://www.dropbox.com/s/s1be6i6aosvcgxp/model_final.pth?dl=1",
    "R_50_FPN_Random_6x":     "https://www.dropbox.com/s/3t0ez4d5z06ks6y/model_final.pth?dl=1",
    #"R_50_FPN_Supervised_1x": "https://www.dropbox.com/s/vunziha5hd11qki/model_final.pth?dl=1",
    #"R_50_FPN_Supervised_2x": "https://www.dropbox.com/s/6ekidu2z19fhe1v/model_final.pth?dl=1",
    "R_50_FPN_Supervised_6x": "https://www.dropbox.com/s/6hotgcfrbluk8a4/model_final.pth?dl=1",
    #"R_50_FPN_InstDis_1x":    "https://www.dropbox.com/s/bc131tpru4iw7n1/model_final.pth?dl=1",
    #"R_50_FPN_InstDis_2x":    "https://www.dropbox.com/s/m9tsvdxx9h5qa5a/model_final.pth?dl=1",
    #"R_50_FPN_PIRL_1x":       "https://www.dropbox.com/s/li01ay7j9z7cwj3/model_final.pth?dl=1",
    #"R_50_FPN_PIRL_2x":       "https://www.dropbox.com/s/riiqvsnx1qk0ec4/model_final.pth?dl=1",
    #"R_50_FPN_MoCo_v1_1x":    "https://www.dropbox.com/s/ljs130qyj9w3zhs/model_final.pth?dl=1",
    #"R_50_FPN_MoCo_v1_2x":    "https://www.dropbox.com/s/eofvp6t9e17b9wq/model_final.pth?dl=1",
    #"R_50_FPN_MoCo_v2_2x":    "https://www.dropbox.com/s/hcoastwec4al1nq/model_final.pth?dl=1",
    #"R_50_FPN_InfoMin_1x":    "https://www.dropbox.com/s/trn58ixmgb4ecnh/model_final.pth?dl=1",    
    #"R_50_FPN_InfoMin_2x":    "https://www.dropbox.com/s/oh0ke3sdil40kge/model_final.pth?dl=1",
    "R_50_FPN_InfoMin_6x":    "https://www.dropbox.com/s/7xrlpd3klx0hwnj/model_final.pth?dl=1"
}

# setup path variables
ROOT = (Path(__file__).parent / "../../..").resolve()
WEIGHTS_DIR  = ROOT / "ext" / "PyContrast/pycontrast/detection/"

for model, url in MODELS.items():
    out_file = WEIGHTS_DIR / f"{model}.pth"
    if not out_file.exists(): download(url, str(out_file))

print(f"Download Contrastive Model Weights: ✓\n")
