from tfci import compress, decompress
import glob

test_img_folder = 'images/LR/*'
for path in glob.glob(test_img_folder):
    compress('hific-hi', path, path.replace("LR", "tfc_comp_hi").replace("png","tfci"))
    decompress(path.replace("LR","tfc_comp_hi").replace("png","tfci"), path.replace("LR","tfc_decomp_hi"))
