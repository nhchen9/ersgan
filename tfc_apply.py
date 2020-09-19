from tfci import compress, decompress
import glob

test_img_folder = 'images/LR/*'
for path in glob.glob(test_img_folder):
    compress('hific-lo', path, path.replace("LR", "tfc_comp").replace("png","tfci"))
    decompress(path.replace("LR","tfc_comp").replace("png","tfci"), path.replace("LR","tfc_decomp"))
