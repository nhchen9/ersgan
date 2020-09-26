from tfci import compress, decompress
import glob

test_img_folder = 'images/DS/*'

for path in glob.glob(test_img_folder):
    compress("bmshj2018-factorized-msssim-8", path, path.replace("DS", "tfc_comp_lr").replace("png","tfci"))
    decompress(path.replace("DS","tfc_comp_lr").replace("png","tfci"), path.replace("DS","tfc_decomp_lr"))
