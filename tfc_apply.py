from tfci import compress, decompress
import glob

test_img_folder = 'images/4kDS/*'

for path in glob.glob(test_img_folder):
    compress("bmshj2018-hyperprior-msssim-8", path, path.replace("4kDS", "tfc_comp_bm").replace("png","tfci"))
    decompress(path.replace("4kDS","tfc_comp_bm").replace("png","tfci"), path.replace("4kDS","tfc_decomp_bm"))
