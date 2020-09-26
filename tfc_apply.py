from tfci import compress, decompress
import glob

test_img_folder = 'images/4kDS/*'

for path in glob.glob(test_img_folder):
    compress("mbt2018-mean-msssim-8", path, path.replace("4kDS", "tfc_comp_mbt").replace("png","tfci"))
    decompress(path.replace("4kDS","tfc_comp_mbt").replace("png","tfci"), path.replace("4kDS","tfc_decomp_mbt"))
