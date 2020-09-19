from tfci import compress

compress('hific-lo', "./images/LR/0801x4.png", "./images/tfc_comp/0801x4.tfci")
decompress("./images/tfc_comp/0801x4.tfci", "./images/tfc_decomp/0801x4.png")
