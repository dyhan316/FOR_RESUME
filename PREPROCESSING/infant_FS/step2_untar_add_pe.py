from IMP_step2_packages import *

tar_dir = "/scratch/08834/tg881334/2019/3.FS_tar"
untar_dir = "/scratch/08834/tg881334/2019/4.FS_untar"
BIDS_dir = "/scratch/08834/tg881334/2019/2.BIDS"


a = whole_class(tar_dir = tar_dir, untar_dir = untar_dir, BIDS_dir = BIDS_dir)

a.parallel_untar()
a.modify_whole_bids()

