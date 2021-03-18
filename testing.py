import stitching
import random

imgs = [1, 2, 3, 4, 5, 6]
order = [1, 2, 3, 4, 5, 6]

# for i in range(5):
#     random.shuffle(imgs)
#     final_order = stitching.main(path1="./img" + str(imgs[0]) + ".png", \
#                                 path2="./img" + str(imgs[1]) + ".png", \
#                                 path3="./img" + str(imgs[2]) + ".png", \
#                                 path4="./img" + str(imgs[3]) + ".png", \
#                                 path5="./img" + str(imgs[4]) + ".png", \
#                                 path6="./img" + str(imgs[5]) + ".png", \
#                                 feature_extraction_method="sift", matcher_method="bruteforce", k_val=2, ratio=0.75)
#     correct_order = [imgs.index(x) for x in order]
#     print(final_order == correct_order)


stitching.main(path1="./img" + str(imgs[2]) + ".png", \
                                path2="./img" + str(imgs[1]) + ".png", \
                                path3="./img" + str(imgs[0]) + ".png", \
                                path4="./img" + str(imgs[3]) + ".png", \
                                path5="./img" + str(imgs[4]) + ".png", \
                                path6="./img" + str(imgs[5]) + ".png", \
                                feature_extraction_method="sift", matcher_method="bruteforce", k_val=2, ratio=0.75)