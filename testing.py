import stitching

img_num = [1, 2, 3, 4]

for i in img_num:
    for j in img_num: 
        if j != i:
            for k in img_num:
                    if k != j and k != i:
                        for l in img_num:
                            if l != k and l != j and l != i:
                                print("IMAGE_NUMS: ", i, ", ", j, ", ", k, ", ", l )
                                final_order = stitching.main(path1="./img" + str(i) + ".png", path2="./img" + str(j) + ".png", path3="./img" + str(k) + ".png", path4="./img" + str(l) + ".png", feature_extraction_method="sift", matcher_method="bruteforce", k_val=2, ratio=0.75)
                                img_list = [i, j, k, l]
                                correct_order = [img_list.index(x) for x in sorted(img_list)]
                                if final_order == correct_order:
                                    print("CORRECT")
                                else:
                                    print("ERROR")
                                    print("Output: ", final_order)
                                    print("Expected: ", correct_order)




