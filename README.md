## Frame Extraction
1. `v2f.py` turns video into a set of frames, with a default rate of 2fps. <br />
    To run: ```$python v2f.py arg1 arg2 arg3``` <br />
    where arg1 is the path of the source video, arg2 is path for the frames to be stored, and arg3 is the name of the frames
2. `f2v.py` turns a set of frames back to a video, with a default rate of 2fps. <br />
    To run: ```$python f2v.py arg1 arg2``` <br />
    where arg1 is path to the set of frames and arg2 is the path for the generated video
3. `crop.ipynb` crops a ultra-wide video into 8 sub-videos and turn each of them into a set of frames. <br />
    To run: just upload the video and run each cell

## Frame Stitching
Suppose we want to stitch N videos each having T frames. We implemented three different programs to run on GCP dataproc:
1. `stitching.py` performs video stitching on a single server
2. `spark_stitching_n.py` performs distributive video stitching on spark using "distribute-over-N" method. In each Spark job, we stitch N frames distributively (by extrating 1 frame from each video). Then, run T of those jobs sequentially. This method is relatively inefficient.
3. `spark_stitching_t.py` performs distributive video stitching on spark using "distribute-over-T" method. We have a single Spark job that processes T stitching jobs distributively, where each stitching job (of N frames) is done in a single server approach.

The arguments of the programs are:
1. Number of frames that each frame contains (T)
2. Input directory. We assume that the input directory contains T sub-directories (frame1 ... frameT), and each sub-directory contains N frames (img1 ... imgN).
3. Feature extraction method. Suggested method is "sift"
4. Matcher method. Currently we only supports "bruteforce".
5. k_val
6. ratio
7. Output directory

Sample command to run:
```
gcloud dataproc jobs submit pyspark stitching2.py \
--cluster=cluster-af17 \
--region=us-east1 \
-- 180 gs://dataproc-staging-us-west4-399405748907-aq6u7ejv/testInput/frame \
sift bruteforce 2 0.75 gs://dataproc-staging-us-west4-399405748907-aq6u7ejv/testOutput
```

The command above will read the 180 directories testInput/frame1 ... testInput/frame180. It will produce 180 stitched images testOutput/output1.jpg ... testOutput/output180.jpg




