# sisap2023
A code base to perform experiments on conjunctive queries.

The code for the paper "Querying with Multiple Objects" is in this repo.

To download the data:
    Mirflkr1M dataset may be downloaded from here: https://press.liacs.nl/mirflickr/dlform.html

Encodings supported: Alexnet, Alexnet_softmax, Alexnet_fc6, Resnet50/18, Resnet50/18_softmax, Dino2

To encode the images:

TODO need info from David

To run the experiments:

1. All experiments assume data is downloaded and encoded locally as described above.

2.  Quantative results shown in Figure \label{fig_cum_sums} calculated with experiments/experimentselected.py.
    This uses selected_queries.txt and imagenet_classes.txt

    Selected queries created used notebooks/pick_queries.ipynb
    In most cases the first query is used, where this is not the case it is documented in this notebook.
    notebooks/pick_queries.ipynb also creates the imagenet_classes.txt file if it does not exist.

    This file was run as follows:
    python -W ignore simcoder experiment /data/mirflickr/mf_dino2 /data/mirflickr/mf_alexnet_softmax/ /data/mirflickr/results/dino2_20/ 100 100 0 0.9
    The parameters documented in the code:
        0: encodings: the encodings directory
        1: softmax: the softmax encodings directory
        2: output_path: the results directory
        3: number_of_categories_to_test
        4: k: which k to use for the NN@k experiments
        5: initial_query_index whether to use the zeroth, first, second, third best image during the search
        6: thresh: the threshold for doing measurement of success

3.  The Siamese cats shown in Figure \label{fig_siamese_cats} and Figure \label{fig_mixed_cats} are created using notebooks/Siamese_cats.ipynb

4.  The bottles and glasses shown in Figure \label{fig_beer_bottles} , \label{fig_beer_glasses}, \label{fig_bottles_and_glasses} are created with notebooks/bottles_glasses.ipynb

5.  The albatros pictures shown in \label{fig_albatross_poly} are created using run_any_nn100.ipynb with param 6 (query = queries[6])
    This notebook also contains:
        a. the results from all the 'good' runs of this notebook.
        b. the index,query,category and category string	of the queries used.





