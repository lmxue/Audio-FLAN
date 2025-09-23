# Audio captioning evaluation metrics

This repository contains code to evaluate translation metrics on audio captioning predictions.

The code from [the Microsoft COCO caption evaluation repository](https://github.com/tylin/coco-caption), in the folder
coco_caption, is used to evaluate the metrics. The code has been refactored to work with Python 3 and to also
evaluate the [SPIDEr metric](https://arxiv.org/abs/1612.00370). Image-specific names and comments in-code were also
changed to be audio-specific.

**Before evaluating metrics, the user must run `coco_caption/get_stanford_models.sh` (or follow the commands in the
bash script) to download the libraries necessary for evaluating the SPICE metric.**

SPICE evaluation uses 8GB of RAM and METEOR uses 2GB (both use Java). To limit RAM usage go to
`coco_caption/pycocoevalcap` and `meteor/meteor.py:18` or `spice/spice.py:63` respectively and change the third argument
of the java command.

The `evaluate_metrics()` function inside `eval_metrics.py` takes as inputs the csv file with the predicted captions and
the csv file with the reference captions. The optional parameter `nb_reference_captions` determines how many
reference captions are used to evaluate the metrics (5 by default).

The input files can be given either as file
paths (string or `pathlib.Path`) or lists of dicts with a dict for each row, the dicts having the column headers as keys
(as given by `csv.DictReader` in Python). The prediction file must have the fields `file_name` and `caption_predicted` and the
reference file must have the fields `file_name` and `caption_reference_XX` with XX being the two-digit index of the
caption, e.g. `caption_reference_01`,...,`caption_reference_05` with five reference captions. 

The metric evaluation function outputs the evaluated metrics in a dict with the lower case metric names
as keys. One score is evaluated for each audio file and its predicted caption. Additionally, a single score is
evaluated for the whole dataset. The format of the output is the following:

    {<metric name in lower case>: {
        'score': <single score value>,
        'scores': {
            <audio file name>: <per-file score value>
        }
    }}
    
This code is maintained by [lippings](https://github.com/lippings)

