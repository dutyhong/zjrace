#!/usr/bin/sh
mkdir ../data
python src_split.py
python ner_sample_build.py ../data/train_content.csv ../data/train_label.csv ../data/train.ner.v2
python ner_sample_build.py ../data/test_content.csv ../data/test_label.csv ../data/test.ner.v2
python ner_utils.py ../data/train.ner.v2 ../data/train.ner.pl
python ner_utils.py ../data/test.ner.v2 ../data/test.ner.pl

python cla_sample_build.py ../data/train_content.csv ../data/train_label.csv ../data/train.cla.v2
python cla_sample_build.py ../data/test_content.csv ../data/test_label.csv ../data/test.cla.v2
python cla_utils.py ../data/train.cla.v2 ../data/train.cla.pl
python cla_utils.py ../data/test.cla.v2 ../data/test.cla.pl
