echo $1
python RingClassifier.py --gen $1 --amp 2 > exp/$1_2.log
python RingClassifier.py --gen $1 --amp 4 > exp/$1_4.log
python RingClassifier.py --gen $1 --amp 6 > exp/$1_6.log
python RingClassifier.py --gen $1 --amp 8 > exp/$1_8.log
python RingClassifier.py --gen $1 --amp 10 > exp/$1_10.log
python RingClassifier.py --gen $1 --amp 12 > exp/$1_12.log