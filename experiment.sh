echo $1
python RingClassifier.py --gen $1 --amp 1 > exp/$1_1.log
python RingClassifier.py --gen $1 --amp 2 > exp/$1_2.log
python RingClassifier.py --gen $1 --amp 3 > exp/$1_3.log
python RingClassifier.py --gen $1 --amp 4 > exp/$1_4.log
python RingClassifier.py --gen $1 --amp 5 > exp/$1_5.log