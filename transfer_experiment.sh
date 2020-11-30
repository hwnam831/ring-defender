python RingClassifier.py --net ff --testnet cnn > exp/ff_cnn.log
python RingClassifier.py --net cnn --testnet cnn > exp/cnn_cnn.log
python RingClassifier.py --net rnn --testnet cnn > exp/rnn_cnn.log

python RingClassifier.py --net ff --testnet ff > exp/ff_ff.log
python RingClassifier.py --net cnn --testnet ff > exp/cnn_ff.log
python RingClassifier.py --net rnn --testnet ff > exp/rnn_ff.log

python RingClassifier.py --net ff --testnet rnn > exp/ff_rnn.log
python RingClassifier.py --net cnn --testnet rnn > exp/cnn_rnn.log
python RingClassifier.py --net rnn --testnet rnn > exp/rnn_rnn.log


