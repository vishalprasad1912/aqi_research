python -m src.training.train --model lstm --station pusa --dropout 0.1
python -m src.training.train --model gcn_lstm --station pusa --dropout 0.1
python -m src.training.train --model gat_lstm --station pusa --dropout 0.1
python -m src.training.train --model lstm --station wazirpur --dropout 0.1
python -m src.training.train --model gcn_lstm --station wazirpur --dropout 0.1
python -m src.training.train --model gat_lstm --station wazirpur --dropout 0.1