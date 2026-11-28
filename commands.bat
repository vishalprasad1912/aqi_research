python -m src.training.train --model gcn_lstm --station pusa
@REM python -m src.training.train --model gcn_lstm --station najafgarh
@REM python -m src.training.train --model gcn_lstm --station vivek_vihar
@REM python -m src.training.train --model gcn_lstm --station dwarka_sector_8
python -m src.training.train --model gat_lstm --station pusa
@REM python -m src.training.train --model gat_lstm --station najafgarh
@REM python -m src.training.train --model gat_lstm --station vivek_vihar
@REM python -m src.training.train --model gat_lstm --station dwarka_sector_8
python -m src.training.train --model lstm --station pusa
@REM python -m src.training.train --model lstm --station najafgarh
@REM python -m src.training.train --model lstm --station vivek_vihar
@REM python -m src.training.train --model lstm --station dwarka_sector_8