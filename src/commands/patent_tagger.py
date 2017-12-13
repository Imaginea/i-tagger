import sys

from data_terators.patent_data_iterator import PatentDataIterator
from models.model_factory import TFEstimatorFactory

sys.path.append("src/")

from preprocessor.patent_data_preprocessor import PatentDataPreprocessor

EXPERIMENT_ROOT_DIR = "experiments/"

preprocessor = PatentDataPreprocessor(
    experiment_root_directory=EXPERIMENT_ROOT_DIR,
    over_write=None,
    use_iob=None,
    out_dir=None,
    train_csvs_path=None,
    val_csv_path=None,
    test_csv_path=None,
    db_reference_file=None,
    text_col=None,
    entity_col=None,
    do_run_time_config=False)
preprocessor.start()

estimator_config, estimator = TFEstimatorFactory.get("bilsm_crf_v0")
estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR, preprocessor.OUT_DIR)
estimator = estimator(estimator_config)

data_iterators = PatentDataIterator(preprocessor.OUT_DIR, batch_size=8)
data_iterators.prepare()




