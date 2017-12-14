import sys
sys.path.append("src/")

from data_iterators.patent_data_iterator import PatentIDataIterator
from models.model_factory import TFEstimatorFactory
from helpers.print_helper import *

from preprocessor.patent_data_preprocessor import PatentDataPreprocessor

EXPERIMENT_ROOT_DIR = "experiments/"
MODEL_DIR = "/opt/0.imaginea/git/i-tagger/experiments/bilstm_crf_v0/charembd_True_lr_0.001_lstmsize_2-32-32_wemb_32_cemb_32_outprob_0.5/"
NUM_EPOCHS = 5
BATCH_SIZE = 2

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

estimator_config, estimator = TFEstimatorFactory.get("bilstm_crf_v0")
if MODEL_DIR:
    config = estimator_config.load(MODEL_DIR)
    if config == None: #Fail safe
        estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR, preprocessor.OUT_DIR)
    else:
        estimator_config = config
else:
    estimator_config = estimator_config.with_user_hyperparamaters(EXPERIMENT_ROOT_DIR, preprocessor.OUT_DIR)
estimator = estimator(estimator_config)

data_iterators = PatentIDataIterator(preprocessor.OUT_DIR, batch_size=BATCH_SIZE)
data_iterators.prepare()

num_samples = data_iterators.NUM_TRAINING_SAMPLES

print_info(num_samples)
max_steps = (num_samples // BATCH_SIZE) * NUM_EPOCHS
print_info("Total number steps: {} ".format(max_steps))

for current_epoch in range(NUM_EPOCHS):
    max_steps = (num_samples // BATCH_SIZE) * (current_epoch + 1)

    train_hooks = []
    train_hooks.append(data_iterators.train_data_init_hook)
    # if len(estimator.hooks) > 0:
    #     train_hooks.extend(tagger.hooks)

    estimator.train(input_fn=data_iterators.train_data_input_fn,
             hooks=train_hooks,
             max_steps=max_steps)

    eval_results = estimator.evaluate(input_fn=data_iterators.val_data_input_fn,
                                   hooks=[data_iterators.val_data_init_hook])  # tf_debug.LocalCLIDebugHook()

    print(eval_results)


data_iterators.predict_on_csv_files(estimator=estimator, csv_files_path="data/test/")