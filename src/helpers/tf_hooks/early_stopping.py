import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook

class EarlyStoppingLossHook(tf.train.SessionRunHook):
    def __init__(self, loss_tensor_name, value, threshold=3):
        '''
        A train hook to stop the training at specified train loss
        Usage:
        loss_monitor = EarlyStoppingLossHook("reduced_mean:0", 0.35, 3)
        estimator.train(input_fn=train_input_fn, hooks=[loss_monitor], ...)

        :param loss_tensor_name: Name of the loss tensor eg: loss:0
        :param value: Value at which the trianing should stop
        :param threshold: number of times to check for the loss value, before stopping the training
        '''
        self._best_loss = value
        self.threshold = threshold
        self.count  = 0
        self.loss_tensor_name = loss_tensor_name
        logging.info("Create EarlyStoppingLossHook for {}".format(self.loss_tensor_name))

    def before_run(self, run_context):
        graph = run_context.session.graph
        tensor_name = self.loss_tensor_name
        loss_tensor_name = graph.get_tensor_by_name(tensor_name)
        return session_run_hook.SessionRunArgs(loss_tensor_name)

    def after_run(self, run_context, run_values):
        last_loss = run_values.results

        if last_loss <= self._best_loss:
            self.count += 1
            if self.count == self.threshold:
                logging.info("EarlyStoppingHook: Request early stop")
                run_context.request_stop()





