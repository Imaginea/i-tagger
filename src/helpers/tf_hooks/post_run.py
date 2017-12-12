import tensorflow as tf

class PostRunTaskHook(tf.train.SessionRunHook):
    """Hook to initialise utils iterator after Session is created."""

    def __init__(self):
        super(PostRunTaskHook, self).__init__()
        self.user_func = None

    def end(self, session):
        self.user_func(session)