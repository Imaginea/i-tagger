import tensorflow as tf

class PreRunTaskHook(tf.train.SessionRunHook):
    """Hook to initialise utils iterator after Session is created."""

    def __init__(self):
        super(PreRunTaskHook, self).__init__()
        self.user_func = None

    def after_create_session(self, session, coord):
        self.user_func(session)

    def before_run(self, run_context):
        session = run_context.session
        self.user_func(session)

    def end(self, session):
        self.user_func(session)