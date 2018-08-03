class TensorflowTrainTest(object):

	def __init__(self, model, logdir, saveModelSecs=0):

		# Generate supervisor.
		sv = tf.train.Supervisor(graph=model.graph, logdir=logdir, save_model_secs=saveModelSecs)

		