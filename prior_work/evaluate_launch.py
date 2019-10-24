import tf_train_model_ASSIGNMENT_FILE as tm
import tf_evaluate_model_code as emc

model = 8

model_eval = emc.evaluate_model()
model_eval.evaluate_model(model_version= model)