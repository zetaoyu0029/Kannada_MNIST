import tf_train_model_ASSIGNMENT_FILE as tm
import tf_train_model_ASSIGNMENT_FILE_extra as tme        
import tf_evaluate_model_code as emc

# whether it's extra credit
isExtra = False

model = 8

if (isExtra):
	model_train = tme.build_train()                 ### comment line out for evaluation only
	model_train.build_train_network(model)          ### comment line out for evaluation only
else:
	model_train = tm.build_train()                  ### comment line out for evaluation only
	model_train.build_train_network(model)          ### comment line out for evaluation only
