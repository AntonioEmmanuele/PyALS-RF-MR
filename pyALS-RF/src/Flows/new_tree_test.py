import logging
from ..ModelTreeVisit.DecisionTree import Decision_Tree_Classifier
from ..ModelTreeVisit.RandomForest import Random_Forest_Classifier
from ..Model import Classifier
from ..ctx_factory import load_configuration_ps, create_classifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import time
from sklearn.metrics import accuracy_score
def create_tree_classifier(ctx):
    
    model_path = ctx.obj["configuration"].model_source
    model_path = model_path[:-5] + ".joblib"
    assert os.path.exists(model_path)
    sklearn_mode = joblib.load(model_path)
    new_model = Random_Forest_Classifier.convert_sklearn_model(sklearn_mode)
    new_model.import_pyals_dataset_description(ctx.obj["configuration"].error_conf.dataset_description)
    new_model.read_test_set(ctx.obj["configuration"].error_conf.test_dataset, ctx.obj["configuration"].error_conf.dataset_description.separated_training)
    new_model.read_training_set(ctx.obj["configuration"].train_dataset, ctx.obj["configuration"].error_conf.dataset_description.separated_training)
    return new_model

def test_decision_tree(ctx, conf, quantization_type, ncpus):
    load_configuration_ps(ctx)
    new_model = create_tree_classifier(ctx)
    logger = logging.getLogger("pyALS-RF")
    logger.info(f"Classifier correctly generated")
    logger.info(f"New Model Estimators: {new_model.n_estimators}")
    
    assert len(new_model.x_train) == len(new_model.y_train)
    assert len(new_model.x_test) == len(new_model.y_test)
    logger.info(f"Train and Test set consistently imported. Train {len(new_model.x_train)} Test {len(new_model.x_test)}")
    tm = time.time()
    for x in new_model.x_test:
        labels = new_model.estimators[0].predict_no_batch(x)
    tm = time.time() - tm
    logger.info(f"Single tree Inference completed in {tm}")
    tm = time.time()
    labels = new_model.estimators[0].predict(new_model.x_test)
    tm = time.time() - tm
    
    logger.info(f"Batch inference completed in {tm}")

    logger.info(f"Acc {accuracy_score(labels, new_model.y_test)}")
    
    
