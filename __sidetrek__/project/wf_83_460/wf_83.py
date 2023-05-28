import sklearn
import os
import typing
from flytekit import task, workflow, Resources
import sidetrek
from project.wf_83_460.main import Hyperparameters
from project.wf_83_460.main import create_df
from project.wf_83_460.main import clean_ds
from project.wf_83_460.main import handle_cat_cols
from project.wf_83_460.main import split_train_test
from project.wf_83_460.main import train_model

@task(requests=Resources(cpu="2",mem="1Gi"),limits=Resources(cpu="2",mem="1Gi"),retries=3)
def dataset_test_org_hr_data()->sidetrek.types.dataset.SidetrekDataset:
	return sidetrek.dataset.build_dataset(io="upload",source="s3://sidetrek-datasets/test-org/hr-data")



_wf_outputs=typing.NamedTuple("WfOutputs",train_model_0=sklearn.ensemble._gb.GradientBoostingClassifier)
@workflow
def wf_83(_wf_args:Hyperparameters)->_wf_outputs:
	dataset_test_org_hr_data_o0_=dataset_test_org_hr_data()
	create_df_o0_=create_df(ds=dataset_test_org_hr_data_o0_)
	clean_ds_o0_=clean_ds(df=create_df_o0_)
	handle_cat_cols_o0_=handle_cat_cols(df=clean_ds_o0_)
	split_train_test_o0_,split_train_test_o1_,split_train_test_o2_,split_train_test_o3_=split_train_test(df=handle_cat_cols_o0_,hp=_wf_args)
	train_model_o0_=train_model(X_train=split_train_test_o0_,y_train=split_train_test_o2_)
	return _wf_outputs(train_model_o0_)