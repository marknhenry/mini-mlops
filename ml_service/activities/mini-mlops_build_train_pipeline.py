# still a little way to go
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core import Workspace, Dataset, Datastore
from azureml.core.runconfig import RunConfiguration
from msrest import pipeline
# from ml_service.pipelines.load_sample_data import create_sample_data_csv
from ml_service.util.attach_compute import get_compute
from ml_service.util.env_variables import Env
from ml_service.util.manage_environment import get_environment
import os

def prep_platform(debug=False): 
    e = Env()
   
    # Get Azure machine learning workspace
    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group,
    )
    if debug: print("\nWorkspace Details: ",aml_workspace, '\n')

    # Get Azure machine learning cluster
    aml_compute = get_compute(aml_workspace, e.compute_name, e.vm_size)
    if aml_compute is not None:
        if debug: print("\nAML Compute Target: ", aml_compute)

    # Create a reusable Azure ML environment
    environment = get_environment(
        aml_workspace,
        e.aml_env_name,
        conda_dependencies_file=e.aml_env_train_conda_dep_file,
        create_new=e.rebuild_env,
    )

    # Create a new Run Configuration
    run_config = RunConfiguration()
    run_config.environment = environment

    # Configure Datastore
    if e.datastore_name:
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    
    run_config.environment.environment_variables[
        "DATASTORE_NAME"
    ] = datastore_name  # NOQA: E501

    print('\nPlatform Ready')
    return e, aml_workspace, aml_compute, environment, run_config, datastore_name

def assert_dataset_exists(e, aml_workspace, datastore_name):
    

    if dataset_name not in aml_workspace.datasets:
        # This call creates an example CSV from sklearn sample data. If you
        # have already bootstrapped your project, you can comment this line
        # out and use your own CSV.
        # create_sample_data_csv()
        file_name = 'diabetes.csv'
        # Use a CSV to read in the data set.file_name = "diabetes.csv"
        if not os.path.exists(file_name):
            raise Exception(
                'Could not find CSV dataset at "%s". If you have bootstrapped your project, you will need to provide a CSV.'  # NOQA: E501
                % file_name
            )
        
        # # Upload file to default datastore in workspace
        datatstore = Datastore.get(aml_workspace, datastore_name)
        target_path = "training-data/"
        datatstore.upload_files(
            files=[file_name],
            target_path=target_path,
            overwrite=True,
            show_progress=False
        )
        # Register dataset
        path_on_datastore = os.path.join(target_path, file_name)
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datatstore, path_on_datastore)
        )
        dataset = dataset.register(
            workspace=aml_workspace,
            name=dataset_name,
            description="diabetes training data",
            tags={"format": "CSV"},
            create_new_version=True,
        )

def main():
    print('\nStarting Script')
    
    # Setup Platform
    e, aml_workspace, aml_compute, environment, run_config, datastore_name = prep_platform()
    
    # Get dataset name
    dataset_name = e.dataset_name

    # Ignore for now, but check how that would work for the customer from ADF
    # assert_dataset_exists(e, aml_workspace, datastore_name)

    # Pipeline Parameters
    model_name_param = PipelineParameter(name="model_name", default_value=e.model_name)
    dataset_version_param = PipelineParameter(name="dataset_version", default_value=e.dataset_version)
    data_file_path_param = PipelineParameter(name="data_file_path", default_value="none")
    caller_run_id_param = PipelineParameter(name="caller_run_id", default_value="none")

    # Create a PipelineData object to pass data between steps
    pipeline_data = PipelineData("pipeline_data", datastore=aml_workspace.get_default_datastore())

    train_step = PythonScriptStep(
        name='Train Model'
        , script_name= e.train_script_path
        , compute_target=aml_compute
        , source_directory=e.sources_directory_train
        , outputs=[pipeline_data]
        , runconfig=run_config
        , allow_reuse=True
        , arguments=[
            '--model_name', model_name_param
            , '--step_output', pipeline_data
            , '--dataset_version', dataset_version_param
            , '--data_file_path', data_file_path_param
            , '--caller_run_id', caller_run_id_param
            , '--dataset_name', dataset_name
        ]
    )

    print('Training Step Created')

    evaluate_step = PythonScriptStep(
        name='Evaluate Model'
        , script_name= e.evaluate_script_path
        , compute_target=aml_compute
        , source_directory=e.sources_directory_train
        , runconfig=run_config
        , allow_reuse=True
        , arguments=[
            '--model_name', model_name_param
            , '--allow_run_cancel', e.allow_run_cancel
        ]
    )

    print('Evaluation Step Created')

    register_step = PythonScriptStep(
         name='Register Model'
        , script_name= e.register_script_path
        , compute_target=aml_compute
        , source_directory=e.sources_directory_train
        , inputs=[pipeline_data]
        , runconfig=run_config
        , allow_reuse=True
        , arguments=[
            '--model_name', model_name_param
            , '--step_input', pipeline_data
        ]
    )

    print('Register Step created')

    # Create Pipeline
    evaluate_step.run_after(train_step)
    register_step.run_after(evaluate_step)
    steps = [train_step, evaluate_step, register_step]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline._set_experiment_name
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=e.pipeline_name,
        description="Model training/retraining pipeline",
        version=e.build_id
        )
    
    print('\nEnd of Script')


if __name__ == '__main__':
    main()