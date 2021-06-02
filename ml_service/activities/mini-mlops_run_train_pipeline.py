from os import name
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Experiment, Workspace, experiment
import argparse
from ml_service.util.env_variables import Env

def main(): 
    print('in main')
    
    # parsing information
    parser = argparse.ArgumentParser('register')
    parser.add_argument('--output_pipeline_id_file', type=str, default='pipeline_id.txt', help='File for pipeline')
    parser.add_argument('--skip_train_execution', action='store_true', help='do not trigger the execution')
    args = parser.parse_args()

    e = Env()

    aml_workspace = Workspace.get(
        name=e.workspace_name,
        subscription_id=e.subscription_id,
        resource_group=e.resource_group
    )

    pipelines = PublishedPipeline.list(aml_workspace)
    matched_pipes = []

    for p in pipelines: 
        if p.name == e.pipeline_name: 
            # Version issue is not solved
            # if p.version == e.build_id: 
            matched_pipes.append(p)
            # print('Found pipeline: ', p)
    

    # Because of debugging, there are many pipelines, and so I'm commenting this part out: 
    # if(len(matched_pipes) > 1):
    #     published_pipeline = None
    #     raise Exception(f"Multiple active pipelines are published for build {e.build_id}.")  # NOQA: E501
    
    if(len(matched_pipes) == 0):
        published_pipeline = None
        raise KeyError(f"Unable to find a published pipeline for this build {e.build_id}")  # NOQA: E501
    else:
        published_pipeline = matched_pipes[0]
        print("published pipeline id is", published_pipeline.id)

        # Save the Pipeline ID for other AzDO jobs after script is complete
        if args.output_pipeline_id_file is not None:
            with open(args.output_pipeline_id_file, "w") as out_file:
                out_file.write(published_pipeline.id)

        
        # Also exluded
        # So the run is included here, but it is split into another step, where the pipeline
        # in the yml file is running it directly as tasks from DevOps.  

        # pipeline_parameters = {'model_name': e.model_name}

        # tags = {'BuildID': e.build_id}
        # if (e.build_uri is not None):
        #     tags['BuildUri'] = e.build_uri
        
        # experiment = Experiment(workspace=aml_workspace, name=e.experiment_name)

        # run = experiment.submit(published_pipeline, tags=tags, pipeline_parameters=pipeline_parameters)

if __name__ == '__main__':
    main()