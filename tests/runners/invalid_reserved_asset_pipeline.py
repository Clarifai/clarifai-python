from clarifai.runners.pipelines import Pipeline, step


@step(id='collision-step', assets=['./pipeline_step.py'])
def collision_step(input_text: str) -> str:
    return input_text


with Pipeline(id='asset-pipeline', user_id='me', app_id='my-app') as pipeline:
    raw_text = pipeline.input('input_text')
    collision_step(input_text=raw_text)