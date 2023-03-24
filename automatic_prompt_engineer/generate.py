from faulthandler import disable
from automatic_prompt_engineer import data, llm
from automatic_prompt_engineer.template import DemosTemplate, GenerationTemplate
from typing import List


def get_query(prompt_gen_template, demos_template, subsampled_data):
    """
    Returns a query for the prompt generator. A query is the prompt that is sent to the LLM.
    Parameters:
        prompt_gen_template: The template for the prompt generator queries.
        demos_template: The template for the demonstrations.
        subsampled_data: The data to use for the demonstrations.
    Returns:
        A query for the prompt generator.
    """
    inputs, outputs = subsampled_data
    demos = demos_template.fill(subsampled_data)
    return prompt_gen_template.fill(input=inputs[0], output=outputs[0], full_demo=demos)


def generate_prompts(
    prompt_gen_template: GenerationTemplate,
    demos_template: DemosTemplate,
    prompt_gen_data: tuple[List, List],
    config: dict):
    """
    Generates prompts using the prompt generator. 

    Parameters:
        prompt_gen_template (GenerationTemplate): The template for the prompt generator queries.
        demos_template (DemosTemplate): The template for the demonstrations.
        prompt_gen_data (tuple): Tuple of (inputs[List], outputs[List]) for the prompt generator.
        config: The configuration dictionary.
    Returns:
        A list of prompts.
    """
    queries = []
    for _ in range(config['num_subsamples']):
        subsampled_data = data.subsample_data(
            prompt_gen_data, config['num_demos'])
        queries.append(get_query(prompt_gen_template,
                                 demos_template, subsampled_data))

    # Instantiate the LLM
    model = llm.model_from_config(config['model'], disable_tqdm=False)
    prompts = model.generate_text(
        queries, n=config['num_prompts_per_subsample'])
    return prompts
