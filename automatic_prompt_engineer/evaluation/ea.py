import numpy as np
import random
import random
import string
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np

from automatic_prompt_engineer import evaluate
def ea_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config):
    base_eval_method = evaluate.get_eval_method(config['base_eval_method'])
    ea_algo = get_ea_algo(config['ea_method'], len(prompts), config)
    rounds = config['rounds']
    if config['num_prompts_per_round'] < 1:
        num_prompts_per_round = int(
            len(prompts) * config['num_prompts_per_round'])
    else:
        num_prompts_per_round = config['num_prompts_per_round']
    num_prompts_per_round = min(num_prompts_per_round, len(prompts))
    for _ in tqdm(range(rounds), desc='Evaluating prompts'):
        # Sample the prompts using EA selection strategy
        sampled_prompts_idx = ea_algo.choose(num_prompts_per_round)
        print(len(sampled_prompts_idx),num_prompts_per_round,len(prompts),'!!!!!')

        sampled_prompts = [prompts[i] for i in sampled_prompts_idx]

        # Evaluate the sampled prompts
        # sampled_eval_results = base_eval_method( todo
        #     sampled_prompts, eval_template, eval_data, demos_template, few_shot_data, config['base_eval_config'])
        # _, scores = sampled_eval_results.in_place(method='mean')
        # Update the EA selection strategy
        #crossover

        for i in range(5):
            parent1,parent2= tournament_selection(prompts, ea_algo.scores, 2)
            child1,child2=crossover_prompts(prompts[parent1], prompts[parent2], crossover_rate=0.5)
            print('child',child1,child2)
            child1=mutate_prompt(child1)
            child2=mutate_prompt(child2)
            print('mutate',child1,child2)
            prompts.append(child1)
            prompts.append(child2)
            prompts.pop(random.randint(0,len(prompts)-1))
            prompts.pop(random.randint(0, len(prompts) - 1))
            np.append(ea_algo.scores,0)
            np.append(ea_algo.scores,0)

        # ea_algo.update(sampled_prompts_idx, scores) todo
    return EAEvaluationResult(prompts, ea_algo.get_scores(), ea_algo.get_infos())
def evolution(prompts, config_dic):#todo
    ea_algo = get_ea_algo(config_dic['ea_method'], len(prompts), config_dic)
    rounds = config_dic['rounds']

    for i in range(rounds):
        parent1,parent2= tournament_selection(prompts, ea_algo.scores, 2)
        child1,child2=crossover_prompts(prompts[parent1], prompts[parent2], crossover_rate=0.5)
        print('child',child1,child2)
        child1=mutate_prompt(child1)
        child2=mutate_prompt(child2)
        print('mutate',child1,child2)
        prompts.append(child1)
        prompts.append(child2)
        prompts.pop(random.randint(0,len(prompts)-1))
        prompts.pop(random.randint(0, len(prompts) - 1))
        np.append(ea_algo.scores,0)
        np.append(ea_algo.scores,0)

        # ea_algo.update(sampled_prompts_idx, scores) todo
    return EAEvaluationResult(prompts, ea_algo.get_scores(), ea_algo.get_infos())
def get_ea_algo(ea_method, num_prompts, config):
    if ea_method == 'tournament':
        pop_size = config['pop_size']
        selection_method = tournament_selection
        ea_algo = EAAlgo(pop_size, num_prompts, selection_method)
        ea_algo.initialize()
        return ea_algo
    else:
        raise ValueError(f'Unknown EA method: {ea_method}')


class EAEvaluationResult(evaluate.EvaluationResult):

    def __init__(self, prompts, scores, infos):
        self.prompts = prompts
        self.scores = scores
        self.infos = infos

    def get_best_prompts(self, num_prompts):
        best_indices = np.argsort(self.scores)[-num_prompts:]
        best_prompts = [self.prompts[i] for i in best_indices]
        best_scores = [self.scores[i] for i in best_indices]
        return best_prompts, best_scores

    def to_dict(self):
        return {
            'prompts': self.prompts,
            'scores': self.scores.tolist(),
            'infos': self.infos
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data['prompts'], np.array(data['scores']), data['infos'])

    def sorted(self, method='default'):
        """Sort the prompts and scores. There is no choice of method for now."""
        idx = np.argsort(self.scores)
        prompts, scores = [self.prompts[i]
                           for i in idx], [self.scores[i] for i in idx]
        # Reverse
        prompts, scores = prompts[::-1], scores[::-1]
        return prompts, scores

    def in_place(self, method='default'):
        """Return the prompts and scores in place. There is no choice of method for now."""
        return self.prompts, self.scores


class EAAlgo:
    def __init__(self, pop_size, num_prompts, selection_method):
        self.pop_size = pop_size
        self.num_prompts = num_prompts
        self.selection_method = selection_method
        self.population = None
        self.scores = None
        self.infos = {}
    def append(self,a):
        self.pop_size+=1
        self.population.append(a)
        # self.scores.append(0)
    def initialize(self):
        self.population = [self.generate_individual() for _ in range(self.pop_size)]
        self.scores = np.zeros(self.pop_size)

    def generate_individual(self):
        return [random.uniform(0, 1) for _ in range(self.num_prompts)]

    def choose(self, n):
        # print(self.population,self.scores,num_prompts)
        # selected_indices = self.selection_method(self.population, self.scores, num_prompts)
        # return selected_indices
        # choose randomly.
        return random.sample(range(self.num_prompts), n)


    def update(self, indices, scores):
        self.scores[indices] = scores
        # update any additional EA-specific info here, if needed

    def get_scores(self):
        return self.scores

    def get_infos(self):
        return self.infos

#
# def tournament_selection(population, scores, num_prompts):
#     tournament_size = 2
#     selected_indices = []
#     for _ in range(num_prompts):
#         tournament_indices = random.sample(range(len(population)), tournament_size)
#         print('?',len(scores),len(population),tournament_indices)
#         tournament_scores = scores[tournament_indices]
#         best_index = tournament_indices[np.argmax(tournament_scores)]
#         selected_indices.append(best_index)
#     return selected_indices
def tournament_selection(population, scores, num_prompts):
    tournament_size = 2
    selected_indices = []
    for _ in range(num_prompts):
        tournament_indices = random.sample(range(len(scores)), tournament_size)  # Use len(scores) instead of len(population)
        tournament_scores = scores[tournament_indices]
        best_index = tournament_indices[np.argmax(tournament_scores)]
        selected_indices.append(best_index)
    return selected_indices


def crossover_prompts(parent1, parent2, crossover_rate=0.5):
    """Crossover two prompts at a randomly chosen point."""
    if random.random() > crossover_rate:
        # Return parents unchanged if crossover is not applied
        return parent1, parent2

    # Choose a random split point
    split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)

    # Create child prompts by combining the parents at the split point
    child1 = parent1[:split_point] + parent2[split_point:]
    child2 = parent2[:split_point] + parent1[split_point:]

    return child1, child2


def mutate_prompt(prompt, mutation_rate=0.005):
    """Mutate a prompt by randomly changing  its characters."""
    mutated = list(prompt)
    for i in range(len(mutated)):

        if random.random() < mutation_rate:
            mutated[i] = random.choice(list(string.ascii_letters + string.punctuation))
    return ''.join(mutated)
