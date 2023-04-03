# First, let's define a simple dataset consisting of words and their antonyms.
words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential",
    "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality",
    "deliberately", "off"]
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential",
    "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality",
    "accidentally", "on"]

#%%

# Now, we need to define the format of the prompt that we are using.

eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

#%%

# Now, let's use APE to find prompts that generate antonyms for each word.
from automatic_prompt_engineer import ape

result, demo_fn,prompts = ape.eape(
    dataset=(words, antonyms),
    eval_template=eval_template,
)

print(result)
print(prompts)

#%% md



#%%

# from automatic_prompt_engineer import ape
#
# manual_prompt = "Write an antonym to the following word."
#
# human_result = ape.simple_eval(
#     dataset=(words, antonyms),
#     eval_template=eval_template,
#     prompts=[manual_prompt],
# )
#
# print(human_result)
