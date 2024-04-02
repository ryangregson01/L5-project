import sys
from pipeline import run_pipeline


model_map = {'l27b-meta': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'], # 2048 max tokens doc
            'l213b-8bit': ['get_l2', 'TheBloke/Llama-2-13B-chat-GPTQ', 'gptq-8bit-64g-actorder_True'], 
            'l270b-4bit': ['get_l2', 'TheBloke/Llama-2-70B-chat-GPTQ', 'main'],
            'mist7b-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mixt-4bit': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            'test-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'flan-large': ['get_flan', 'google/flan-t5-large', 'main'], # 320 max tokens doc
            'l2-nochat': ['get_l2', 'meta-llama/Llama-2-7b-hf', 'main'],
            'mist-awq': ['get_model', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ', 'main'],
            'l2bnb': ['get_l2_bits', 'meta-llama/Llama-2-13b-hf', 'main'],
            'mist-noreply': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'l27b-noreply': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'],
            'mixt-noreply': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            'flanxl-noreply': ['get_flan', 'google/flan-t5-xl', 'main']
            }

model_name = sys.argv[1]
model_list = model_map.get(model_name)
load_func = model_list[0]
model_path = model_list[1]
revision = model_list[2]
if len(sys.argv) > 2:
    device = sys.argv[2]
else:
    device = 'auto'

#print(load_func, model_path, revision, device)
#prompts = ['base_sens', 'base2_sens', 'context_b1_sens', 'context_b2_sens', 'base_personal', 'context_b1_personal', 'fixed_fewshot_personal', 'base_personal_explanation', 'purely_personal', 'itspersonalgenres', 'multi_category_noanseng', 'multi_category', 'base_classify', 'barlit', 'barlit2']
prompts = ['barlit', 'barlit2']
prompts = ['text', 'pdc', 'cg', 'textqa', 'pdcqa', 'cgqa']
prompts = ['text', 'pdc', 'cg', 'pdc2', 'textfew', 'pdcfew', 'cgfew', 'textqa', 'pdcqa', 'cgqa']
#prompts = ['pdc2']
#prompts = ['pdcfewsim']
end_prompt = '[/INST]'
sample_size = 2

print('Starting experiment:', model_name)
print(prompts)
run_pipeline(model_name, load_func, model_path, revision, device, prompts, end_prompt) #, sample_size)
print('Finished experiment:', model_name)
