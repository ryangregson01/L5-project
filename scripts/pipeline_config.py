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
            'mixt-noreply': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'], #'gptq-4bit-32g-actorder_True'], #'main'],
            'flanxl-noreply': ['get_flan', 'google/flan-t5-xl', 'main'],
            'mist-noreply-nameless': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'flanxl-textchunk': ['get_flan', 'google/flan-t5-xl', 'main'], # flan chunking at 512 chars
            'mist-noreply-verbose': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mixt-verbose': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            'mist-awq-verbose': ['get_model', 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ', 'main'],
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
prompts = ['base', 'sens_cats', 'all_cats', 'base_sens', 'sens_cats_sens', 'all_cats_sens', 'base_few', 'sens_cats_few', 'all_cats_few', 'base_sens_few', 'sens_cats_sens_few', 'all_cats_sens_few']
end_prompt = '[/INST]'
sample_size = 1

print('Starting experiment:', model_name)
print(prompts)
run_pipeline(model_name, load_func, model_path, revision, device, prompts, end_prompt) #, sample_size)
print('Finished experiment:', model_name)
