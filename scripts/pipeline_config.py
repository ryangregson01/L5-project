import sys
from pipeline import run_pipeline


model_map = {'l27b-meta': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'],
            'l213b-8bit': ['get_l2', 'TheBloke/Llama-2-13B-chat-GPTQ', 'gptq-8bit-64g-actorder_True'], 
            'l270b-4bit': ['get_l2', 'TheBloke/Llama-2-70B-chat-GPTQ', 'main'],
            'mist7b-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mixt-4bit': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
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
prompts = ['itspersonal', 'itspersonal_2', 'itspersonalfewshot', 'itspersonalsys', 'itspersonal_2sys', 'itspersonalfewshotsys'] #['b1', 'b2', 'b1_2', 'b2_2', 'b1sys', 'b2sys', 'b1_2sys', 'b2_2sys'] #['itspersonal', 'itspersonal_2', 'itspersonalfewshot']
end_prompt = '[/INST]'
sample_size = 2

print('Starting experiment:', model_name)
run_pipeline(model_name, load_func, model_path, revision, device, prompts, end_prompt) #, sample_size)
print('Finished experiment:', model_name)
