model_map = {'l27b-meta': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'], # 2048 max tokens doc
            'mist7b-mist': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'mixt-4bit': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            'mist-noreply': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'l27b-noreply': ['get_l2', 'meta-llama/Llama-2-7b-chat-hf', 'main'],
            'mixt-noreply': ['get_model', 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ', 'main'],
            'flanxl-noreply': ['get_flan', 'google/flan-t5-xl', 'main'],
            'mist-noreply-nameless': ['get_model', 'mistralai/Mistral-7B-Instruct-v0.2', 'main'],
            'flanxl-textchunk': ['get_flan', 'google/flan-t5-xl', 'main'], # flan chunking at 512 chars
            }