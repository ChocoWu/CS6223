# LLama-2 for Translation Evaluation

## File Structure
```
├── llama2.py     # the main code
├── prompt.py     # saving the prompt for LLama-2
├── simple        # with simple prompt
│   ├── filter    # after manually checking if the mutant is killed
│   │   ├── mutant_detected_samples_gold.json                      
│   │   ├── mutant_detected_samples_llm.json                         
│   ├── detected_samples_gold.json                          # evaluation of the translation
│   ├── detected_samples_llm.json                         
│   ├── postprocess_detected_samples_gold.json              # postprocess the evaluation results
│   ├── postprocess_detected_samples_llm.json                         
│   ├── mutant_detected_samples_gold.json                   # auto-matic checking if the mutant is killed
│   ├── mutant_detected_samples_llm.json
├── complex      # with complex prompt
│   ├── filter    # after manually checking if the mutant is killed
│   │   ├── mutant_detected_samples_gold.json                      
│   │   ├── mutant_detected_samples_llm.json                         
│   ├── detected_samples_gold.json                          # evaluation of the translation
│   ├── detected_samples_llm.json                         
│   ├── postprocess_detected_samples_gold.json              # postprocess the evaluation results
│   ├── postprocess_detected_samples_llm.json                         
│   ├── mutant_detected_samples_gold.json                   # auto-matic checking if the mutant is killed
│   ├── mutant_detected_samples_llm.json                 
├── data
│   ├── samples_gold.json    # the mutated data for evaluating         
│   ├── samples_llm.json
│   ├── datast
│   │   ├── base_dataset.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── cc3m_datast.py                # process and load text-image pair dataset
│   │   ├── audiocap_datast.py            # process and load text-audio pair dataset
│   │   ├── webvid_dataset.py             # process and load text-video pair dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   ├── T-T+X_instruction_dataset.py  # process and load text-to-text+x instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model                     
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├── common
│   │   ├── anyToImageVideoAudio.py       # the main model file
│   │   ├── agent.py
│   │   ├── modeling_llama.py
│   │   ├── custom_ad.py                  # the audio diffusion 
│   │   ├── custom_sd.py                  # the image diffusion
│   │   ├── custom_vd.py                  # the video diffusion
│   │   ├── layers.py                     # the output projection layers
│   │   └── ...  
│   ├── scripts
│   │   ├── train.sh                      # training NExT-GPT script
│   │   └── app.sh                        # deploying demo script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   ├── inference.py                      # inference
│   ├── demo_app.py                       # deploy Gradio demonstration 
│   └── ...
├── ckpt                           
│   ├── delta_ckpt                        # tunable NExT-GPT params
│   │   ├── nextgpt         
│   │   │   ├── 7b_tiva_v0                # the directory to save the log file
│   │   │   │   ├── log                   # the logs
│   └── ...       
│   ├── pretrained_ckpt                   # frozen params of pretrained modules
│   │   ├── imagebind_ckpt
│   │   │   ├──huge                       # version
│   │   │   │   └──imagebind_huge.pth
│   │   ├── vicuna_ckpt
│   │   │   ├── 7b_v0                     # version
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   │   ├── tokenizer.model
│   │   │   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```


