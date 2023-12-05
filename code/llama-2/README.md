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
│   ├── samples_gold.json             # the mutated data for evaluating         
│   ├── samples_llm.json
│   ├── datast
├── repair_data.json                  # sampled data for evaluating repairing
├── rectified_data.json               # rectified via LLama-2
├── postprocess_rectified_data.json   # postprocessing                     
├── README.md
```


