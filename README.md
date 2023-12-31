# CS6223

## Data Processing

- Data Source: We take the ***WMT-20*** as the source data.
- Random sampling ***400*** instances for testing, which is saved in `data/sample_400.csv`. 
  - Then, we employ the first ***200*** instances (`data/sample_200_for_llm.csv`) for testing LLM-based translation.
    - The stylized translations for the 200 sentences are saved in `data/data_from_llm.csv`, where 1-50, 51-100, 101-150, 151-200 sentences are translated in the style of *Polite*, *Casual*, *Technical*, and *Humorous*, respectively. 
  - The rest ***200*** instances (`data/sample_200_for_gold.csv`) are taken as gold translation from Translator (e.g., Google Translator).  

- **Generated Data**: Based on each instance, we apply different mutations, and the results are saved in `data/generated_data/`.
  - The mutation is chosen from \[*grammer_swap*, *grammer_agreement*,  *grammer_punc*, *semantic_substitution*, *semantic_omission*, *semantic_addition*, *semantic_ambiguity*\]

- **Final Data**: Utilizing the `NLTK` tool to perform mutation. Details refer to `nltk_mutate.ipynb`.
  - `samples_gold.json` stores the mutation data based on gold translation.
  - `samples_llm.json` stores the mutation data based on llm translation.

## Evaluating via LLama-2
- Please refer to the directory `code/llama-2` for the details.
  
## Evaluating via ChatGPT
- Please refer to the directory `code/chatgpt-based` for the details. 
