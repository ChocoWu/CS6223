This path saves the generated 400 samples with different mutations.
``samples_gold.json`` is generated from the orgininal gold sentence. 
``samples_llm.json`` is generated from the llm gold sentence.

- Each files contain 200 samples, respectively. 

- For both files, **Syntax mutations** are applied to 1-100 samples, and **semantics mutations** are applied to 101-200 samples. The last ten sentences have ambiguity mutation.
