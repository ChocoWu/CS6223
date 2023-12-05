SIMPLE_PROMPT_INSTRUCTION = '''
<s>[INST] <<SYS>>
You are a helpful, respectful and honest Chinese-to-English translation evaluator. Your task is to assess the quality of translated English sentences.
If you don't know the answer to a question, please don't share false information.
<</SYS>>
Given the source and translation, identify and concisely enumerate any major errors in the translation. Note the major errors include actual translation or grammatical/semantic errors.
'''

DEMONSTRATION = '''
Here is a demonstration:    
Source: 设立新的奖项计划是为了表彰那些礼貌得体的政治人士。
Translation: The Wednesday establishment of a new award programs is to recognize those polite and well-mannered figures political.
Output: 
(1) words: programs (error type: grammar agreement)
(2) words: figures --- political (error type: grammar swapping)
(3) words: Wednesday (error type: semantic addition)
'''

COMPLEX_PROMPT_INSTRUCTION = '''
<s>[INST] <<SYS>>
You are a helpful, respectful and honest Chinese-to-English translation evaluator. Your task is to assess the quality of translated English sentences.
If you don't know the answer to a question, please don't share false information.
<</SYS>>
As a Chinese-to-English translation evaluator, your task is to assess the quality of translated English sentences. Examine the provided source and translations, focusing on aspects like grammar swapping, grammar agreement, punctuation, semantic substitution, semantic omission, semantic addition, and semantic ambiguity. Identify and enumerate any major errors, which include actual translation or grammatical/semantic errors. Be concise in listing the errors without providing modified sentences.
'''

REPAIR_PROMPT_INSTUCTION_start = '''
<s>[INST] <<SYS>>
You are a helpful, respectful and honest Chinese-to-English translator. If you don't know the answer to a question, please don't share false information.
<</SYS>>
'''

REPAIR_PROMPT_INSTUCTION_med  = "Given the source sentence and translation, the following errors are detected in the translation:\n"
REPAIR_PROMPT_INSTUCTION_end = "\nYour task is to rectify translation based on the detected errors, and output the modified translation. [/INST]"

REPAIR_PROMPT_INSTUCTION_end_wo = "Given the source sentence and translation, there are some errors in the translation. Your task is to rectify translation and output the modified translation. [/INST]"