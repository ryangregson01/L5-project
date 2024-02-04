def base(document):
  return f"""Sensitivity is a measure of sensitive personal information in a purely personal and personal in a professional context. 
The Text is delimited by [TEXT]. Which one of the attributes “sensitive”, “non-sensitive” classifies the Text. Only respond with the attribute after 'Answer:'.
[TEXT]{document}[/TEXT].
Answer:
"""

def persona(document):
  return f"""Sensitivity is a measure of sensitive personal information in a purely personal and personal in a professional context. From now on, act as an expert sensitivity reviewer. Pay close attention to the personal information details within the document.
The Text is delimited by [TEXT]. Which one of the attributes “sensitive”, “non-sensitive” classifies the Text. Only respond with the attribute after 'Answer:'.
[TEXT]{document}[/TEXT].
Answer:
"""

def cot(document):
  return f"""Sensitivity is a measure of sensitive personal information in a purely personal and personal in a professional context. 
The Text is delimited by [TEXT]. Which one of the attributes “sensitive”, “non-sensitive” classifies the Text. Only respond with the attribute after 'Answer:'. Lets think step by step.
[TEXT]{document}[/TEXT].
Answer:
"""

def b1(document):
  return f"You are identifying documents containing personal sensitive information. Is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['non-sensitive']. \nText: {document} \n"

def b2(document):
  return f"You are identifying documents containing personal sensitive information. Is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['sensitive']. \nText: {document} \n"

def b3(document):
  return f"You are identifying documents containing personal sensitive information. Which one of the attributes 'sensitive', 'non-sensitive' describes a given text? You must answer with a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n"


def get_prompt(prompt_name):
    prompt_dict = {'base': base,
                  'persona': persona,
                  'cot': cot,
                  'b1': b1,
                  'b2': b2,
                  'b3': b3}
    return prompt_dict.get(prompt_name)

