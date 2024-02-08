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


def b1_2(document):
  return f"We want to identify documents containing sensitive personal information to protect them. Using your knowledge of PII from the GDPR: is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['non-sensitive']. \nText: {document} \n"

def b2_2(document):
  return f"We want to identify documents containing sensitive personal information to protect them. Using your knowledge of PII from the GDPR: is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['sensitive']. \nText: {document} \n"

def b3_2(document):
  return f"We want to identify documents containing sensitive personal information to protect them. Using your knowledge of PII from the GDPR: is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n"

# SYS tags
def b1sys(document):
  return f"<<SYS>>\nYou are identifying documents containing sensitive personal information.\n<</SYS>>\n\nIs the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['non-sensitive']. \nText: {document} \n"

def b2sys(document):
  return f"<<SYS>>\nYou are identifying documents containing sensitive personal information.\n<<SYS>>\n\nIs the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['sensitive']. \nText: {document} \n"

def b3sys(document):
  return f"<<SYS>>\nYou are identifying documents containing sensitive personal information.\n<<SYS>>\n\nWhich one of the attributes 'sensitive', 'non-sensitive' describes a given text? You must answer with a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n"


def b1_2sys(document):
  return f"<<SYS>>\nWe want to identify documents containing sensitive personal information to protect them.\n<<SYS>>\n\nUsing your knowledge of PII from the GDPR: is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['non-sensitive']. \nText: {document} \n"

def b2_2sys(document):
  return f"<<SYS>>\nWe want to identify documents containing sensitive personal information to protect them.\n<<SYS>>\n\nUsing your knowledge of PII from the GDPR: is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a the class inside a Python list and provide no further explanation, for example ['sensitive']. \nText: {document} \n"

def b3_2sys(document):
  return f"<<SYS>>\nWe want to identify documents containing sensitive personal information to protect them.\n<<SYS>>\n\nUsing your knowledge of PII from the GDPR: is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n"



def bfor70b_2(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. We want to identify documents containing sensitive personal information to protect them. Classify the following document as either sensitive or non-sensitive. You must answer with a Python list, for example ['sensitive'] or ['non-sensitive'], and provide no further explanation. \nDocument: {document} [/INST] \nClassification: ["

def bfor70b(document):
  return f"[INST] You are identifying documents containing sensitive personal information. Which one of the attributes: 'sensitive', 'non-sensitive' describes the following text? You must answer with a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} [/INST] \nClassification: ["



def get_prompt(prompt_name):
    prompt_dict = {'base': base,
                  'persona': persona,
                  'cot': cot,
                  'b1': b1,
                  'b2': b2,
                  'b3': b3,
                  'b1_2': b1_2,
                  'b2_2': b2_2,
                  'b3_2': b3_2,
                  'b1sys': b1sys,
                  'b2sys': b2sys,
                  'b3sys': b3sys,
                  'b1_2sys': b1_2sys,
                  'b2_2sys': b2_2sys,
                  'b3_2sys': b3_2sys,
                  'bfor70b': bfor70b,
                  'bfor70b_2': bfor70b_2}

    return prompt_dict.get(prompt_name)

