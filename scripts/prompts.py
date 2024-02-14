def base(document):
  return f"""Sensitivity is a measure of sensitive personal information in a purely personal and personal in a professional context. 
The Text is delimited by [TEXT]. Which one of the attributes “sensitive”, “non-sensitive” classifies the Text. Only respond with the attribute after 'Answer:'.
[TEXT]{document}[/TEXT].
Answer:
"""

def b1(document):
  return f"You are identifying documents containing sensitive personal information. Is the given text classified as 'sensitive' or 'non-sensitive'? You must answer with the classification inside a Python list and provide no further explanation, for example ['non-sensitive']. \nText: {document} \n"


def b1(document):
  return f"Your task is to identify documents containing sensitive personal information. Does the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list, with no additional text or explanation. Use ['sensitive'] for email that contain sensitive personal information and ['non-sensitive'] for all other emails. This classification helps in understanding how to handle the information within these documents. \nText: {document} \n"


# Used in experiments
def b1(document):
  return f"[INST] Your task is to identify documents containing sensitive personal information. Does the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list and provide no further explanation. Use ['sensitive'] if the text contains sensitive personal information and ['non-sensitive'] for all other documents. \nText: {document} \n[/INST]\nClassification:"

def b2(document):
  return f"[INST] Your task is to identify documents containing sensitive personal information. Which one of the attributes 'sensitive', 'non-sensitive' describes a given text? Write your answer in the form of a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n[/INST]\nClassification:"

def b1_2(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. Your task is to identify documents containing sensitive personal information to protect them. Does the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list and provide no further explanation. Use ['sensitive'] if the text contains sensitive personal information and ['non-sensitive'] for all other documents. \nText: {document} \n[/INST]\nClassification:"

def b2_2(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. Your task is to identify documents containing sensitive personal information to protect them. Which one of the attributes 'sensitive', 'non-sensitive' describes a given text? Write your answer in the form of a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n[/INST]\nClassification:"


# SYS tags
def b1sys(document):
  return f"[INST] <<SYS>>\nYour task is to identify documents containing sensitive personal information.\n<</SYS>>\n\nDoes the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list and provide no further explanation. Use ['sensitive'] if the text contains sensitive personal information and ['non-sensitive'] for all other documents. \nText: {document} \n[/INST]\nClassification:"

def b2sys(document):
  return f"[INST] <<SYS>>\nYour task is to identify documents containing sensitive personal information.\n<</SYS>>\n\nWhich one of the attributes 'sensitive', 'non-sensitive' describes a given text? Write your answer in the form of a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n[/INST]\nClassification:"

def b1_2sys(document):
  return f"[INST] <<SYS>>\nYou are an expert sensitivity reviewer with knowledge about PII from the GDPR. Your task is to identify documents containing sensitive personal information to protect them.\n<</SYS>>\n\nDoes the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list and provide no further explanation. Use ['sensitive'] if the text contains sensitive personal information and ['non-sensitive'] for all other documents. \nText: {document} \n[/INST]\nClassification:"

def b2_2sys(document):
  return f"[INST] <<SYS>>\nYou are an expert sensitivity reviewer with knowledge about PII from the GDPR. Your task is to identify documents containing sensitive personal information to protect them.\n<</SYS>>\n\nWhich one of the attributes 'sensitive', 'non-sensitive' describes a given text? Write your answer in the form of a Python list containing the only appropriate attribute and provide no further explanation. \nText: {document} \n[/INST]\nClassification:"


def bb(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. We want to identify documents containing sensitive personal information to protect them. Classify the document as either sensitive or non-sensitive. You must answer after 'Classification', and provide no further explanation. \nDocument: {document} [/INST] \nClassification: "

def bb2(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. We want to identify documents containing sensitive personal information to protect them. \nDocument: {document} \n\nClassify the document as either sensitive or non-sensitive. You must answer after 'Classification', and provide no further explanation. [/INST] \nClassification: "



def get_prompt(prompt_name):
    prompt_dict = {'base': base,
                  'b1': b1,
                  'b2': b2,
                  'b1_2': b1_2,
                  'b2_2': b2_2,
                  'b1sys': b1sys,
                  'b2sys': b2sys,
                  'b1_2sys': b1_2sys,
                  'b2_2sys': b2_2sys,
                  'bb': bb,
                  'bb2': bb2}

    return prompt_dict.get(prompt_name)

