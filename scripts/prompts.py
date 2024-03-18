# Used in experiments
def base_sens(document):
  return f"""[INST] Your task is to identify documents containing sensitive personal information. Does the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list and provide no further explanation. Use ['sensitive'] if the text contains sensitive personal information and ['non-sensitive'] for all other documents. 
Text: {document}. 
[/INST] 
Classification:"""

def base2_sens(document):
  return f"""[INST] Your task is to identify documents containing sensitive personal information. Which one of the attributes 'sensitive', 'non-sensitive' describes a given text? Write your answer in the form of a Python list containing the only appropriate attribute and provide no further explanation.
Text: {document}. 
[/INST] 
Classification:"""

def context_b1_sens(document):
  return f"""[INST] You should use knowledge about PII from the GDPR to review potentially sensitive documents. Your task is to identify documents containing sensitive personal information. Does the given text contain 'sensitive' personal information or is it 'non-sensitive'? Respond by placing your classification in a Python list and provide no further explanation. Use ['sensitive'] if the text contains sensitive personal information and ['non-sensitive'] for all other documents. 
Text: {document}. 
[/INST] 
Classification:"""

def context_b2_sens(document):
  return f"""[INST] You should use knowledge about PII from the GDPR to review potentially sensitive documents. Your task is to identify documents containing sensitive personal information. Which one of the attributes 'sensitive', 'non-sensitive' describes a given text? Write your answer in the form of a Python list containing the only appropriate attribute and provide no further explanation. 
Text: {document}. 
[/INST] 
Classification:"""

'''Come back if above works
def bb(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. We want to identify documents containing sensitive personal information to protect them. Classify the document as either sensitive or non-sensitive. You must answer after 'Classification', and provide no further explanation. \nDocument: {document} [/INST] \nClassification: "

def bb2(document):
  return f"[INST] You are an expert sensitivity reviewer with knowledge about PII from the GDPR. We want to identify documents containing sensitive personal information to protect them. \nDocument: {document} \n\nClassify the document as either sensitive or non-sensitive. You must answer after 'Classification', and provide no further explanation. [/INST] \nClassification: "
'''

def base_personal(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? 
Message: {document}. 
[/INST] 
The text does"""

def context_b1_personal(document):
  return f"""[INST] You should use knowledge about PII from the GDPR to review potentially sensitive documents. Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? 
Message: {document}. 
[/INST] 
The text does"""

def fixed_fewshot_personal(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? 

Example:
Message: original message from julie sent tuesday june am to kaminski vince j subject extra books vince hope you are well weve been in contact with rice universitys bookshop and they informed us that they have a few books left over from the course we told them originally that if they didnt sell them all we would take them back we would rather have them sent over to you as complimentary copies if thats ok with you whats new busy summer so far hope that the storm didnt cause you any problems julie.
Response: The text does contain personal information.

Example:
Message: aruna i shall be in london this week please call me on monday next week best time is between and my time vince.
Response: The text does not contain personal information.

Now answer:
Message: {document}. 
[/INST] 
Response: The text does"""

# Require 150 max new tokens to be generated
def base_personal_explanation(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? Explain your decision.
Message: {document}. 
[/INST] 
The text does"""

def purely_personal(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Does the message contain purely personal information? 
Message: {document}. 
[/INST] 
The text does"""

def itspersonalgenres(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. The message may be classified as one of eight genres: 
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message
Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? 
Message: {document}. 
[/INST] 
The text does"""

def multi_category_noanseng(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. Does the message contain purely personal information, information that is personal a professional context, or is it non-personal? 
Message: {document}. 
[/INST] 
The text does"""

def multi_category(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. Does the message contain purely personal information or information that is personal a professional context? 
Message: {document}. 
[/INST] 
The text does"""

def base_classify(document):
  return f"""[INST] Classify an email message from a work inbox as sensitive or not sensitive based on the presence of sensitive personal information. An email should be categorized as sensitive if it includes personal information unrelated to work tasks, such as personal life details or issues. Additionally, classify an email as sensitive if it contains personal opinions or feelings related to work, like comments on work quality or sentiments about employee treatment. All other emails discussing company business strategies, or professional tasks, without personal information or opinions, should not be considered sensitive. If an email does not meet the criteria for being sensitive, categorise it as non-sensitive. Carefully review the content and assign it to one of 'sensitive' or 'non-sensitive' based on these guidelines. 
Message: {document}. 
[/INST] 
Classification:"""

def get_prompt(prompt_name):
    prompt_dict = {
                  'base_sens': base_sens,
                  'base2_sens': base2_sens,
                  'context_b1_sens': context_b1_sens,
                  'context_b2_sens': context_b2_sens,
                  'base_personal': base_personal,
                  'context_b1_personal': context_b1_personal,
                  'fixed_fewshot_personal': fixed_fewshot_personal,
                  'base_personal_explanation': base_personal_explanation,
                  'purely_personal': purely_personal,
                  'itspersonalgenres': itspersonalgenres,
                  'multi_category_noanseng': multi_category_noanseng,
                  'multi_category': multi_category,
                  'base_classify': base_classify
                  }

    return prompt_dict.get(prompt_name)

