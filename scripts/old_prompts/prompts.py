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

'''
def base_personal(document):
  return f"[INST] Your task is to determine if the email message from a work email contains sensitive personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. If the message contains purely personal information or personal information but in a professional context classify the message as sensitive, otherwise for any other professional email use non-sensitive.
Message: {document}. 
[/INST] 
Classification:"

def barlit(document):
  return f"Would the following be protected under FOIA exemption '40'? Answer with yes or no.
{document}
  "
def barlit(document):
  return f"[INST] Would the following be protected under FOIA exemption 6? Classify the document with 'exempt' if it should be protected, and 'non-exempt' otherwise. [/INST]
Document: {document}. 
Classification:"
'''

def barlit(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains sensitive personal information. Would the following document be protected under FOIA exemption 6? Answer with 'sensitive' if the document should be protected, and 'non-sensitive' if it does not need to be protected.
Document: {document}. 
[/INST] 
Answer:"""

def barlit2(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains sensitive personal information. The UK FOIA Section '40' protects sensitive personal information by classifying documents as exempt if it contains sensitive personal information. Would the following document be exempt under UK FOIA Section '40'? Answer with 'sensitive' if the document should be exempt, and 'non-sensitive' if it is not exempt.
Document: {document}. 
[/INST] 
Answer:"""


def sysjson(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information.
Always output JSON.
You will be given an email message from a work email. Output only one section: ['Classification'].

Email messages may contain sensitive personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. If the email is purely personal or personal but in professional context, the message is sensitive, otherwise non-personal messages are not sensitive. The classification should be sensitive or not sensitive. <</SYS>> 
message: {document}. 
[/INST] 
"""


def prev_sysjson(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information.
Always output JSON.
You will be given an email message from a work email. Output only one section: ['Classification'].

Email messages may contain sensitive personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. If the email is purely personal or personal but in professional context, the message is sensitive, otherwise non-personal messages are non-sensitive. The classification should be sensitive or non-sensitive. <</SYS>> 
message: {document}. 
[/INST] 
{{"classification":"""

def olsysjson(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. The sensitivity classification is either sensitive or non-sensitive. Sensitive personal information can be purely personal or personal but in a professional context. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. Any other email message that is professional and does not include personal information is non-sensitive.
Always output JSON.
You will be given an email message from a work email. Output only one section: ['Classification']. <</SYS>>
message: {document}. 
[/INST] 
{{"classification":"""

def sysjson(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. Your task is to determine if the email message from a work email contains sensitive personal information (emails that are purely personal or personal but in a professional context), or contains no sensitive personal information (emails that discuss company business and strategy, logistic arrangements, employment arrangements, document editing/checking, or empty messages). Purely personal emails include personal information and do not include any relation to work being done. Personal but in a professional context emails include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-sensitive emails are professional emails that do not include personal information. You must classify the email as sensitive or non-sensitive for containing sensitive personal information.
Always output JSON.
You will be given an email message from a work email. Output only one section: ['Classification']. <</SYS>>
Example:
email: original message from julie sent tuesday june am to kaminski vince j subject extra books vince hope you are well weve been in contact with rice universitys bookshop and they informed us that they have a few books left over from the course we told them originally that if they didnt sell them all we would take them back we would rather have them sent over to you as complimentary copies if thats ok with you whats new busy summer so far hope that the storm didnt cause you any problems julie.
{{"classification": "sensitive"}}

Example:
email: aruna i shall be in london this week please call me on monday next week best time is between and my time vince.
{{"classification": "non-sensitive"}}

Now answer:
email: {document}. 
[/INST] 
{{"classification":"""

def sysjsonrank(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. Your task is to determine if the email message from a work email contains sensitive personal information (emails that are purely personal or personal but in a professional context), or contains no sensitive personal information (emails that discuss company business and strategy, logistic arrangements, employment arrangements, document editing/checking, or empty messages). Purely personal emails include personal information and do not include any relation to work being done. Personal but in a professional context emails include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-sensitive emails are professional emails that do not include personal information. You must classify the email as sensitive or non-sensitive regarding it containing sensitive personal information. To do this you will score how sensitive the document is using an integer between 0 and 10, where 0 is entirely non-sensitive and 10 is very sensitive.
Always output JSON.
You will be given an email message from a work email. Output only one section: ['classification score']. <</SYS>>
email: {document}. 
[/INST] 
{{"classification score":"""

def multi_category_fewshot(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. Does the message contain purely personal information or information that is personal a professional context? 
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

def sysclass(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for email messages that potentially have sensitive personal information. You will be given an email message from a work email. Your task is to determine if the message contains sensitive personal information. You must classify the message as sensitive or non-sensitive. Classify purely personal messages as sensitive; these include personal information and do not include any relation to work being done. Classify personal but in a professional context messages as sensitive; these include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Classify all other emails as non-sensitive.
Always output JSON. Output only one section: ['Classification']. <</SYS>>
message: {document}. 
[/INST] 
{{"classification":"""

def sysqa(document):
  return f"""[INST] <<SYS>> 
You generate sensitivity classifications for email messages that potentially have sensitive personal information. You will be given an email message from a work email. Your task is to determine if the message contains sensitive personal information. You must classify the message as sensitive or non-sensitive. Treat purely personal messages as sensitive; these include personal information and do not include any relation to work being done. Treat personal but in a professional context messages as sensitive; these include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Treat non-personal messages that are professional and do not include personal information as non-sensitive.
Does the message contain sensitive personal information?
Output your answer following 'The message does'. <</SYS>>
email: {document}. 
[/INST] 
The message does"""


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
                  'base_classify': base_classify,
                  'barlit': barlit,
                  'barlit2': barlit2,
                  'sysjson': sysjson,
                  'sysjsonrank': sysjsonrank,
                  'multi_category_fewshot': multi_category_fewshot,
                  'sysclass': sysclass,
                  'sysqa': sysqa
                  }

    return prompt_dict.get(prompt_name)
