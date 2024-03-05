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



def itspersonal(document):
  return f"[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \nMessage: {document} \n[/INST] \n The text does"


def itspersonal_2(document):
  return f"[INST] You are an expert email reviewer and have knowledge about PII from the GDPR. Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \nMessage: {document} \n[/INST] \n The text does"

def itspersonalfewshot(document):
  return f"[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \n\nExample:\nMessage: Vince,\nHope you are well.  \nWe've been in contact with Rice University's bookshop and they informed us  that they have a few books left over from the course.  We told them  originally that if they didn't sell them all we would take them back.  We  would rather have them sent over to you as complimentary copies if that's  OK with you?\nWhat's new?  Busy summer so far?  Hope that the storm didn't  cause you any problems.\nJulie\nResponse: The text does contain personal information.\n\nExample:\nMessage: Aruna,\nI shall be in London this week. Please, call me on Monday next week. \nBest time is between 7:30 and 8:30  my time.\nVince\nResponse: The text does not contain personal information.\n\nNow answer:\nMessage: {document}. \n[/INST] \nResponse: The text does"

def itspersonalsys(document):
  return f"[INST] <<SYS>> \nYour task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \n<</SYS>> \nMessage: {document} \n[/INST] \nThe text does"


def itspersonal_2sys(document):
  return f"[INST] <<SYS>> \nYou are an expert email reviewer and have knowledge about PII from the GDPR. Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \n<</SYS>> \nMessage: {document} \n[/INST] \nThe text does"

def itspersonalfewshotsys(document):
  return f"[INST] <<SYS>> Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \n\nExample:\nMessage: Vince,\nHope you are well.  \nWe've been in contact with Rice University's bookshop and they informed us  that they have a few books left over from the course.  We told them  originally that if they didn't sell them all we would take them back.  We  would rather have them sent over to you as complimentary copies if that's  OK with you?\nWhat's new?  Busy summer so far?  Hope that the storm didn't  cause you any problems.\nJulie\nResponse: The text does contain personal information.\n\nExample:\nMessage: Aruna,\nI shall be in London this week. Please, call me on Monday next week. \nBest time is between 7:30 and 8:30  my time.\nVince\nResponse: The text does not contain personal information.\n\nNow answer: \n <</SYS>> \nMessage: {document}. \n[/INST] \nResponse: The text does"

  #\nMessage: {document} \n[/INST] \n The text does"


def fewshotsim(document, doc_sens, doc_nonsens):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? 

Example:\nMessage: {doc_sens}
Response: The text does contain personal information.

Example:\nMessage: {doc_nonsens}
Response: The text does not contain personal information.

Now answer:
Message: {document} 
[/INST] 
Response: The text does"""


# Require 150 max new tokens to be generated
def itspersonalverbose(document):
  return f"[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? Explain your decision. \nMessage: {document} \n[/INST] \n The text does"


def itspersonalverbosecot(document, response):
  return f"Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? Explain your decision. \nMessage: {document} \n {response}</s> \n[INST] Based on your response would you classify the text as containing personal information (personal) or containing no personal information (non-personal). You must answer with personal or non-personal in a Python list. [/INST] \n["


def itspersonal(document):
  return f"[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \nMessage: {document} \n[/INST] \n The text does"

def itspersonalpurely(document):
  return f"[INST] Your task is to determine if the email message from a work email contains personal information. Purely personal messages include personal information and do not include any relation to work being done. Does the message contain purely personal information? \nMessage: {document} \n[/INST] \n The text does"

def itspersonalpurelygenres(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. The message may be classified as one of eight genres: 
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message
Purely personal messages include personal information and do not include any relation to work being done. Does the message contain purely personal information? \nMessage: {document} \n[/INST] \n The text does"""

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
Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information or information that is personal a professional context? \nMessage: {document} \n[/INST] \n The text does"""

def itspersonalgenresnodesc(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. The message may be classified as one of eight genres: 
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message
Does the message contain purely personal information or information that is personal a professional context? \nMessage: {document} \n[/INST] \n The text does"""

def itspersonalbusinessdiff(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. The message may be classified as one of eight genres: 
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message
Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, but include sensitive personal information for example comments about the quality of people's work and expressions of feelings about employee treatment. Does the message contain purely personal information? \nMessage: {document} \n[/INST] \n The text does"""

def itspersonalverydiff(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains personal information. The message may be classified as one of eight genres: 
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message
Messages that include sensitive personal information are purely personal and do not include any relation to work being done, or personal but in a professional context messages and include personal information that is related to work, but also include comments about the quality of people's work and expressions of feelings about employee treatment for example. Does the message contain sensitive personal information? \nMessage: {document} \n[/INST] \n The text does"""



def get_prompt(prompt_name):
    prompt_dict = {
                  'b1': b1,
                  'b2': b2,
                  'b1_2': b1_2,
                  'b2_2': b2_2,
                  'b1sys': b1sys,
                  'b2sys': b2sys,
                  'b1_2sys': b1_2sys,
                  'b2_2sys': b2_2sys,
                  'bb': bb,
                  'bb2': bb2,
                  'itspersonal': itspersonal,
                  'itspersonal_2': itspersonal_2,
                  'itspersonalfewshot': itspersonalfewshot,
                  'itspersonalsys': itspersonalsys,
                  'itspersonal_2sys': itspersonal_2sys,
                  'itspersonalfewshotsys': itspersonalfewshotsys,
                  'fewshotsim': fewshotsim,
                  'itspersonalverbose': itspersonalverbose,
                  'itspersonalverbosecot': itspersonalverbosecot,
                  'itspersonalpurely': itspersonalpurely,
                  'itspersonalpurely': itspersonalpurely,
                  'itspersonalpurelygenres': itspersonalpurelygenres,
                  'itspersonalgenres': itspersonalgenres,
                  'itspersonalgenresnodesc': itspersonalgenresnodesc,
                  'itspersonalbusinessdiff': itspersonalbusinessdiff,
                  'itspersonalverydiff': itspersonalverydiff}

    return prompt_dict.get(prompt_name)

