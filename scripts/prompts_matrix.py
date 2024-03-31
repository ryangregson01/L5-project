def base_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. 
Which one of the attributes: "sensitive", "non-sensitive" describes a given text? 
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Text: {document}. 
[/INST] 
"""

def base_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information.
Does the text contain sensitive personal information? <</SYS>> 
Text: {document}. 
[/INST] 
The text does"""


def gdpr_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You should use knowledge about PII from the GDPR to help you reach a decision.
Which one of the attributes: "sensitive", "non-sensitive" describes a given text?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Text: {document}. 
[/INST] 
"""

def gdpr_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You should use knowledge about PII from the GDPR to help you reach a decision.
Does the text contain sensitive personal information? <</SYS>> 
Text: {document}. 
[/INST] 
The text does"""

def workemail_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
Which one of the attributes: "sensitive", "non-sensitive" describes a given message?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Message: {document}. 
[/INST] 
"""

def workemail_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
Does the message contain sensitive personal information? <</SYS>> 
Message: {document}. 
[/INST] 
The message does"""


def context1_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Which one of the attributes: "sensitive", "non-sensitive" describes a given message?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Message: {document}. 
[/INST] 
"""

def context1_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Does the message contain sensitive personal information? <</SYS>> 
Message: {document}. 
[/INST] 
The message does"""


def context2_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.

The message belongs to one of the following eight genres:
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message

The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal or personal but in a professional. 
non-sensitve messages are professional emails that do not include personal information.

Which one of the attributes: "sensitive", "non-sensitive" describes a given message?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Message: {document}. 
[/INST] 
"""

def context2_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.

The message belongs to one of the following eight genres:
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message

The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal or personal but in a professional. 
non-sensitve messages are professional emails that do not include personal information.

Does the message contain sensitive personal information? <</SYS>> 
Message: {document}. 
[/INST] 
The message does"""


def context1_fewshot_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Which one of the attributes: "sensitive", "non-sensitive" describes a given message?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Example:
Message: original message from julie sent tuesday june am to kaminski vince j subject extra books vince hope you are well weve been in contact with rice universitys bookshop and they informed us that they have a few books left over from the course we told them originally that if they didnt sell them all we would take them back we would rather have them sent over to you as complimentary copies if thats ok with you whats new busy summer so far hope that the storm didnt cause you any problems julie.
Response: [ "sensitive" ]

Example:
Message: aruna i shall be in london this week please call me on monday next week best time is between and my time vince.
Response: [ "non-sensitive" ]

Now answer:
Message: {document}. 
[/INST] 
"""

def context1_fewshot_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Does the message contain sensitive personal information? <</SYS>> 
Example:
Message: original message from julie sent tuesday june am to kaminski vince j subject extra books vince hope you are well weve been in contact with rice universitys bookshop and they informed us that they have a few books left over from the course we told them originally that if they didnt sell them all we would take them back we would rather have them sent over to you as complimentary copies if thats ok with you whats new busy summer so far hope that the storm didnt cause you any problems julie.
Response: The message does contain personal information.

Example:
Message: aruna i shall be in london this week please call me on monday next week best time is between and my time vince.
Response: The message does not contain personal information.

Now answer:
Message: {document}. 
[/INST] 
The message does"""

def multi_category(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. Does the message contain purely personal information or information that is personal a professional context? 
Message: {document}. 
[/INST] 
The text does"""

def context1_qa(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.
Does the message contain sensitive personal information? This is personal information that should be protected under GDPR. Note names and email addresses within the message are not classed as sensitive personal information.
<</SYS>> 
Message: {document}. 
[/INST] 
The message does"""

def persnameless(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.
Does the message contain sensitive personal information? This is personal information that should be protected under GDPR.
<</SYS>> 
Message: {document}. 
[/INST] 
The message does"""

def persnamelessv2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.
Does the message contain sensitive personal information?
<</SYS>> 
Message: {document}. 
[/INST] 
The message does"""

def context1_fewshot_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Does the message contain sensitive personal information? <</SYS>> 
Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Response: The message does contain personal information.

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Response: The message does not contain personal information.

Now answer:
Message: {document}. 
[/INST] 
The message does"""


def get_prompt_matrixold(prompt_name):
    prompt_dict = {
        'gdpr_class': gdpr_class,
        'gdpr_qa': gdpr_qa,
        'workemail_class': workemail_class,
        'workemail_qa': workemail_qa,
        'context1_class': context1_class,
        'context1_qa': context1_qa,
        'context2_class': context2_class,
        'context2_qa': context2_qa,
        'context1_fewshot_class': context1_fewshot_class,
        'context1_fewshot_qa': context1_fewshot_qa,
        'persnameless': persnameless,
        'persnamelessv2': persnamelessv2
        }

    return prompt_dict.get(prompt_name)




def base_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. 
Which one of the attributes: "sensitive", "non-sensitive" describes a given text? 
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Text: {document}. 
[/INST] 
"""

def base_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information.
Does the text contain sensitive personal information? <</SYS>> 
Text: {document}. 
[/INST] 
The text does"""

def context1_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Which one of the attributes: "sensitive", "non-sensitive" describes a given message?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Message: {document}. 
[/INST] 
"""

def context1_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.
The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal and do not include any relation to work being done. 
sensitive messages are messages that include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. 
non-sensitve messages are professional emails that do not include personal information.

Does the message contain sensitive personal information? <</SYS>> 
Message: {document}. 
[/INST] 
The message does"""


def context2_class(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.

The message belongs to one of the following eight genres:
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message

The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal or personal but in a professional. 
non-sensitve messages are professional emails that do not include personal information.

Which one of the attributes: "sensitive", "non-sensitive" describes a given message?
Always your answer in the form of a Python list containing the appropriate attribute. <<SYS>> 
Message: {document}. 
[/INST] 
"""

def context2_qa(document):
    return f"""[INST] <<SYS>> 
You generate sensitivity classifications for emails that have potentially sensitive personal information. You will be given an email message from a work email.

The message belongs to one of the following eight genres:
1. Company business and strategy
2. Purely personal
3. Personal but in a professional context (e.g., it was good working with you)
4. Logistic arrangements (meeting scheduling, technical support, etc.)
5. Employment arrangements (job seeking, hiring, recommendations, etc.)
6. Document editing/checking (collaboration)
7. Empty message (due to missing attachment)
8. Empty message

The following information describes sensitive personal information that may be found within emails:
sensitive messages are messages that are purely personal or personal but in a professional. 
non-sensitve messages are professional emails that do not include personal information.

Does the message contain sensitive personal information? <</SYS>> 
Message: {document}. 
[/INST] 
The message does"""

def multi_category(document):
  return f"""[INST] Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information. Does the message contain purely personal information or information that is personal a professional context? 
Message: {document}. 
[/INST] 
The text does"""

def context1_qa(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.
Does the message contain sensitive personal information? This is personal information that should be protected under GDPR. Note names and email addresses within the message are not classed as sensitive personal information.
<</SYS>> 
Message: {document}. 
[/INST] 
The message does"""



def pers_multi(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which of these choices best describes the following message?: "purely personal", "personal but in a professional context", or "non-personal".
Always your answer in the form of a Python list containing the appropriate attribute.
<</SYS>> 
Message: {document}. 
[/INST] 
Answer: ["""

def pers_multi2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include senstive personal information about others, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which of these choices best describes the following message?: "purely personal", "personal but in a professional context", or "non-personal".
Always your answer in the form of a Python list containing the appropriate attribute.
<</SYS>> 
Message: {document}. 
[/INST] 
Answer: ["""

def pers_multi3(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email is purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which of these choices best describes the following message?: "purely personal", "personal but in a professional context", or "non-personal".
Always your answer in the form of a Python list containing the appropriate attribute.
<</SYS>> 
Message: {document}. 
[/INST] 
Answer: ["""



def get_prompt_matrix(prompt_name):
    prompt_dict = {
        'pers_multi': pers_multi,
        'pers_multi2': pers_multi2,
        'pers_multi3': pers_multi3,
        }

    return prompt_dict.get(prompt_name)



def text(document):
    return f"""[INST] <<SYS>> 
Your task is to identify documents containing sensitive personal information. 
Which one of the attributes: "sensitive", "non-sensitive" describes the following document? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Document: {document}. 
[/INST] 

Answer: ["""


def pdc(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. All other messages that do not contain personal information are non-personal.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def cg(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def fs(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. All other messages that do not contain personal information are non-personal.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["purely personal"]

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-personal"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""


def textqa(document):
    return f"""[INST] <<SYS>> 
Your task is to identify documents containing sensitive personal information. 

Does the document contain sensitive personal information? <</SYS>> 

Document: {document}. 
[/INST] 

The document does"""


def pdcqa(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains sensitive personal information.

Sensitive personal information is exposed within purely personal and personal but in a professional context messages. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. All other messages that do not contain personal information are non-personal.

Does the document contain sensitive personal information? <</SYS>> 

Message: {document}. 
[/INST] 

The document does"""


def cgqa(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains sensitive personal information.

Sensitive personal information is exposed within purely personal and personal but in a professional context messages. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Does the document contain sensitive personal information? <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def pdc2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def text(document):
    return f"""[INST] <<SYS>> 
Your task is to identify documents containing sensitive personal information. 
Which one of the attributes: "sensitive", "non-sensitive" describes the following document? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Document: {document}. 
[/INST] 

Answer: ["""


def pdc2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def cg(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""



def get_prompt_matrix(prompt_name):
    prompt_dict = {
        'text': text,
        'pdc': pdc,
        'cg': cg,
        'textqa': textqa,
        'pdcqa': pdcqa,
        'cgqa': cgqa,
        'pdc2': pdc2,
        }

    return prompt_dict.get(prompt_name)

