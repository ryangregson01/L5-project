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


def textfew(document):
    return f"""[INST] <<SYS>> 
Your task is to identify documents containing sensitive personal information. 
Which one of the attributes: "sensitive", "non-sensitive" describes the following document? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["sensitve"]

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-sensitve"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""


def pdcfew(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

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


def cgfew(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

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


def pdcfewsim(document, shot1, shot2):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: {shot1}.
Answer: ["personal"]

Example:
Message: {shot2}.
Answer: ["non-personal"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""


def hop1(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains personal information.

State what personal information is present in the message if any. <</SYS>> 

Message: {document}. 
[/INST] 

Answer:"""

def hop2(document, reasoning):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email belongs to a personal or non-personal email topic.

You have already identified if personal information could be present within the message, which is appended to the message.

Personal messages discuss purely personal or personal in a professional context information. Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

State if the personal information you identified is purely personal, personal but in a professional context, or belongs to a non-personal email topic. <</SYS>> 

Message: {document}. 

Identified: {reasoning}.
[/INST] 

Answer:"""

def hop3(document, reasoning):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

You have already identified if personal information is present within the message, and then what email topic this message belongs to. This reasoning is shown under the message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute <</SYS>> 

Message: {document}. 

Reasoning: {reasoning}.
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
        'textfew': textfew,
        'pdcfew': pdcfew,
        'cgfew': cgfew,
        'pdcfewsim': pdcfewsim,
        'hop1': hop1
        }

    return prompt_dict.get(prompt_name)

