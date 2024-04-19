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
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. All other messages that do not contain personal information are non-personal.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def cg(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def fs(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

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
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

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
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

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
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

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
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

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



def sens1(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message is sensitive and contains personal information that is purely personal or personal but in a professional context, or is non-sensitive and contains non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""

def sens2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message is sensitive and contains personal information that is purely personal or personal but in a professional context, or is non-sensitive and contains non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-sensitive messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""

def sens3(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message is sensitive and contains personal information that is purely personal or personal but in a professional context, or is non-sensitive and contains non-personal information.

Sensitive personal information in this context includes personal life details, health information, personal relationships, financial status, and personal identifiers. Additionally, sensitive personal information that is personal but in a professional context are any expressions of sentiment regarding colleagues work or employee treatment.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-sensitive messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def sens4(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message is sensitive and contains personal information that is purely personal or personal but in a professional context, or is non-sensitive and contains non-personal information.

Sensitive personal information in this context includes personal life details, health information, personal relationships, financial status, and personal identifiers. Additionally, sensitive personal information that is personal but in a professional context include any expressions of sentiment regarding colleagues work or employee treatment.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def sens5(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains sensitive personal information or not.

Sensitive personal information in this context includes personal life details, health information, personal relationships, financial status, and personal identifiers.

Messages that include sensitive personal information are not related to work being done. Messages that include comments about the quality of peoples work and expressions of feelings about employee treatment should be treated as sensitive personal information. Non-sensitive messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""

def sens6(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains sensitive personal information or not.

Sensitive personal information in this context includes personal life details, health information, personal relationships, financial status, and personal identifiers.

Messages that include sensitive personal information are not related to work being done. Messages that include any expressions of sentiment regarding colleagues or employee treatment should be treated as sensitive personal information. Non-sensitive messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""

def sens7(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email does or does not contain sensitive personal information.

Sensitive personal information in this context includes personal life details, health information, personal relationships, financial status, and personal identifiers.

You use the "sensitive" attribute for messages that do contain sensitive personal information. You use the "sensitive" attribute for messages that are purely personal and are not related to work. You use the "sensitive" attribute for messages that include any expressions of sentiment regarding colleagues or employee treatment.

You use the "non-sensitive" attribute for messages that do not contain sensitive personal information. You use the "non-sensitive" attribute for messages that discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def sens8(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email does or does not contain sensitive personal information.

Sensitive personal information in this context includes personal life details, health information, personal relationships, financial status, and personal identifiers.

You use the "sensitive personal information" attribute for messages that do contain sensitive personal information. You use the "sensitive personal information" attribute for messages that are purely personal and are not related to work. You use the "sensitive personal information" attribute for messages that include any expressions of sentiment regarding colleagues or employee treatment.

You use the "no sensitive personal information" attribute for messages that do not contain sensitive personal information. You use the "no sensitive personal information" attribute for messages that discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive personal information", "no sensitive personal information" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def sens9(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email does or does not contain sensitive personal information.

You use the "sensitive" attribute for messages that do contain sensitive personal information. You use the "sensitive" attribute for messages that are purely personal and are not related to work. You use the "sensitive" attribute for messages that include any expressions of sentiment regarding colleagues or employee treatment.

You use the "non-sensitive" attribute for messages that do not contain sensitive personal information. You use the "non-sensitive" attribute for messages that discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive", "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def fewshotsimone(document, shot1, label1):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: {shot1}.
Answer: ["{label1}"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""

def pdcfewsim2(document, shot1, label1, shot2, label2):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: {shot1}.
Answer: ["{label1}"].

Example:
Message: {shot2}.
Answer: ["{label2}"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""



def textverbose(document):
    return f"""[INST] <<SYS>> 
Your task is to identify documents containing sensitive personal information. 
Which one of the attributes: "sensitive", "non-sensitive" describes the following document? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Document: {document}. 
[/INST] 

Answer: ["""

def text2verbose(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def pdc2verbose(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails that do not include personal information.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def cgverbose(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains purely personal, personal but in a professional context, or non-personal information.

Purely personal messages include personal information and do not include any relation to work being done. Personal but in a professional context messages include personal information that are related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages are professional emails discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "purely personal", "personal but in a professional context", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def cgtrial(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains sensitive personal information or non-sensitive information.

Sensitive personal messages include personal information and do not include any relation to work being done. Sensitive personal messages include personal information that is related to work, for example comments about the quality of people's work and expressions of feelings about employee treatment. Non-sensitive messages are email messages discussing topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "sensitive personal", or "non-sensitive" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def cgtrial2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the email message from a work email contains sensitive personal information or non-sensitive information.

Sensitive personal information includes details like health status, financial information, family matters, personal opinions, and feelings that do not directly relate to the job or work being performed. It also includes personal comments related to work, such as opinions on the quality of someone's work or feelings about how employees are treated.

Non-sensitive information covers topics such as company business and strategy, logistic arrangements (like meeting scheduling and technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking for collaboration, and messages that are empty or seem non-contextual due to missing attachments.

Given an email message, classify it as either "sensitive personal" or "non-sensitive." If a message could arguably fit both categories, consider the dominant theme or intent of the message for classification.

Respond with a Python list containing the appropriate attribute, like ["sensitive personal"] or ["non-sensitive"].

Which one of the attributes: "sensitive personal", or "non-sensitive" best describes the following message? 
<</SYS>>

Message: {document}. 
[/INST] 

Answer: ["""



def details(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def detailspdc(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def detailsfew(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Personal messages messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["personal"]

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-personal"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""


def det1(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

You are protecting sensitive personal information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom.

Personal messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def det2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Sensitive personal information relates to an identifiable person.

Personal messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""


def det3(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Sensitive personal information refers to data that is considered private and could potentially be used to harm an individual's privacy or security if it is disclosed without consent. This type of information often includes details such as financial information, personal identification information, health information, legal information, sexual orientation and gender identity, religious or philosophical beliefs.

Personal messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""



def det4(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Sensitive personal information refers to data that is considered private and could potentially be used to harm an individual's privacy or security if it is disclosed without consent.

Personal messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 

Answer: ["""



def det1few(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

You are protecting sensitive personal information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom.

Personal messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment. Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support, etc.), employment arrangements (job seeking, hiring, recommendations, etc.), document editing/checking (collaboration), empty message (due to missing attachment), empty message.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["personal"]

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-personal"].

Now answer:
Message: {document}. 
[/INST] 
Answer: ["""

def detbase(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 
Answer: ["""

def detsens(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

You are protecting sensitive personal information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute. <</SYS>> 

Message: {document}. 
[/INST] 
Answer: ["""


def detsens2(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email.

You are reviewing emails from Enron company that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life.

You are protecting sensitive personal information refers to data that is considered private and could potentially harm an individual's privacy or security if disclosed without consent. This includes, but is not limited to, financial records, personal identification details, health information, legal matters, sexual orientation, gender identity, and religious or philosophical beliefs.

Which one of the attributes: "personal", or "non-personal" describes the following message? 
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
        'textfew': textfew,
        'pdcfew': pdcfew,
        'cgfew': cgfew,
        'pdcfewsim': pdcfewsim,
        'hop1': hop1,
        'sens1': sens1,
        'sens2': sens2,
        'sens3': sens3,
        'sens4': sens4,
        'sens5': sens5,
        'sens6': sens6,
        'sens7': sens7,
        'sens8': sens8,
        'sens9': sens9,
        'pdcfewsim2': pdcfewsim2,
        'textverbose': textverbose,
        'pdc2verbose': pdc2verbose,
        'cgverbose': cgverbose,
        'cgtrial': cgtrial,
        'cgtrial2': cgtrial2,
        'details': details,
        'detailspdc': detailspdc,
        'detailsfew': detailsfew,
        'det1': det1,
        'det2': det2,
        'det3': det3,
        'det4': det4,
        'det1few': det1few,
        'detbase': detbase,
        'detsens': detsens,
        'detsens2': detsens2,
        }

    return prompt_dict.get(prompt_name)
