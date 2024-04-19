def task(): return """Your task is to determine if the following email message contains sensitive personal information that should be protected, or is a non-personal email."""

def collection_context(): return """You are reviewing emails from the company Enron that are related to the California energy crisis, trading discrepancies, and other messages occurring near the end of the company's life."""

def question_config(): return """Which one of the attributes: "personal", or "non-personal" describes the following message? 
Always answer in the form of a Python list containing the appropriate attribute."""

def message(d): return f"""Message: {d}. """

def answer(): return """Answer: ["""

def base(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{question_config()} <</SYS>> 

{message(document)}
[/INST] 

{answer()}"""

def sens_cat_info(): return """Personal messages may be purely personal and do not include any relation to work being done. Personal messages that are related to work will contain comments about the quality of people's work and expressions of feelings about employee treatment."""

def non_sens_cat_info(): return """Non-personal messages discuss topics such as company business and strategy, logistic arrangements (meeting scheduling, technical support), employment arrangements (job seeking, hiring, recommendations), document editing/checking (collaboration), empty message (due to missing attachment), empty message."""

def sens_cats(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_cat_info()}

{question_config()} <</SYS>> 

{message(document)}
[/INST] 

{answer()}"""


def all_cats(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_cat_info()} {non_sens_cat_info()}

{question_config()} <</SYS>> 

{message(document)}
[/INST] 

{answer()}"""


def few():
    return f"""Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["personal"]

Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-personal"].

Now answer:
"""


def flip_few():
    return f"""Example:
Message: original message from alvarez ray sent thursday august pm to sanders richard b williams robert c cc steffes james d comnes alan robertson linda shapiro richard subject california refund proceeding privileged and confidential attorney work product attorney client communication in our litigation meeting last tuesday we learned that netting of spot purchases and sales is of paramount importance in limiting our exposure to refunds under the fercs july order in both the california and pacific northwest proceedings however in the california proceeding we are also afforded the opportunity in the order to offset our receivables against any refund that would otherwise have to be paid by us also to the extent that our receivables exceed the refund amount the order provides for payment of interest to us in accordance with cfr section a at present i am aware of the existence of three receivables as you will recall i mentioned at tuesdays meeting that we should be able to treat as a receivable for purposes of offset the revenues due us from the application of the caiso underscheduling penalty i communicated the possibility of offsetting this receivable against any potential refund at this mornings western wholesale conference call and subsequent research conducted by alan comnes indicates that the caiso owes us mm as per the caisos own numbers the caiso also owes us approximately mm due to nonpayment for one to two months of business these two figures alone totalling approximately mm exceed the estimate of our potential exposure in the california proceeding finally the negative ctc issue is another receivable due us by ious that under the plain language of the order could potentially be used to offset any refund amount i am not aware of the magnitude of this receivable but do understand that there are some sensitivities regarding this issue including concerns about revealing the amount of the credit due us it is likely that the amount would become public if it is presented in the california proceeding please give us your thoughts as to the use of the ctc receivable to offset potential refunds in the california proceeding.
Answer: ["non-personal"].

Example:
Message: i am so sorry you guys gave him a really good life robert kean on pm please respond to to pat hilleman e mail greg smith e mail greg smith e mail colin lamb e mail maggie kean e mail nora kean e mail nora kean e mail sann evelyn gossum e mail doug karen reiman e mail jean kean e mail kathy wedig e mail melissa kean e mail phil kean e mail steve melissa kean e mail cc subject houston houston died today after a long battle with auto immune hemolytic anemia rest in peace rob kean.
Answer: ["personal"]

Now answer:
"""


def base_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{question_config()} <</SYS>> 

{few()}
{message(document)}
[/INST] 

{answer()}"""


def sens_cats_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_cat_info()}

{question_config()} <</SYS>> 

{few()}
{message(document)}
[/INST] 

{answer()}"""


def all_cats_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_cat_info()} {non_sens_cat_info()}

{question_config()} <</SYS>> 

{few()}
{message(document)}
[/INST] 

{answer()}"""


def sens_context(): return """You are protecting sensitive personal information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom."""
"""
You are protecting sensitive personal information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom.
Sensitive personal information relates to an identifiable person.
Sensitive personal information refers to data that is considered private and could potentially be used to harm an individual's privacy or security if it is disclosed without consent. This type of information often includes details such as financial information, personal identification information, health information, legal information, sexual orientation and gender identity, religious or philosophical beliefs.
Sensitive personal information refers to data that is considered private and could potentially be used to harm an individual's privacy or security if it is disclosed without consent.
You are reviewing emails to identify sensitive personal information, which is defined as data that is considered private and could harm an individual's privacy or security if disclosed without consent. This includes financial information, personal identification, health and legal information, sexual orientation, gender identity, and religious or philosophical beliefs. The aim is to protect information that would be exempt under Section 40 of the Freedom of Information Act in the United Kingdom.

Sensitive personal information refers to data that is considered private and could potentially harm an individual's privacy or security if disclosed without consent. This includes, but is not limited to, financial records, personal identification details, health information, legal matters, sexual orientation, gender identity, and religious or philosophical beliefs.
"""

def base_sens(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{question_config()} <</SYS>> 

{message(document)}
[/INST] 

{answer()}"""

def sens_cats_sens(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{sens_cat_info()}

{question_config()} <</SYS>> 

{message(document)}
[/INST] 

{answer()}"""


def all_cats_sens(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{sens_cat_info()} {non_sens_cat_info()}

{question_config()} <</SYS>> 

{message(document)}
[/INST] 

{answer()}"""


def base_sens_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{question_config()} <</SYS>> 

{few()}
{message(document)}
[/INST] 

{answer()}"""


def sens_cats_sens_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{sens_cat_info()}

{question_config()} <</SYS>> 

{few()}
{message(document)}
[/INST] 

{answer()}"""


def all_cats_sens_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{sens_cat_info()} {non_sens_cat_info()}

{question_config()} <</SYS>> 

{few()}
{message(document)}
[/INST] 

{answer()}"""


def all_cats_sens_flip_few(document):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{sens_cat_info()} {non_sens_cat_info()}

{question_config()} <</SYS>> 

{flip_few()}
{message(document)}
[/INST] 

{answer()}"""


def all_cats_sens_hop1(document):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains personal information.

{collection_context()}

State what personal information is present in the message if any. <</SYS>> 

Message: {document}. 
[/INST] 

Answer:"""

def identified(i): return f"""Identified: {i}. """

def full_reasoning(r): return f"""Reasoning: {r}. """

def all_cats_sens_hop2(document, reasoning):
    return f"""[INST] <<SYS>> 
Your task is to determine if the following email message contains sensitive personal information.

{collection_context()}

{sens_context()}

You have already identified if personal information could be present within the message, which is shown after the message.

State if the personal information you identified is sensitive personal information that should be protected. <</SYS>> 

{message(document)}

{identified(reasoning)}
[/INST] 

Answer:"""


def all_cats_sens_hop3(document, reasoning):
    return f"""[INST] <<SYS>> 
{task()}

{collection_context()}

{sens_context()}

{sens_cat_info()} {non_sens_cat_info()}

You have already identified if personal information is present within the message, and then if this personal information is sensitive. This reasoning is shown after the message.

{question_config()} <</SYS>> 

{message(document)}

{full_reasoning(reasoning)}
[/INST] 

{answer()}"""




def get_prompt_matrix(prompt_name):
    prompt_dict = {
        'base': base,
        'sens_cats': sens_cats,
        'all_cats': all_cats,
        'base_few': base_few,
        'sens_cats_few': sens_cats_few,
        'all_cats_few': all_cats_few,
        'base_sens': base_sens,
        'sens_cats_sens': sens_cats_sens,
        'all_cats_sens': all_cats_sens,
        'base_sens_few': base_sens_few,
        'sens_cats_sens_few': sens_cats_sens_few,
        'all_cats_sens_few': all_cats_sens_few,
        'all_cats_sens_hop1': all_cats_sens_hop1,
        'all_cats_sens_hop2': all_cats_sens_hop2,
        'all_cats_sens_hop3': all_cats_sens_hop3,
        'all_cats_sens_flip_few': all_cats_sens_flip_few,
        }

    return prompt_dict.get(prompt_name)





#print(all_cats_sens_hop3('test', 'test2'))