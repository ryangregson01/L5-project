def base_prompt_template(document):
  return f"""[INST]
  The Text is sensitive if it contains sensitive personal information. Note email metadata such as the headers and signatures are not regarded as sensitive personal information.
  Which one of the attributes “sensitive”, “non-sensitive” classifies the Text.
  The Text is delimited by triple backticks. You must answer after 'Attribute:'[/INST]

  Text: ```{document}```.

  Note: You must choose the attribute "sensitive" or "non-sensitive" to classify the Text and output only the attribute after 'Attribute:'.
  Attribute:
  """

def explain_base_prompt_template(document):
  return f"""[INST]
  The Text is sensitive if it contains sensitive personal information. Note email metadata such as the headers and signatures are not regarded as sensitive personal information.
  Which one of the attributes “sensitive”, “non-sensitive” classifies the Text.
  The Text is delimited by triple backticks. You must answer after 'Attribute:'[/INST]

  Text: ```{document}```.

  Note: You must choose the attribute "sensitive" or "non-sensitive" to classify the Text and output only the attribute after 'Attribute:'. You must explain the choice for this attribute.
  Attribute:
  """


def display_gen_textans(output):
  end_template = output.find('\n  <ANSWER>')
  return output[end_template:]

def display_gen_textattr(output):
  end_template = output.find('\n  Attribute:')
  return output[end_template:]


def get_prompt(prompt_name):
    prompt_dict = {'base_prompt_template' : base_prompt_template, 
                  'explain_base_prompt_template': explain_base_prompt_template}
    return prompt_dict.get(prompt_name)

