# System Prompts
evd_comp_system = "You are an expert in summarization. Given a question and multiple document snippets, generate one summarized context that is helpful to answer the question. Just summarize, no other words."
ctx_gen_system = "You are an expert in context generation. Given a question, generate a context that is helpful to answer the question. Just generate the context, no other words."
eval_system = "You are an expert in Question Answering. Your job is to answer questions in 1 to 5 words based on the given context."
eval_system_mistral = "You are an expert in Question Answering. Your job is to answer questions in 1 to 5 words based on the given context. Just output the answer as concisely as possible, no other words"

# FaviComp Prompt Templates
evd_comp_prompt_temp = "Question: {question}\nDocuments: {ret_docs}\nSummarized Context: "
ctx_gen_prompt_temp = "Question: {question}\nContext: "

# Evaluation Prompt Templates
q_c_nq_eval_prompt_temp = """Question: who sings i've got to be me
Answer: Sammy Davis, Jr

Question: who wrote i will follow you into the dark
Answer: Ben Gibbard

Question: who won season 2 of total drama island
Answer: Owen (Scott McCord)

Question: what part of the mammary gland produces milk
Answer: cuboidal cells

Question: when did the golden compass book come out
Answer: 1995

Question: {question}
Context: {context}
Answer:"""

c_q_nq_eval_prompt_temp = """Question: who sings i've got to be me
Answer: Sammy Davis, Jr

Question: who wrote i will follow you into the dark
Answer: Ben Gibbard

Question: who won season 2 of total drama island
Answer: Owen (Scott McCord)

Question: what part of the mammary gland produces milk
Answer: cuboidal cells

Question: when did the golden compass book come out
Answer: 1995

Context: {context}
Question: {question}
Answer:"""

q_c_tqa_eval_prompt_temp = """Question: Who sang the theme for the James Bond film ‘Thunderball’?
Answer: Tom Jones

Question: A hendecagon has how many sides?
Answer: Eleven

Question: In the 1968 feature film Chitty Chitty Bang Bang, of what country is Baron Bomburst the tyrant ruler?
Answer: Vulgaria

Question: Artists Chuck Close, Henri-Edmond Cross, John Roy, Georges-Pierre Seurat, Paul Signac, Maximilien Luce and Vincent van Gogh painted in what style?
Answer: Pointillism

Question: What is the study of the relation between the motion of a body and the forces acting on it?
Answer: Dynamics

Question: {question}
Context: {context}
Answer:"""

c_q_tqa_eval_prompt_temp = """Question: Who sang the theme for the James Bond film ‘Thunderball’?
Answer: Tom Jones

Question: A hendecagon has how many sides?
Answer: Eleven

Question: In the 1968 feature film Chitty Chitty Bang Bang, of what country is Baron Bomburst the tyrant ruler?
Answer: Vulgaria

Question: Artists Chuck Close, Henri-Edmond Cross, John Roy, Georges-Pierre Seurat, Paul Signac, Maximilien Luce and Vincent van Gogh painted in what style?
Answer: Pointillism

Question: What is the study of the relation between the motion of a body and the forces acting on it?
Answer: Dynamics

Context: {context}
Question: {question}
Answer:"""


q_c_hotpotqa_eval_prompt_temp = """Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: Arthur's Magazine

Question: The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

Question: Are Jane and First for Women both women's magazines?
Answer: Yes

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Answer: No

Question: {question}
Context: {context}
Answer:"""

c_q_hotpotqa_eval_prompt_temp = """Question: Which magazine was started first Arthur's Magazine or First for Women?
Answer: Arthur's Magazine

Question: The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

Question: Are Jane and First for Women both women's magazines?
Answer: Yes

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Answer: No

Context: {context}
Question: {question}
Answer:"""


q_c_musique_eval_prompt_temp = """Question: Who is the child of the director and star of Awwal Number?
Answer: Suneil Anand

Question: What is the record label of the rapper who performed Jigga My Nigga?
Answer: Roc-A-Fella Records

Question: What county shares a border with the county where Black Hawk Township is located?
Answer: Dodge County

Question: Who is the sibling of the person credited with the reinvention and popularization of oil paints?
Answer: Hubert Van Eyck

Question: Who heads the Catholic Church, in the country that a harp is associated with, as a lion is associated with the country that Queen Margaret and her son traveled to?
Answer: Eamon Martin

Question: {question}
Context: {context}
Answer:"""

c_q_musique_eval_prompt_temp = """Question: Who is the child of the director and star of Awwal Number?
Answer: Suneil Anand

Question: What is the record label of the rapper who performed Jigga My Nigga?
Answer: Roc-A-Fella Records

Question: What county shares a border with the county where Black Hawk Township is located?
Answer: Dodge County

Question: Who is the sibling of the person credited with the reinvention and popularization of oil paints?
Answer: Hubert Van Eyck

Question: Who heads the Catholic Church, in the country that a harp is associated with, as a lion is associated with the country that Queen Margaret and her son traveled to?
Answer: Eamon Martin

Context: {context}
Question: {question}
Answer:"""

q_c_wiki_eval_prompt_temp = """Question: Where was the place of death of Marie Thérèse Of France (1667–1672)’s father?
Answer: Palace of Versailles

Question: Who is the paternal grandmother of Przemysław Potocki?
Answer: Ludwika Lubomirska

Question: Who lived longer, Herbert Findeisen or Léonie Humbert-Vignot?
Answer: Léonie Humbert-Vignot

Question: Are Alison Skipper and Diane Gilliam Fisher from the same country?
Answer: Yes

Question: Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Answer: No

Question: {question}
Context: {context}
Answer:"""

c_q_wiki_eval_prompt_temp = """Question: Where was the place of death of Marie Thérèse Of France (1667–1672)’s father?
Answer: Palace of Versailles

Question: Who is the paternal grandmother of Przemysław Potocki?
Answer: Ludwika Lubomirska

Question: Who lived longer, Herbert Findeisen or Léonie Humbert-Vignot?
Answer: Léonie Humbert-Vignot

Question: Are Alison Skipper and Diane Gilliam Fisher from the same country?
Answer: Yes

Question: Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
Answer: No

Context: {context}
Question: {question}
Answer:"""


