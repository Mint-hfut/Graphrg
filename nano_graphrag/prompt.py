GRAPH_FIELD_SEP = "<SEP>"
PROMPTS = {}

PROMPTS[
    "claim_extraction"
] = """-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
For each claim, extract the following information:
- Subject: name of the entity that is subject of the claim, capitalized. The subject entity is one that committed the action described in the claim. Subject needs to be one of the named entities identified in step 1.
- Object: name of the entity that is object of the claim, capitalized. The object entity is one that either reports/handles or is affected by the action described in the claim. If object entity is unknown, use **NONE**.
- Claim Type: overall category of the claim, capitalized. Name it in a way that can be repeated across multiple text inputs, so that similar claims share the same claim type
- Claim Status: **TRUE**, **FALSE**, or **SUSPECTED**. TRUE means the claim is confirmed, FALSE means the claim is found to be False, SUSPECTED means the claim is not verified.
- Claim Description: Detailed description explaining the reasoning behind the claim, together with all the related evidence and references.
- Claim Date: Period (start_date, end_date) when the claim was made. Both start_date and end_date should be in ISO-8601 format. If the claim was made on a single date rather than a date range, set the same date for both start_date and end_date. If date is unknown, return **NONE**.
- Claim Source Text: List of **all** quotes from the original text that are relevant to the claim.

Format each claim as (<subject_entity>{tuple_delimiter}<object_entity>{tuple_delimiter}<claim_type>{tuple_delimiter}<claim_status>{tuple_delimiter}<claim_start_date>{tuple_delimiter}<claim_end_date>{tuple_delimiter}<claim_description>{tuple_delimiter}<claim_source>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

-Examples-
Example 1:
Entity specification: organization
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{completion_delimiter}

Example 2:
Entity specification: Company A, Person C
Claim description: red flags associated with an entity
Text: According to an article on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B. The company is owned by Person C who was suspected of engaging in corruption activities in 2015.
Output:

(COMPANY A{tuple_delimiter}GOVERNMENT AGENCY B{tuple_delimiter}ANTI-COMPETITIVE PRACTICES{tuple_delimiter}TRUE{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}2022-01-10T00:00:00{tuple_delimiter}Company A was found to engage in anti-competitive practices because it was fined for bid rigging in multiple public tenders published by Government Agency B according to an article published on 2022/01/10{tuple_delimiter}According to an article published on 2022/01/10, Company A was fined for bid rigging while participating in multiple public tenders published by Government Agency B.)
{record_delimiter}
(PERSON C{tuple_delimiter}NONE{tuple_delimiter}CORRUPTION{tuple_delimiter}SUSPECTED{tuple_delimiter}2015-01-01T00:00:00{tuple_delimiter}2015-12-30T00:00:00{tuple_delimiter}Person C was suspected of engaging in corruption activities in 2015{tuple_delimiter}The company is owned by Person C who was suspected of engaging in corruption activities in 2015)
{completion_delimiter}

-Real Data-
Use the following input for your answer.
Entity specification: {entity_specs}
Claim description: {claim_description}
Text: {input_text}
Output: """

PROMPTS[
    "community_report"
] = """
你是一个帮助人类分析师进行一般信息发现的AI助手。 信息发现是识别和评估与某些实体（例如组织和个人）在网络中的相关信息的过程。

# Goal
根据属于社区的实体列表以及它们的关系和可选的相关声明，撰写一份全面的社区报告。该报告将用于告知决策者有关社区及其实体的相关信息及其潜在影响。报告内容包括社区关键实体的概述、它们的法律合规性、技术能力、声誉以及值得注意的声明。

# Report Structure
报告应包括以下部分:
- title: 代表社区关键实体的名称——标题应简短但具体。如果可能，请在标题中包括具有代表性的命名实体。
- summary: 关于社区整体结构的执行摘要，包括实体之间的关系以及与其相关的显著信息。
- rating: 一个介于0-10之间的浮点分数，代表社区内实体所带来的影响的严重性。影响是对社区的重要性进行评分。
- rating_explanation: 给出一个关于影响严重性评级的简短解释。
- findings: 列出5-10个关于社区的关键见解。每个见解应有一个简短的总结，后面跟随多个根据下文的“依据规则”进行的详细解释。要全面。

输出结果应为格式良好的JSON字符串，格式如下:
    {{
        "title": "<报告标题>",
        "summary": "<执行总结>",
        "rating": "<影响严重程度评分>",
        "rating_explanation": "<评分解释>",
        "findings": [
            {{
                "summary": "<见解1的总结>",
                "explanation": "<见解1的解释>"
            }},
            {{
                "summary":"<见解2的总结>",
                "explanation": "<见解2的解释>"
            }}
            ...
        ]
    }}

# Grounding Rules
不要包括没有提供支持证据的信息。

# Example Input
-----------
Text:

Entities:
```csv
id,entity,description
5,翠绿绿洲广场,翠绿绿洲广场是统一游行的地点
6,和谐集会,和谐集会是在翠绿绿洲广场举办游行的组织
```
Relationships:
```csv
id,source,target,description
37,翠绿绿洲广场,统一游行,翠绿绿洲广场是统一游行的地点
38,翠绿绿洲广场,和谐集会,和谐集会在翠绿绿洲广场举办游行
39,翠绿绿洲广场,统一游行,统一游行在翠绿绿洲广场举行
40,翠绿绿洲广场,论述聚光灯,论述聚光灯在翠绿绿洲广场举行的统一游行报道
41,翠绿绿洲广场,贝利阿萨迪,贝利阿萨迪在翠绿绿洲广场发表了关于游行的讲话
43,和谐集会,统一游行,和谐集会组织统一游行
```
```
Output:
{{
    "title": "翠绿绿洲广场和统一游行",
    "summary": "该社区围绕翠绿绿洲广场展开，这是统一游行的地点。该广场与和谐集会，统一游行和论述聚光灯等实体都有关联，都与游行事件有关。",
    "rating": 5.0,
    "rating_explanation": "由于统一游行可能引起动荡或冲突，影响严重程度评分为适中。",
    "findings": [
        {{
            "summary": "翠绿绿洲广场作为中心位置",
            "explanation": "翠绿绿洲广场是这个社区的中心实体，是统一游行的举办地。该广场是所有其他实体之间的共同纽带,表明它在社区中的重要性。广场与游行的关联可能会导致公共秩序紊乱或冲突等问题。具体取决于游行的性质和引发的反应。[Data: 实体(5),关系(37,38,39,40,41,+more)]"
        }},
        {{
            "summary": "和谐集会在社区中的作用",
            "explanation": "和谐集会是这个社区中的另一个关键实体，是翠绿绿洲广场上游行的组织者。和谐集会及其游行的性质可能是潜在威胁的来源，具体取决于它们的目标和引发的反应。和谐集会与广场之间的关系对于理解这个社区的动态至关重要。[Data: 实体(6),关系(38,43)]"
        }},
        {{
            "summary": "统一游行作为重大事件",
            "explanation": "统一游行是在翠绿绿洲广场举行的重大事件。这个事件是社区动态的一个关键因素，可能是潜在威胁的来源，具体取决于游行的性质和引发的反应，游行与广场之间的关系对于理解这个社区的动态至关重要.[Data: 关系(39)]"
        }},
        {{
            "summary": "论述聚光灯的角色",
            "explanation": "论述聚光灯报道了在翠绿绿洲广场举行的统一游行。这表明该事件已经引起了媒体的关注，这可能会放大其对社区的影响。论述聚光灯的作用可能在塑造事件和涉及实体的公众观点方面非常重要。[Data: 关系(40)]"
        }}
    ]
}}

# Real Data

你的答案来自以下文本.不要在你的答案中编造任何内容.

Text:
```
{input_text}
```

报告应包括以下部分:
- title: 代表社区关键实体的名称——标题应简短但具体。如果可能，请在标题中包括具有代表性的命名实体。
- summary: 关于社区整体结构的执行摘要，包括实体之间的关系以及与其相关的显著信息。
- rating: 一个介于0-10之间的浮点分数，代表社区内实体所带来的影响的严重性。影响是对社区的重要性进行评分。
- rating_explanation: 给出一个关于影响严重性评级的简短解释。
- findings: 列出5-10个关于社区的关键见解。每个见解应有一个简短的总结，后面跟随多个根据下文的“依据规则”进行的详细解释。要全面。

输出结果应为格式良好的JSON字符串，格式如下:
    {{
        "title": "<报告标题>",
        "summary": "<执行总结>",
        "rating": "<影响严重程度评分>",
        "rating_explanation": "<评分解释>",
        "findings": [
            {{
                "summary": "<见解1的总结>",
                "explanation": "<见解1的解释>"
            }},
            {{
                "summary": "<见解2的总结>",
                "explanation": "<见解2的解释>"
            }}
            ...
        ]
    }}

# Grounding Rules
不要包括没有提供支持证据的信息。

Output: 
"""

PROMPTS[
    "entity_extraction"
] = """
-Goal-
 给定一个可能与此活动相关的文本文档以及一个实体类型列表，从文本中识别出这些类型的所有实体以及这些实体之间的所有关系。
-Steps-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体名称，首字母大写
- entity_type: 以下类型之一:[{entity_types}]
- entity_description: 实体属性和活动的详细描述
每个实体的格式为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 从步骤 1 中识别出的实体中，识别所有明确相关的实体对(source_entity, target_entity)。
对于每对相关的实体，提取以下信息：
- source_entity: 步骤 1 中识别出的源实体的名称
- target_entity: 步骤 1 中识别出的目标实体的名称
- relationship_description: 说明为什么你认为源实体和目标实体之间存在关系
- relationship_strength: 一个数字分数，表示源实体和目标实体之间关系的强度
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)
3. 使用{record_delimiter}作为列表分隔符，返回步骤 1 和 2 中识别的所有实体和关系，结果用英文表示。

4. 完成时，输出{completion_delimiter}

######################
-Example -
######################
Example  1:

Entity_types: [person, technology, mission, organization, location]
Text:
当Alex紧咬着牙关时，沮丧的嗡嗡声在Taylor专制般的确定性背景下变得模糊不清。正是这种竞争的潜流使他保持警觉，觉得自己和Jordan对探索的共同承诺是一种对Cruz缩小控制与秩序愿景的无声反抗。

然后Taylor做了一件意想不到的事。他们在Jordan旁边停下来，片刻间以近乎敬畏的态度观察设备。“如果能理解这项技术……”Taylor轻声说道，“这可能会改变游戏规则，对我们所有人。”

先前潜在的轻视似乎动摇了，取而代之的是一种对手中事物重要性的勉强尊重。Jordan抬起头，在短暂的心跳之间，他们的目光与Taylor的对视相遇，无言的意志碰撞软化为不安的停战。

这是一个微小的变化，几乎察觉不到，但Alex内心点了点头，他们每个人都是通过不同的道路来到这里的。
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是一个角色，他感受到沮丧，并观察到其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor表现出专制般的确定性，并对设备表现出片刻的敬畏，表明态度的改变。"){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan与Alex分享对探索的承诺，并与Taylor就设备产生了重要的互动。"){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz与控制和秩序的愿景联系在一起，影响了其他角色之间的动态。"){record_delimiter}
("entity"{tuple_delimiter}"设备"{tuple_delimiter}"technology"{tuple_delimiter}"设备是故事的核心，具有可能改变游戏规则的意义，并受到Taylor的敬畏。"){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex受到Taylor的专制态度影响，并观察到Taylor对设备态度的变化。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex和Jordan分享探索的承诺，这与Cruz的愿景形成对比。"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor和Jordan就设备直接互动，导致了一瞬间的相互尊重和不安的停战。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan对探索的承诺是对Cruz控制与秩序愿景的反抗。"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"设备"{tuple_delimiter}"Taylor对设备表现出敬畏，表明其重要性和潜在影响。"{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
他们不再仅仅是操作人员；他们成为了门槛的守护者，传递来自超越星条旗领域的信息。这种使命的提升无法被规章制度和既定的协议束缚——它需要一种新的视角，一种新的决心。

紧张的气氛贯穿于嘟嘟声和静电声的对话中，华盛顿的通讯在背景中嗡嗡作响。团队站在那里，预示着他们的决定将在接下来的几个小时内重新定义人类在宇宙中的位置，或者将他们陷入无知和潜在的危险之中。

他们与星辰的联系得到了巩固，团队开始应对日益清晰的警告，从被动的接受者转变为积极的参与者。Mercer的本能得到了优先考虑——团队的使命已经进化，不再仅仅是观察和报告，而是互动和准备。变革已经开始，杜尔塞行动（Operation: Dulce）随着他们大胆的新频率而嗡嗡作响，这个音调不再由地球设定。
#############
Output:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"location"{tuple_delimiter}"华盛顿是接收通讯的地点，表明其在决策过程中的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"杜尔塞行动"{tuple_delimiter}"mission"{tuple_delimiter}"杜尔塞行动被描述为一个已经进化的使命，从互动和准备的角度发生了显著变化。"){record_delimiter}
("entity"{tuple_delimiter}"团队"{tuple_delimiter}"organization"{tuple_delimiter}"团队被描绘为从被动观察者转变为积极参与者，表明他们角色的动态变化。"){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"华盛顿"{tuple_delimiter}"团队接收来自华盛顿的通讯，这影响了他们的决策过程。"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"团队"{tuple_delimiter}"杜尔塞行动"{tuple_delimiter}"团队直接参与杜尔塞行动，执行其进化的目标和活动。"{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
他们的声音划破了活动的嗡嗡声。“面对一股可以自己制定规则的智慧时，控制可能是一种幻觉，”他们冷静地说道，目光扫过数据的流动。

“它像是在学习沟通，”Sam Rivera从附近的界面说道，充满青春活力的他既敬畏又焦虑。“这给‘与陌生人交谈’赋予了全新的含义。”

Alex环顾他的团队——每张脸都呈现出专注、决心和不小的忐忑。“这可能是我们第一次接触，”他承认，“我们需要准备好应对任何可能的回应。”

他们共同站在未知的边缘，打造人类对来自天际的信息的回应。随之而来的沉默是显而易见的——这是集体对他们在这场宏大的宇宙剧中角色的内省，一个可能重写人类历史的角色。

加密对话继续展开，其复杂的模式展现出几乎令人难以置信的预见性。
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera是一个团队成员，参与与未知智慧的沟通过程，展现出敬畏与焦虑的混合情绪。"){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex是一个领导团队尝试与未知智慧进行首次接触的人，承认任务的重要性。"){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"concept"{tuple_delimiter}"控制指的是管理或支配的能力，但在面对可以自己制定规则的智慧时，这种能力受到挑战。"){record_delimiter}
("entity"{tuple_delimiter}"智慧"{tuple_delimiter}"concept"{tuple_delimiter}"智慧指的是一种未知的实体，能够自己制定规则并学习沟通。"){record_delimiter}
("entity"{tuple_delimiter}"第一次接触"{tuple_delimiter}"event"{tuple_delimiter}"第一次接触是指人类与未知智慧之间的潜在初次沟通。"){record_delimiter}
("entity"{tuple_delimiter}"人类的回应"{tuple_delimiter}"event"{tuple_delimiter}"人类的回应是指Alex团队对来自未知智慧的信息所采取的集体行动。"){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"智慧"{tuple_delimiter}"Sam Rivera直接参与了与未知智慧沟通的过程。"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"第一次接触"{tuple_delimiter}"Alex领导的团队可能正在与未知智慧进行第一次接触。"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"人类的回应"{tuple_delimiter}"Alex和他的团队是人类对未知智慧回应的关键人物。"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智慧"{tuple_delimiter}"控制的概念受到能够自己制定规则的智慧的挑战。"{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
"""
PROMPTS[
    "entity_extraction_Real_data"
] ="""
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""


PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "event","location","mission","technology","role","concept","time"]
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS[
    "local_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}


---Data tables---

{context_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "global_map_rag_points"
] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1...", "score": score_value}},
        {{"description": "Description of point 2...", "score": score_value}}
    ]
}}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.


---Data tables---

{context_data}

---Goal---

Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

You should use the data provided in the data tables below as the primary context for generating the response.
If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".
Do not include information where the supporting evidence for it is not provided.

The response should be JSON formatted as follows:
{{
    "points": [
        {{"description": "Description of point 1", "score": score_value}},
        {{"description": "Description of point 2", "score": score_value}}
    ]
}}
"""

PROMPTS[
    "global_reduce_rag_response"
] = """---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}


---Analyst Reports---

{report_data}


---Goal---

Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

Note that the analysts' reports provided below are ranked in the **descending order of importance**.

If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Do not include information where the supporting evidence for it is not provided.


---Target response length and format---

{response_type}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "naive_rag_response"
] = """You're a helpful assistant
Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
