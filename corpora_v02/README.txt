XML structure:
Root node - its direct children are topics (10), each representing a chart.
Each topic has children (story, 23) that are individual descriptions.
Each story has 2 children: text and annotations.
Text branches out to the 1) content (story text) and 2) sentences (each sentence is child of sentences).
Annoations has only one child - events. Each child of events is a label.

Each topics, stories, tokens and labels have their own indices. Here is how to read them (with examples):
* topics: they go from "01" to "10", topic_id="01"
* stories: topic_id+story, e.g. story_id="03-07" means that this is story "07" of topic "03"
* tokens: story_id+token, e.g. token_id="04-03-2-5" marks the token that is in topic "04", story "03", sentence "2" and within it at position "5", where the position is the within-sentence index of the token, starting at 1
* events: story_id+event, e.g. id="08-18-6" is a label in topic "08", story "18" at position "6", where position is the index of the label, starting at the first label within a story at 1
