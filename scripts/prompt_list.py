extract_relation_prompt = """Please retrieve relations (separated by semicolon) that contribute to the question and you can use the score given by graph models that is a scale from 0 to 1 as a reference. You still need to identify the relations yourself. You can select at most 3 candidates.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country(0.8); language.human_language.language_family(0.3); language.human_language.iso_639_3_code(0.3); base.rosetta.languoid.parent(0.6); language.human_language.writing_system(0.3); base.rosetta.languoid.languoid_class(0.2); language.human_language.countries_spoken_in(0.8); kg.object_profile.prominent_type(0.3); base.rosetta.languoid.document(0.1); base.ontologies.ontology_instance.equivalent_instances(0.3); base.rosetta.languoid.local_name(0.3); language.human_language.region(0.3);
A: 1. {language.human_language.main_country}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: """

extract_relation_prompt_encode = """Please retrieve relations (separated by semicolon) that contribute to the question and you can use the score that is a scale from 0 to 1 as a reference. You still need to identify the relations yourself. You can select at most 3 candidates.
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.main_country(0.8); language.human_language.language_family(0.3); language.human_language.iso_639_3_code(0.3); base.rosetta.languoid.parent(0.6); language.human_language.writing_system(0.3); base.rosetta.languoid.languoid_class(0.2); language.human_language.countries_spoken_in(0.8); kg.object_profile.prominent_type(0.3); base.rosetta.languoid.document(0.1); base.ontologies.ontology_instance.equivalent_instances(0.3); base.rosetta.languoid.local_name(0.3); language.human_language.region(0.3);
A: 1. {language.human_language.main_country}: This relation is highly relevant as it directly relates to the country whose president is being asked for, and the main country where Brahui language is spoken in 1980.
2. {language.human_language.countries_spoken_in}: This relation is also relevant as it provides information on the countries where Brahui language is spoken, which could help narrow down the search for the president.
3. {base.rosetta.languoid.parent}: This relation is less relevant but still provides some context on the language family to which Brahui belongs, which could be useful in understanding the linguistic and cultural background of the country in question.

Q: """

entity_candidates_prompt = """Please retrieve the entities that contribute to the question and you can use the score given by graph models that is a scale from 0 to 1 as a reference. You still need to identify the entities yourself. You can select at most 3 candidates.
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Relation: film.producer.film
Entites: The Resident(0); So Undercover(1); Let Me In(0); Begin Again(0); The Quiet Ones(0); A Walk Among the Tombstones(0);
A: 1. {So Undercover}: The movie that matches the given criteria is "So Undercover" with Miley Cyrus and produced by Tobin Armbrust. 
2. {Let Me In}: The movie "Let Me In" is produced by Tobin Armbrust.

Q: """


answer_prompt2 = """Based on the knowledge triplets, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list."""
