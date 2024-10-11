# Method2
Trying to reduce the confusion between classes and improve accuracy in small classes.

1. Tagging the entities in texts and returned a dataframe with columns:
    - `text_index`
    - `e1_id`: id of the subject/head entity in the list of entities of the text
    - `e2_id`: id of the object/tail entity in the list of entities of the text
    - `e1_type`: subject/head entity type
    - `e2_type`: object/tail entity type
    - `text`: full text with entity mentions tagged according to their respective type.
    - `relations`: list of relations between the entities
2. Filtering to identify whether the tagged text might expressed a possible relation
    1. Rule-based filtering tagged texts that imply:
        - if head == tail, an entity type is in a unary relation in the train set
        - otherwise, an entity type pair that is in a binary relation in the train set
    2. ML-based filtering using a binary classification to distinguishes in scope and out-of scope (`oos`) relation: i.e. **does the tagged text express any relation?**
3. If the text has a relation then determine what relations are expressed:
    - if a single entity is tagged, then a unary relation extractor is used to determine what attribute of the entity is expressed
    - Otherwise, a binary relation extractor is use to identify which ones of the other relations are expressed


## Tokenizer
the input of the tokenizer should be a pair of texts ax in the task of predicting the next sentence (sentence1, sentence2):
- binary relation: (`tagged text`, `e1_type e2_type`)
- unary relation: (`tagged text`, `e1_type`)


# Difficult cases

## Unary relation i.e. e1.id == e2.id difficult to understand
example format: `[text_index]: [text extract with tagged entities]`
- PLACE IS_LOCATED_IN PLACE (6 examples): 
````
3676: Ce jour du 12 mai 2005, à <Ouagadougou>, ...  agiront bientôt sur la <capitale>.

2593: Ce matin à <Milan>,... de la <ville>.

2337: ... dans un petit <village> d’<Afrique> la nuit du ...

3898: ... notamment sa <résidence> de <Montréal>, ses bijoux ...

1113: ... se rendaient au <nord> du <Chili>, pour ...

1175: ... dans le <sud du Togo>. ... reliant le <Togo> au ...
````

- ACCIDENT	HAS_CONSEQUENCE	ACCIDENT (1 example)
````
1201: ... dû au <dysfonctionnement des freins> de sa voiture, en effet, ils ont <lâché> quand ...
````

- NON_MILITARY_GOVERNMENT_ORGANISATION	INITIATED	NON_MILITARY_GOVERNMENT_ORGANISATION (1 example)
````
1201: ... dû au <dysfonctionnement des freins> de sa voiture, en effet, ils ont <lâché> quand ...
````

- GROUP_OF_INDIVIDUALS	IS_PART_OF	GROUP_OF_INDIVIDUALS (3 examples)
````
41726: ... les <couples> ont décoré la ... par les <nouveaux mariés>. Tout ...

3712: ... Les <victimes> étaient pour la plupart des <travailleurs> de la centrale ...

158: ...  Plusieurs <corps> sans vie de <migrants> ont été repêchés ... a ordonné que les <corps> sans ...
````

- MILITARY	IS_IN_CONTACT_WITH	MILITARY (1 example)
````
3667: ... la décapitation d’un <homme> cagoulé a été diffusée à la télévision. Il s’agissait du Général <Martin Kumba>, reconnue grâce à une montre de marque qu’<il> avait à la main. ...
````


## Binary relations


# Issues

# labelling errors

- CIVILIAN GENDER_FEMALE CIVILIAN (label error, should be GENDER_MALE because of `président` and `a été blessé`)
````
3667: <Anam Destresse>, président de ... le coup. <Anam Destresse>, qui faisait
````

- CIVILIAN	CIVILIAN	['GENDER_MALE'] - error on gender
````
41922: ... Il ne cessait de crier le nom d'une < femme >, puis a appuyé sur la gâchette. ...
````

- MATERIEL	LENGTH	['HAS_FOR_LENGTH'] - missing entity mention labelling for `Il`:
````
168: ... Il mesurait [ 77,46 mètres ] de long avec une capacité de 1017 passagers. ...
````

## Spelling errors

- 3787: _... à Londres. **Malrgé** la pluie ..._

## two entities having the same mention

````
51698: ... que le { propriétaire } de la boutique...
````