# Code Soup Repository Structure

For every algorithm or ingredient of our code soup we have three targets implementation, unit-tests, tutorials. Pertaining to these three we have three folders **code-soup**, **tests**, **Tutorials**

```bash
.code-soup/
+-- code_soup/ #main package
+-- tests/ #unit-tests
+-- Tutorials/ #tutorials
```
Each of these follows a book like structure as shown below


Code-Soup, main package
---

Main package to be used in fencing and build attacks / defenses

```bash
.code-soup/code_soup/
+-- common/ #Used across the package, parallel to glossary
|	+-- vision/
|	|	+-- models/ #Commonly used models for eg GPT-2
|	|	+-- utils/ #Commonly used utils like accuracy metric etc
|	|   +-- dataset/ #Datasets used in the chapter
|	+-- text/	#Same as above
|	+-- rl/		#Same as above
# For every chapter ->
+-- ch{Chapter_Number}/ #Code refering to a particular chapter
|	+-- algorithms/ #Attackers or Defenders used in the chapter
|	|	+--{Name_of_Attack/Defense}.py
# There will be exactly one file pertaining to the agents.
# This is supposed to be parallel to the pseudcode in a book.
# Therefore only model states and step functions for attack/defense should be here
```

Tests, Unit tests
---

For Unit testing of each module in the package

```bash
# Exactly same structure would be followed for the test
.code-soup/tests/
+-- test_common/
|	+-- test_vision/
|	|	+-- test_models/
|	|	+-- test_utils/
|   |   +-- test_dataset/
|	+-- test_text/
|	+-- test_rl/
|	+-- test_utils/

# For every chapter ->
+-- test_ch{Chapter_Number}/
|	+-- test_algorithms/
|	|	+--test_{Name_of_Attack/Defense}.py
```

Tutorials
---

Tutorial, Demonstration, Success, and Visualisation for each algorithm.

```bash
# Follows a similar structure
.code-soup/Tutorial
# For every chapter ->
+-- ch{Chapter_Number}
|	+-- {Name_of_Attack/Defense}/
|   |   +--{Name_of_Attack/Defense}_Tutorial.ipynb #Main Tutorial Notebbok
|   |   +--config.json #For storing hyper parameters etc
|   |   +--results.md #(optional) for storing the results obtained
```

For detailed information of particular syntax/structure followed check the readme in folders of those chapters
