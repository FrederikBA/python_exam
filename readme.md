**Automatiseret Håndskrevet Diktat Korrekturlæser**

Frederik, Janus, Rasmus


**Problemstilling**

Ideen går ud på at lave et program der kan automatisere den proces en folkeskolelærer skal igennem når diktater skal rettes og point skal tildeles eleverne.

Opgaven bygger på at udarbejde et program der ved hjælp af billedgenkendelse kan indsamle elevens svar på diktaten, hvorefter vi vil oprette et neuralt netværk, der kan genkende håndskrift og dermed holde de håndskrevne ord op mod en facitliste til den givne diktat. Vi træner det neurale netværk med en række billeder af alfabetet. 

**Datasæt**
Datasæt med billeder af alfabetet i csv format:
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format


**Teknologier**
OpenCV
Keras
Sklearn
Numpy
Pandas
Matplotlib


**Installations Guide**
Build og kør docker containeren.


**Hvordan eksekveres programmet**
Kør hele prepare_data.ipynb notebooken


**Hovedudfordringer:**

At oprette et neuralt netværk og træne en model ud fra givet data

At læse tekst ud fra billedgenkendelse (og sammensætte det genkendte til hele ord)

At plotte relevant data for at visualisere og formidle ovenstående

**Status**
Projektet er i teorien udført efter vores beskrivelse. Dog er successraten med billedgenkendelses algoritmen med eksterne billeder ikke utrolig høj.



