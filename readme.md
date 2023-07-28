# General overview


## TDA overview
Topological Data Analysis (TDA) is an approach to the analysis of datasets using well established techniques from topology. In what follows we try to give intuitive but far from rigorous explainations of concepts we use, the interested reader can refer to other material like this [TDA survey for data scientists](https://www.frontiersin.org/articles/10.3389/frai.2021.667963/full) for precise definitions and for a comphrensive overview of the topic.      
One of main tools of TDA is Persistence Homology (PH), which allows to encode multiscale topological features of datasets.  
The topological features we will consider are called homology groups and are topological invariants, namely they are not changed by continuous deformation of the topological space under consideration.  
To give some intuition: if you have a piece of clay and you deform it without cutting or gluing pieces together what you will get is topologically equivalent to initial object and hence have same homology.

*Example: a cup and a doughnut are topologically equivalent*  
<img src="./cup_deformation.jpg" alt="drawing" width="800"/>

  
Informally speaking, homology counts the number of i-dimensional “holes” of the topological space.  
We can define one homology group for each space dimension, in particular in what follows we will be intersted in dimensions 0, 1 and 2, where homology groups count:  
- i=0, number of connected components,    
- i=1, number of daily life "holes",  
- i=2, number of cavities. 

As toy example, a football (like a rugby ball, since these two are topologically equivalent) has:  
- one dimensional 0-th homology group (since it is connected)  
- zero dimensional 1-st homology group (if it had an hole you could not play!)  
- one dimensional 2-nd homology group (since the football is empty inside)
 
*Example: on first column are displayed examples of 1-dim and 2-dim hole, while the objects in second and third columns can be continuously deformed to a point, hence their 1-dim and 2-dim homologies are trivial*  
<img src="./holes.jpg" alt="drawing" width="800"/>  

Homology is useful to describe topological spaces, but a cloud of points is not very interesting from a topological point of view: it is just a bunch of topologically trivial connected components with no 1 and 2 dimensional holes!  
The idea behind Persistent Homology is building a family of non-trivial topological spaces originated from the point cloud and then computing the homology of these spaces. In this way PH allows the encode the evolving shape of a cloud of points.  

<img src="./cloud_points_filtration.png" alt="drawing" width="800"/>



## PH for protein encoding
In this work we use PH to get insights on topological shape of macromolecules, in particular of protein assemblies.  
We describe molecules by 3d-coordinates of their atoms and define a familiy of topological spaces using euclidean distance as filtration. In simpler words we take atoms as points in 3d space, we create balls of zero radius centered in each atom and then inflate the balls by increasing their radius. In this way we are creating an increasing family of topological spaces and PH allows us to see how the topology of these spaces evolves.

*Example: 3d representation of Epstein Barr virus gr42 protein. Edges (bonds) are useful for visualization but we will not use them for TDA encoding.*
<img src="./epstein_barr_virus_gr42_protein.png" alt="drawing" width="800"/>  


### Protein encoding pipeline
1) Query [PDB entries](https://www.rcsb.org/) of protein assemblies (those having a quaternary structure, namely those comprising two or more protein chains) to get their PDB IDS  
2) Download PDB files of protein assemblies  
3) Extract atoms 3d-coordinates from PDB files  
4) Compute PH of atoms clouds using alpha complexes and represent it as barcodes and persistence landscapes (see  [overview](https://www.frontiersin.org/articles/10.3389/frai.2021.667963/full) or [GUDHI tutorial](https://github.com/GUDHI/TDA-tutorial) for terminology)  
5) Approximate landscape curves using Fourier series  
6) Encode PH of each assembly by leveraging Fourier approximations of landscapes  


### Considerations around data and encoding
TDA can be applied to a huge variety of datasets, all it needs is a cloud of points and a filtration defining a family of increasing topological spaces, anyway not all data are equally suitable for TDA application.
#### Topological shape of data  
First questions to ask before applying TDA is whether we expect the data to have an interesting topological shape and if so, what can we do with it.  
In our case, data are 3d-objects and by using Euclidean distance as filtration we retain an intuitive geometric interpretation of protein topological shape.  
Topology of proteins with quaternary structure can be quite interesting (e.g forming rings, cages, etc, see [1](https://pubs.rsc.org/en/content/articlelanding/2016/cs/c5cs00157a) and [2](https://nanoconvergencejournal.springeropen.com/articles/10.1186/s40580-021-00294-3)) and that's the main reason for focusing on protein assemblies rather than on proteins formed by a single unit.  

*Example: haemoglobin quaternary structure is made by four subunits*
<img src="./haemoglobin.png" alt="drawing" width="600" height="400"/>  


*Rmk*: [single unit proteins can be knotted](https://www.technologyreview.com/2016/10/26/156418/the-mystery-of-knotted-proteins/), but this is not detected by PH, since all knots in 3d Euclidean space are homotopy equivalent to the unknot (circle).  

*Example: knotted single unit protein*
<img src="./knotted_protein.png" alt="drawing" width="800"/>  



#### Homology groups
Direct homology computation of the topological spaces we are interested in is usually unfeasible, nevertheless Nerve theorem tells us that the original topological space is homotopy equivalent (and hence share same homology) to spaces whose homology is easier to compute. 
Rips-Vietoris complexes are often used to transition from original to simpler space, as they enable fast computation of homology groups that approximates the ones of the original space.  
This is not optimal as sacrifices some interpretability, what we will do instead is relying on alpha complexes, for which Nerve theorem applies, without the need of approximations.  

*Example: comparison of different tools used to compute PH*
<img src="./alpha_rips_complexes.png" alt="drawing" width="800" height="400"/> 

We notice that encoding proteins via their atoms 3d-coordinates allows us to encode as much topological information as possible, without bothering to compute homology in higher dimensions, which in general is much more domputationally expensive; indeed in this setting homology is non-trivial only in dimensions 0, 1 and 2.



#### PH featurization
PH can be encoded in various equivalent ways: common choices are barcodes, persistence diagrams and persistence landscapes.  

*Example: PH representation as barcode and persistence diagram*
<img src="./persistence_representation.png" alt="drawing" width="800" height="400"/> 

Efficiently encoding PH information for ML tasks is an active research topic; in this work we use barcode representation to vectorize 0-dimensional PH, while for 1 and 2 dimensional PH we consider landscape representation as periodic curve and then be approximate it as Fourier series.  

<ins>Featurization details</ins>:  
- *dim 0:* take five largest bounded bar end points of the 0-dimensional bars (note there are as many bars as atoms and one of the bar has infinite length)  
- *dim 1:* each landscape is approximated by an order 5 Fourier series along with landscape start and end point. we restrict to the first three landscapes.  
- *dim 2:* each landscape is approximated by an order 5 Fourier series along with landscape start and end point. we restrict to the first three landscapes.

Dimension 0 encoding requires 5 variables, while dimensions 1 and 2 require 39 = 3 \* [(5 \*2 + 1) + 2] each. In total each protein is encoded by 83 = (39\*2) + 5 variables.

*Example: first three landscapes in dim 1 for PDB protein 1A1M*
<img src="./1A1M_land_dim1.png" alt="drawing" width="800" height="400"/> 


## Applications
The pipeline described above is quite general and can be used to get insights on the topological shapes of a variety of molecular 3d-structures, or in general of 3d-objects. What comes next is finding useful applications.  
The application discussed here is classification of protein assemblies topological shapes.  
To do that we define a metric on the space of vectorized proteins (the distance between two landscapes is defined in terms of area of the difference between the two curves), we compute pairwise distance between proteins and finally we fed this to a HDBSCAN clustering algorithm that creates cluster of proteins according to their shape.   

This is just one small application but there can be many, if you have any suggestions feel free to reach out! 



------- WIP ---------  

TODO (in progress): add application details.




[//]: # ( TODO: add other works on different filtrations and diff homologies...)



# Code

## Project set up

Software
All PH computations as well as persistence barcodes and landscapes visualization is done using GUDHI library.



Computational time
The most computational intensive part of the process is computing PH, which usually takes few seconds and took *** minutes on a 2.3 GHz MacBook Pro laptop from 2017 to few minutes for proteins having thousands of atoms on a single core of a laptop)






Create virtual env
Install requirements using pip from requirements.txt file



What 
 -invariant by isomteries like translation, rotation etc
 - limitations: encodes shape not 

